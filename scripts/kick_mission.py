"""BallKick mission: go to the ball, line up, swap in the kick policy, kick, swap back.

Run from the microduck_rl repo root:
  uv run python scripts/kick_mission.py <velstand.onnx> <kick_right.onnx> [--runs 5]
  uv run python scripts/kick_mission.py <velstand.onnx> <kick_right.onnx> --viewer [--ball 1.2,0.3 --kick-heading 30]

Phases (one control loop at 50 Hz, VelStand driving except during KICK):
  NAVIGATE  go-to-point supervisor (scripts/goto_supervisor.py, unchanged) to a
            point on the kick line behind the stance spot, via a side detour if
            the straight path would cross the ball.
  ALIGN     pulse-and-settle fine alignment until the ball sits at the kick
            policy's trained offset in the duck's yaw frame (0.09, -0.042)
            within +-ALIGN_TOL and the heading is within ALIGN_HEADING_TOL of
            the kick direction. Measured round-3 pulse responses (from rest,
            settled): pivot 0.8 rad/s ~0.8 deg/tick with ~no translation;
            strafe 0.15 m/s x5 ticks ~ +23 mm left / -11 mm right; forward
            0.2 m/s <= 10 ticks shuffles 3-4 mm, 15 ticks is one ~38 mm step;
            backward barely moves (-7 mm at 20 ticks).
  KICK      hard ONNX swap to the kick policy (same 61-D buffer, all-zero
            command as in kick training), held KICK_HOLD_S, then swap back.
  SETTLE    VelStand, zero twist. A fall here gets the go-to-point fall rule:
            zero twist until upright 1 s.
With --precision <onnx> the final approach is the PrecisionStep policy instead:
  PRECISE_PIVOT  VelStand pivots onto the kick line (|heading err| < 8 deg; the
                 step policy trained on +-15 deg).
  PRECISE        hard swap to the step policy; the 3 twist slots carry the
                 remaining offset to the stance pose [dx, dy, dyaw] in the trunk
                 yaw frame (option A). On target (+-12 mm, 3 deg) and still for
                 0.5 s -> KICK directly.
  The pulse ALIGN stays as the FALLBACK: a 6 s timeout or a fall in PRECISE swaps
  VelStand back in and hands over to it; a bumped ball re-enters PRECISE at most
  twice. Every mission records its approach_mode.
ALIGN uses its own clamps (|vx| <= 0.2, |vy| <= 0.15, pulses <= 20 ticks, only
within ALIGN_ZONE_M of the stance): go-to-point's forward-only / no-strafe
clamps can't place the duck to +-15 mm.
"""
import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_policy as ip  # noqa: E402
from goto_point import CONTROL_DT, RunLog  # noqa: E402
from goto_supervisor import GotoConfig, GotoSupervisor, wrap_angle  # noqa: E402
from kick_battery import BALL_OFFSET_ABS_Y, BALL_OFFSET_X, SCENES, KickSim  # noqa: E402
from soccer_referee import Goal, GoalReferee  # noqa: E402
from pose_provider import ODOM_CALIBS, STALE_S, GroundTruthBall, GroundTruthProvider, make_provider, pose_error  # noqa: E402

# Rung 6 Stage 2: the supervisor reads pose and ball only through providers (pose_provider.py);
# ground truth is the scoring reference (referee, ball_travel, peak, falls in KICK/SETTLE). While
# the pose is ABSENT (older than STALE_S) the duck holds still (zero twist, no guessed motion) and
# picks up where it was when the pose comes back; absent for longer than this ends the mission
# cleanly as aborted "pose_lost".
POSE_LOST_ABORT_S = 5.0

# GoalKick v0 (rung 5, Option A: frozen policies, the goal is a mission feature). With
# --goal-dist D a goal (scene_allcollisions_ball_goal.xml, 210 mm opening) stands D m past
# the ball on the kick line; the duck kicks along the ball->goal line with the per-foot
# --calib aim, and soccer_referee.GoalReferee scores goal / post / miss / no_cross.
GOAL_TIMEOUT_S = 10.0   # from the kick swap-in; the frictionless ball rolls ~0.17-0.25 m/s after a kick
GOAL_APPROACH_DEG = (0, 45, -45, 90, -90)   # duck walk-in vs ball->goal line; never from behind the goal

ARENA = (-2.5, 2.5, -2.5, 2.5)
PRE_STANCE_M = 0.12        # approach point this far behind the stance spot, on the kick line
LINE_UP_M = 0.40           # a second waypoint further back, so the last leg runs down the kick line
DETOUR_CLEAR_M = 0.25      # straight path closer than this to the ball -> side detour
ALIGN_ZONE_M = 0.30        # ALIGN clamps only apply this close to the stance spot
ALIGN_TOL = 0.012          # m, lateral: inside the kick policy's +-15 mm training noise
ALIGN_X_WINDOW = (-0.020, 0.012)  # m, along the kick line (+ = ball farther than nominal), see Aligner.pulse
BALL_WAIT_S = 6.0          # max wait for a bumped ball to stop (the sim ball has ~no rolling resistance)
APPROACH_FROM = 0.06       # m of along-line error above which ALIGN walks in continuously
APPROACH_STOP = 0.012      # cut the walk here (trunk-relative); it lands ~0-20 mm further on
STEP_MIN = 0.045           # a 15-tick step's swing reaches the ball from closer than this (it kicked it 0.2-1.9 m)
MAX_BACKOFFS = 2          # then correct in place rather than loop
BACKOFF_TO = 0.08          # backing up (continuous, it works; short back pulses don't) until this far out
ALIGN_HEADING_TOL = math.radians(3)
ALIGN_SETTLE_TICKS = 40    # 0.8 s standing between pulses
ALIGN_MAX_PULSES = 40
SQUARE_TOL = float(os.environ.get("KICK_SQUARE_TOL", "0.015"))  # kicking-foot x vs HOME before a swap-in
SUPPORT_TOL = float(os.environ.get("KICK_SUPPORT_TOL", "0.012"))  # support-foot x vs HOME
SQUARE_MAX = 3
# Squaring move: strafe left 5 ticks then right 10 (settling between). Branched from the
# real staggered stances of failed missions, it brought the support foot to within
# 1.3 mm of HOME and the kicking foot to 7-8 mm in all three; back 20 only fixed a
# kicking foot that was behind, pivot pairs barely changed the stance, and 12-tick
# forward pulses knocked the ball. It shifts the duck ~3 cm sideways, which the
# lateral pulses then take back out.
SQUARE_MOVE = [((0.0, 0.15, 0.0), 5), ((0.0, -0.15, 0.0), 10)]
KICK_HOLD_S = 2.0
SETTLE_S = 2.0
# Precision-step final approach (--precision <onnx>; Mjlab-PrecisionStep-Flat-MicroDuck).
# The pulse aligner above stays as the FALLBACK: a precision timeout, a fall, or
# too many re-entries hands the last stretch back to it.
PRECISE_PIVOT_TOL = math.radians(8)   # VelStand pivots first: the step policy trained on +-15 deg
PRECISE_PIVOT_MAX_S = 5.0
PRECISE_STILL_V = 0.02                # m/s trunk speed counted as "stopped"
PRECISE_HOLD_TICKS = 25               # 0.5 s on target and still -> kick
PRECISE_TIMEOUT_S = 6.0               # = the training episode length
PRECISE_MAX_ENTRIES = 2               # (re-)entries after a ball bump, then pulses
PRECISE_MAX_R = 0.25                  # offset clipped to the training range
# Camera channel (rung 6 Stage 3). Head pose is a VelStand command (trained to +-1.1 rad pitch).
# Measured (face-board-v1, standing): head neutral sees the ball from ~0.4 m out; head pitch +1.1
# points the camera 53 deg down and sees 0.15-0.30 m at 1-3.5 mm ray->plane error, which covers
# the pre-stance point (ball ~0.21 m ahead).
LOOK_HEAD = (0.0, 1.1, 0.0, 0.0)      # neck_pitch, head_pitch, head_yaw, head_roll
LOOK_SETTLE_S = 0.8                   # head down, wait for it to arrive and the trunk to settle
LOOK_N, LOOK_SPACING_S = 3, 0.1       # looks per re-observe, averaged
LOOK_RESTORE_S = 0.6                  # head back to neutral before the step policy engages
RESCAN_HEAD_YAWS = (0.0, 0.5, -0.5)   # re-scan: look again, then left, then right
RESCAN_MAX = 3
SCAN_WZ, SCAN_MAX_S = 0.6, 15.0       # no ball at start: turn in place looking for it
CAM_REBASE_TOL = 0.06                 # a re-look within this of the held estimate is drift, not a bump
CAM_BALL_SPEED_TOL = 0.05             # m/s from successive looks (noise ~2 mm per 0.1-0.2 s)
PRECISE_MAX_YAW = math.radians(15)
# Calibration (2026-11-11, model_3999, seeds 0+1, 60 missions): the step policy parks the
# trunk on target but the feet end ~5 mm behind HOME, so measured from the kicking foot the
# ball sat +10.2 mm (median; 0.3-14.2) farther than the kick trained on and rolled a median
# 1.01 m (the pulse aligner, which parks at +0.7 mm, rolls 1.22 m). The stance target handed
# to the step policy moves this far forward along the kick line. Precision path only: the
# pulse fallback keeps its own (foot-relative) target.
# Per kicking foot: each foot's kick checkpoint is its own measurement.
# LEFT (2026-11-12, BallKick left ts0.25 model_1250, seeds 0+1, 60 missions): 0 by decision.
# Unshifted, the ball sits +9.4 mm far of the kick spot but already rolls 1.19 m (60/60,
# 0 bumps). +9 mm was tested and REJECTED: 2/60 bumps, the step policy's stepping foot clipped
# the ball during PRECISE (PrecisionStep trained with the ball on the right side only).
#
# Calibrations are properties of a kick CHECKPOINT, so each kick generation has its own table
# (--calib v1|v2, default v1). Env vars still override any table value.
# v2 (BallKick v2, 2026-11-13, through PrecisionStep v2 model_2500, seeds 0+1, 60 missions):
#   RIGHT (ball_kick_v2_right model_1250): stance 0 by decision (Dillon). Unshifted the ball sits
#   +7.4 mm far; +7 mm put it on the spot and rolled 0.87 -> 0.99 m but raised the peak ball speed
#   0.37 -> 0.41 m/s (31/60 over 0.40), against v2's overshoot goal. Drift at 0 mm: +6.7 deg
#   median (std 3.1, flat across distance: +6.8/+6.8/+6.3) -> aim -6.7 -> -0.1 deg median.
#   LEFT (ball_kick_v2_left model_500): stance 0 by decision (Dillon). Unshifted the ball sits
#   +10.5 mm far; +9 mm (bump-free under PrecisionStep v2) tightened the spread (std 2.4 -> 1.8)
#   and rolled 0.81 -> 1.00 m but raised the peak 0.365 -> 0.44 m/s (57/60 over 0.40). Drift at
#   0 mm: -12.2 deg (RIGHT; std 2.8, flat: -12.8/-12.2/-11.3) -> aim +12.2 -> -0.1 deg median.
CALIBRATIONS = {  # (stance shift m, aim offset deg) per kicking foot
    "v1": {"right": (0.010, -17.9), "left": (0.0, 11.2)},
    "v2": {"right": (0.0, -6.7), "left": (0.0, 12.2)},
}
PRECISE_STANCE_SHIFT_M = {"right": float(os.environ.get("KICK_STANCE_SHIFT", "0.010")),
                          "left": float(os.environ.get("KICK_STANCE_SHIFT_LEFT", "0.0"))}
# Aim offset for BallKick v1's (right foot, ts0.25 model_1000) left drift. The duck lines up
# facing kick_heading + KICK_AIM_OFFSET; the ball result is still measured against kick_heading.
# Sign: +yaw is LEFT (CCW, z up), kick_dir_err_deg > 0 means the ball went left, so the offset
# is NEGATIVE (aim right). Calibrated 2026-11-11 against the +10 mm stance above: drift
# +17.9 deg median (std 1.7, IQR 16.5..19.1, flat across ball distance), 60 missions, seeds 0+1.
# LEFT (BallKick left ts0.25 model_1250, 2026-11-12): drifts RIGHT, -11.2 deg median (std 1.3,
# IQR -12.3..-10.3, flat across ball distance) at the unshifted stance, so the offset is
# POSITIVE (aim left). 60 missions, seeds 0+1.
KICK_AIM_OFFSET = {"right": math.radians(float(os.environ.get("KICK_AIM_OFFSET_DEG", "-17.9"))),
                   "left": math.radians(float(os.environ.get("KICK_AIM_OFFSET_DEG_LEFT", "11.2")))}
_CALIB_ENV = {"right": ("KICK_STANCE_SHIFT", "KICK_AIM_OFFSET_DEG"),
              "left": ("KICK_STANCE_SHIFT_LEFT", "KICK_AIM_OFFSET_DEG_LEFT")}


def apply_calibration(calib):
    """Load one kick generation's table into the per-foot dicts (env vars still win)."""
    for foot, (shift, aim) in CALIBRATIONS[calib].items():
        env_s, env_a = _CALIB_ENV[foot]
        PRECISE_STANCE_SHIFT_M[foot] = float(os.environ.get(env_s, shift))
        KICK_AIM_OFFSET[foot] = math.radians(float(os.environ.get(env_a, aim)))


def clamp_align(vx, vy, wz):
    return (min(max(vx, -0.2), 0.2), min(max(vy, -0.15), 0.15), min(max(wz, -1.0), 1.0))


class Aligner:
    """Chooses one pulse at a time from the settled ball-in-duck-frame error."""

    def __init__(self, foot="right"):
        self.target = np.array([BALL_OFFSET_X, -BALL_OFFSET_ABS_Y if foot == "right" else BALL_OFFSET_ABS_Y])
        self.last_axis, self.last_err, self.gain = None, None, 1.0

    def errors(self, pose, ball_xy, kick_heading, foot_local=None, home_foot=None):
        """Ball placement error in the duck's yaw frame (+ ex: ball too far ahead, + ey: too far left).

        With the kicking foot's position (foot_local, and where it sits at HOME), the
        error is measured foot-relative: after walking and pulses the feet can end up
        staggered by 2+ cm, and a trunk-relative fit then misses the kick (measured:
        a 23 mm foot stagger turned an in-tolerance trunk fit into a whiff).
        """
        x, y, yaw = pose
        d = np.asarray(ball_xy) - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        if foot_local is not None:
            ex, ey = (local - foot_local) - (self.target - home_foot)
        else:
            ex, ey = local - self.target
        return ex, ey, wrap_angle(yaw - kick_heading), local

    def _escalate(self, axis, err):
        """Pulse length multiplier: grows while the same axis keeps making < 30% progress.

        The measured pulse table is from a square stance; with the feet split a short
        pivot can do nothing at all (seen: 36 identical 4-tick pivots, heading stuck).
        """
        if axis == self.last_axis and abs(err) > 0.7 * abs(self.last_err):
            self.gain = min(self.gain * 1.6, 4.0)
        else:
            self.gain = 1.0
        self.last_axis, self.last_err = axis, err
        return self.gain

    def pulse(self, ex, ey, eh, allow_backoff=True):
        """(twist, ticks, axis) or None when aligned."""
        if allow_backoff and ex < 0.0 and (abs(ey) > 0.020 or abs(eh) > math.radians(15)):
            # Ball already closer than nominal and a big sideways/heading fix to make:
            # the strafe or long pivot sweeps the kicking foot through the ball (5 of 7
            # bumps in 60 missions). Back off and walk in again instead. (Heading only
            # past 15 deg: walk-ins routinely end 8-10 deg off, and backing off for
            # that looped forever.)
            return None, 0, "backoff"
        if abs(eh) > ALIGN_HEADING_TOL:
            g = self._escalate("heading", eh)
            k = int(np.clip(round(g * math.degrees(abs(eh)) / 0.8), 2, 20))
            return (0.0, 0.0, -math.copysign(0.8, eh)), k, "heading"
        if abs(ey) > ALIGN_TOL:
            g = self._escalate("lateral", ey)
            rate = 4.6 if ey > 0 else 2.1  # mm per tick (measured, <= 5 ticks)
            k = int(np.clip(round(g * abs(ey) * 1000 / rate), 2, 10))
            return (0.0, math.copysign(0.15, ey), 0.0), k, "lateral"
        self._escalate("x", ex)
        # Forward is quantized: <= 10 ticks only twists the duck in place, 15 ticks is
        # one step (25-35 mm in the loop, plus a few degrees of yaw the heading pulse
        # then fixes). The x window is lopsided because the kick is (measured,
        # ts0.25 model_1000): a ball up to 30 mm CLOSER than nominal is always kicked,
        # farther than +15 mm it starts to whiff (+20: 7/8, +25: 5/8, +40: 0/8). A
        # step from just past the far edge lands inside, on the close side.
        if ex > STEP_MIN:
            return (0.2, 0.0, 0.0), 15, "step"
        if ex > ALIGN_X_WINDOW[1]:
            return None, 0, "backoff"   # too close to step safely, too far to kick: back off, walk in again
        if ex < ALIGN_X_WINDOW[0]:
            return (-0.2, 0.0, 0.0), 20, "back"     # ~7 mm
        return None


class Mission:
    def __init__(self, sim, kick_session, kick_heading, log, foot="right", precision_session=None, goal=None,
                 pose_src=None, ball_src=None):
        self.sim, self.kick_session = sim, kick_session
        self.pose_src = pose_src if pose_src is not None else GroundTruthProvider(sim)
        self.ball_src = ball_src if ball_src is not None else GroundTruthBall(sim)
        self.estimated = self.pose_src.name != "gt"
        self.pose_lost_t = None
        self.goal = goal              # soccer_referee.Goal, or None for a plain kick mission
        self.referee = None
        self.measured = False
        self.kh_goal = kick_heading                              # where the ball should go
        self.foot = foot
        self.kh = wrap_angle(kick_heading + KICK_AIM_OFFSET[foot])  # where the duck faces (aim-corrected)
        self.precision_session = precision_session
        self.precise_entries = 0
        self.on_ticks = 0
        self.walk_session = sim.policy.ort_session
        self.log = log
        self.aligner = Aligner(foot)
        self.state = "NAVIGATE"
        self.t_state = 0.0
        self.pulses = 0
        self.squares = 0
        self.pulse_queue = []
        self.backoffs = 0
        self.pulse_left = 0
        self.settle_left = 0
        self.pulse_twist = (0.0, 0.0, 0.0)
        self.result = {"approach_mode": "precision" if precision_session is not None else "pulse"}
        self.camera = hasattr(self.ball_src, "look")   # camera ball: scan, 5 Hz looks, re-observe
        self.nav = None
        if self.camera:
            sim.policy.head_offset[:] = 0.0
            self.ball_move_tol, self.ball_speed_tol = CAM_REBASE_TOL, CAM_BALL_SPEED_TOL
            self.look_attempt = 0
            self.ball_src.look(sim.t, force=True)
            b, _ = self.ball_src.get(sim.t)
            if b is None:
                self.ball0 = None
                self.state = "SCAN"
                log.event({"t": 0.0, "event": "SCAN_START"})
                return
        else:
            self.ball_move_tol, self.ball_speed_tol = 0.01, 0.02
            b, _ = self.ball_src.get(sim.t)
        self._plan(b, 0.0)

    def _plan(self, b, t):
        sim, log, goal, foot, kick_heading = self.sim, self.log, self.goal, self.foot, self.kh_goal
        self.ball0 = b.copy()
        stance = self.stance_point(b)
        k = np.array([math.cos(self.kh), math.sin(self.kh)])
        line_up, pre = stance - LINE_UP_M * k, stance - PRE_STANCE_M * k
        start = self.pose_src.get(sim.t)
        if start.pose is None:
            raise RuntimeError(f"mission started without a pose ({start.lost})")
        x, y, _ = start.pose
        route = [tuple(line_up), tuple(pre)]
        if self._seg_dist(np.array([x, y]), line_up, b) < DETOUR_CLEAR_M:
            side = np.array([-k[1], k[0]])
            side = side if (np.array([x, y]) - b) @ side > 0 else -side
            route.insert(0, tuple(b + 0.45 * side - 0.1 * k))
        self.nav = GotoSupervisor(route, getattr(sim, "arena", ARENA), GotoConfig(arrival_radius=0.06), log=log.event)
        self.state, self.t_state = "NAVIGATE", t
        log.event({"t": round(t, 3), "event": "MISSION_START", "ball": [round(v, 3) for v in b],
                   "kick_heading_deg": round(math.degrees(kick_heading), 1),
                   "aim_heading_deg": round(math.degrees(self.kh), 1),
                   "foot": foot, "aim_offset_deg": round(math.degrees(KICK_AIM_OFFSET[foot]), 1),
                   "stance_shift_mm": round(PRECISE_STANCE_SHIFT_M[foot] * 1000, 1),
                   "stance": [round(v, 3) for v in stance], "route": [[round(v, 3) for v in p] for p in route]})
        if goal is not None:
            log.event({"t": round(t, 3), "event": "GOAL_PLACED", "centre": [round(goal.cx, 3), round(goal.cy, 3)],
                       "into_deg": round(math.degrees(goal.yaw), 1), "opening_mm": round(goal.opening * 1000, 1),
                       "ball_to_goal_m": round(float(math.hypot(goal.cx - b[0], goal.cy - b[1])), 3)})

    def stance_point(self, ball_xy):
        """Duck position that puts the ball at the trained offset when facing the kick heading."""
        c, s = math.cos(self.kh), math.sin(self.kh)
        ox, oy = self.aligner.target
        return np.asarray(ball_xy) - np.array([c * ox - s * oy, s * ox + c * oy])

    @staticmethod
    def _seg_dist(a, b, p):
        ab = b - a
        t = np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
        return float(np.linalg.norm(a + t * ab - p))

    def _goto(self, state):
        if state == "DONE" and getattr(self, "camera", False):
            self.sim.policy.head_offset[:] = 0.0
        self.state, self.t_state = state, self.sim.t

    def _enter_fine(self, t):
        """Hand the last stretch to the precision policy if there is one (and entries left), else pulses."""
        if self.precision_session is not None and self.precise_entries < PRECISE_MAX_ENTRIES:
            self.settle_left = 20
            self._goto("PRECISE_PIVOT")
        else:
            self.log.event({"t": round(t, 3), "event": "ALIGN_START"})
            self.settle_left = ALIGN_SETTLE_TICKS
            self._goto("ALIGN")

    # --- camera channel (rung 6 Stage 3): scan for the ball, re-observe before committing ---
    def _scan(self, t, b):
        """No ball yet: turn in place, looking at 5 Hz, until the camera finds it (or give up)."""
        if b is not None:
            self.log.event({"t": round(t, 3), "event": "BALL_FOUND", "after_s": round(t - self.t_state, 2),
                            "meas": self.ball_src.last_meas and {k: self.ball_src.last_meas[k] for k in ("pixels", "u", "v")}})
            self.result["scan_s"] = round(t - self.t_state, 2)
            self._plan(b, t)
            return (0.0, 0.0, 0.0)
        if t - self.t_state > SCAN_MAX_S:
            self.result["aborted"] = "ball_not_found"
            self.log.event({"t": round(t, 3), "event": "BALL_NOT_FOUND", "phase": "SCAN"})
            self._goto("DONE")
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, SCAN_WZ)

    def _start_look(self, t):
        self.look_phase, self.look_t, self.look_hits = "down", t, []
        head = list(LOOK_HEAD)
        head[2] = RESCAN_HEAD_YAWS[self.look_attempt % len(RESCAN_HEAD_YAWS)]
        self.sim.policy.head_offset[:] = head
        self.log.event({"t": round(t, 3), "event": "LOOK_START", "attempt": self.look_attempt + 1, "head": head})
        self._goto("LOOK")

    def _look(self, t, pose):
        """Head down, settle, take LOOK_N looks, head back up, settle, then the fine approach. Ball not
        found = hold and re-scan (head yaw sweep); after RESCAN_MAX attempts, abort. Never kick blind."""
        sim = self.sim
        if self.look_phase == "down":
            if t - self.look_t >= LOOK_SETTLE_S:
                self.look_phase, self.look_t = "looking", t
            return (0.0, 0.0, 0.0)
        if self.look_phase == "looking":
            if t - self.look_t >= LOOK_SPACING_S * len(self.look_hits) - 1e-9 and len(self.look_hits) < LOOK_N:
                ok = self.ball_src.look(t, force=True)
                self.look_hits.append(self.ball_src.get(t)[0] if ok else None)
            if len(self.look_hits) < LOOK_N:
                return (0.0, 0.0, 0.0)
            hits = [h for h in self.look_hits if h is not None]
            self.result["looks"] = self.result.get("looks", 0) + LOOK_N
            self.result["looks_found"] = self.result.get("looks_found", 0) + len(hits)
            sim.policy.head_offset[:] = 0.0
            if not hits:
                self.look_attempt += 1
                self.result["rescans"] = self.result.get("rescans", 0) + 1
                self.log.event({"t": round(t, 3), "event": "LOOK_NOT_FOUND", "attempt": self.look_attempt})
                if self.look_attempt >= RESCAN_MAX:
                    self.result["aborted"] = "ball_not_found"
                    self.log.event({"t": round(t, 3), "event": "BALL_NOT_FOUND", "phase": "LOOK"})
                    self._goto("DONE")
                    return (0.0, 0.0, 0.0)
                self.look_phase, self.look_t = "up", t
                self.look_retry = True
                return (0.0, 0.0, 0.0)
            fresh = np.mean(hits, axis=0)
            self.ball_src.ball = fresh.copy()       # the fresh relative estimate, in the estimated frame
            moved = float(np.linalg.norm(fresh - self.ball0))
            self.log.event({"t": round(t, 3), "event": "LOOK_DONE", "found": len(hits), "rebase_mm": round(moved * 1000, 1)})
            self.result["look_rebase_mm"] = round(moved * 1000, 1)
            # relative-ball error (measurement only): camera ball vs truth, both in the duck's yaw frame
            true_b, _ = sim.ball()
            true_pose, _, _ = sim.pose()
            loc_c = self.aligner.errors(pose, fresh, self.kh)[3]
            loc_t = self.aligner.errors(true_pose, true_b, self.kh)[3]
            self.result["look_rel_err_mm"] = [round(float(v) * 1000, 1) for v in (loc_c - loc_t)]
            self.ball0 = fresh.copy()
            self.look_phase, self.look_t, self.look_retry = "up", t, False
            return (0.0, 0.0, 0.0)
        # "up": head back to neutral and settle before the step policy (trained head-neutral) engages
        if t - self.look_t >= LOOK_RESTORE_S:
            if self.look_retry:
                self._start_look(t)
            else:
                self._enter_fine(t)
        return (0.0, 0.0, 0.0)

    def _precise_fail(self, t, why):
        """Precision step gave up: VelStand back in, the pulse aligner (fallback) takes over."""
        self.sim.policy.ort_session = self.walk_session
        self.result["approach_mode"] = "precision->pulse fallback"
        self.result["precise_fail"] = why
        if why == "fall":
            self.result["falls"] = self.result.get("falls", 0) + 1
            self.down = True
        self.log.event({"t": round(t, 3), "event": "PRECISE_FALLBACK", "why": why})
        self.settle_left = ALIGN_SETTLE_TICKS
        self._goto("ALIGN")
        return (0.0, 0.0, 0.0)

    def precise_stance(self, ball_xy):
        """Stance target for the step policy: the trained stance, PRECISE_STANCE_SHIFT_M further along the kick line."""
        return self.stance_point(ball_xy) + PRECISE_STANCE_SHIFT_M[self.foot] * np.array([math.cos(self.kh), math.sin(self.kh)])

    def precise_offset(self, pose, ball_xy):
        """Option-A command: remaining offset to the stance pose in the trunk yaw frame, clipped to training range."""
        x, y, yaw = pose
        st = self.precise_stance(ball_xy)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = c * (st[0] - x) + s * (st[1] - y), -s * (st[0] - x) + c * (st[1] - y)
        r = math.hypot(dx, dy)
        if r > PRECISE_MAX_R:
            dx, dy = dx * PRECISE_MAX_R / r, dy * PRECISE_MAX_R / r
        return (dx, dy, float(np.clip(wrap_angle(self.kh - yaw), -PRECISE_MAX_YAW, PRECISE_MAX_YAW)))

    def _swap_kick(self, t, pose, b, tilt, ex, ey, eh, aligned):
        sim = self.sim
        foot, lfoot = sim.site_local(sim.kick_site), sim.site_local(sim.support_site)
        vb = sim.data.qvel[sim.va:sim.va + 3]
        self.result.update(foot_vs_home_mm=[round(v, 1) for v in (foot - sim.home_foot) * 1000],
                           lfoot_vs_home_mm=[round(v, 1) for v in (lfoot - sim.home_lfoot) * 1000],
                           trunk_speed_mm_s=round(float(np.linalg.norm(vb[:2])) * 1000, 1),
                           tilt_deg=round(tilt, 1))
        q = np.array([sim.data.qpos[j] for j in sim.policy.joint_qpos_indices]) - sim.policy.default_pose
        # the kick only ever started from HOME +-0.05 rad; precision run 1 arrived crouched and never kicked
        self.result["leg_dev_deg"] = round(math.degrees(float(np.abs(q[[0, 1, 2, 3, 4, 9, 10, 11, 12, 13]]).max())), 1)
        self.result.update(align_ex_mm=round(ex * 1000, 1), align_ey_mm=round(ey * 1000, 1),
                           align_eh_deg=round(math.degrees(eh), 2), pulses=self.pulses,
                           aligned=aligned, t_swap=round(t, 2))
        self.log.event({"t": round(t, 3), "event": "KICK_SWAP_IN", "aligned": aligned, "pulses": self.pulses,
                        "mode": self.result["approach_mode"],
                        "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)], "err_heading_deg": round(math.degrees(eh), 2)})
        if self.estimated:   # the gap at the moment it matters: where the duck thinks it is vs where it is
            true_pose, _, _ = sim.pose()
            exy, eyaw = pose_error(pose, true_pose)
            ball_true, _ = sim.ball()
            ex_t, ey_t, eh_t, _ = self.aligner.errors(true_pose, ball_true, self.kh,
                                                     sim.site_local(sim.kick_site), sim.home_foot)
            self.result.update(pose_err_mm=round(exy * 1000, 1), pose_err_yaw_deg=round(math.degrees(eyaw), 2),
                               true_align_ex_mm=round(ex_t * 1000, 1), true_align_ey_mm=round(ey_t * 1000, 1),
                               true_align_eh_deg=round(math.degrees(eh_t), 2))
        sim.policy.ort_session = self.kick_session
        if self.goal is not None:
            self.referee = GoalReferee(self.goal, GOAL_TIMEOUT_S)
        self.kick_dir = np.array([math.cos(pose[2]), math.sin(pose[2])])  # what the kick policy aims at
        self.kick_ball0 = sim.ball()[0].copy()   # scoring origin: ground truth
        self.peak = 0.0
        self._goto("KICK")
        return (0.0, 0.0, 0.0)

    def tick(self):
        """Returns the twist for this control tick (and swaps sessions as the state changes)."""
        sim, t = self.sim, self.sim.t
        est = self.pose_src.get(t)
        pose, tilt, z = est.pose, est.tilt_deg, est.trunk_z     # the supervisor's view
        b, bv = self.ball_src.get(t)
        bt, bvt = sim.ball()                                     # ground truth: scoring only
        ball_post = False
        if self.goal is not None:
            ball_post, duck_post = goal_contacts(sim)
            if duck_post and not self.result.get("duck_post_contact"):
                self.result["duck_post_contact"] = True
                self.log.event({"t": round(t, 3), "event": "DUCK_POST_CONTACT", "phase": self.state})
            if self.referee is not None and not self.referee.done:
                if self.referee.update(t, float(bt[0]), float(bt[1]), ball_post) is not None:
                    self.log.event({"t": round(t, 3), "event": "GOAL_DECIDED", **self.referee.result})
        if getattr(sim, "board_geoms", None) and not self.result.get("duck_board_contact") and duck_board_contact(sim):
            self.result["duck_board_contact"] = True
            self.log.event({"t": round(t, 3), "event": "DUCK_BOARD_CONTACT", "phase": self.state})
        if self.state in ("SCAN", "NAVIGATE", "LOOK", "ALIGN", "PRECISE_PIVOT", "PRECISE"):
            if pose is None:   # pose absent: hold still, never guess
                if self.pose_lost_t is None:
                    self.pose_lost_t = t
                    self.result["pose_lost"] = self.result.get("pose_lost", 0) + 1
                    self.log.event({"t": round(t, 3), "event": "POSE_LOST", "phase": self.state, "lost": est.lost,
                                    "age_s": round(est.age_s, 3)})
                if t - self.pose_lost_t > POSE_LOST_ABORT_S:
                    sim.policy.ort_session = self.walk_session
                    self.result["aborted"] = "pose_lost"
                    self.log.event({"t": round(t, 3), "event": "POSE_LOST_ABORT", "phase": self.state,
                                    "after_s": round(t - self.pose_lost_t, 2)})
                    self._goto("DONE")
                return (0.0, 0.0, 0.0)
            if self.pose_lost_t is not None:
                self.log.event({"t": round(t, 3), "event": "POSE_BACK", "phase": self.state,
                                "after_s": round(t - self.pose_lost_t, 2)})
                self.result["pose_lost_s"] = round(self.result.get("pose_lost_s", 0.0) + t - self.pose_lost_t, 2)
                self.pose_lost_t = None
        if self.state in ("KICK", "SETTLE"):   # scoring phases: truth (as before the seam)
            pose, tilt, z = sim.pose()
        if self.camera and self.state in ("SCAN", "NAVIGATE"):    # head neutral: 5 Hz looks
            if self.ball_src.look(t):
                b, bv = self.ball_src.get(t)
                if self.ball0 is not None and np.linalg.norm(b - self.ball0) < self.ball_move_tol:
                    self.ball0 = b.copy()   # a re-look re-bases the estimate (odometry drift), not a bump
        if self.state == "SCAN":
            return self._scan(t, b)
        if self.state == "LOOK":
            return self._look(t, pose)
        if self.state in ("NAVIGATE", "ALIGN", "PRECISE_PIVOT", "PRECISE"):
            # A bumped ball: stand still until it stops rolling, then re-plan from where
            # it rests (ALIGN re-navigates if the new stance is out of reach). Chasing a
            # rolling ball with pulses never converges.
            if np.linalg.norm(bv) > self.ball_speed_tol or np.linalg.norm(b - self.ball0) > self.ball_move_tol:
                if not getattr(self, "ball_rolling", False):
                    self.ball_rolling, self.t_bump = True, t
                    self.pulse_left, self.approaching = 0, False
                    self.result["ball_disturbed"] = self.result.get("ball_disturbed", 0) + 1
                    sim.policy.ort_session = self.walk_session   # (a no-op unless the step policy had it)
                    self.log.event({"t": round(t, 3), "event": "BALL_DISTURBED", "phase": self.state})
                if np.linalg.norm(bv) > self.ball_speed_tol:
                    if t - self.t_bump < BALL_WAIT_S:
                        return (0.0, 0.0, 0.0)
                    # Still creeping after the wait (the sim ball has ~no rolling
                    # resistance): stop for good rather than chase it. Bounded failure.
                    self.result["aborted"] = "ball_lost"
                    self.log.event({"t": round(t, 3), "event": "BALL_LOST", "ball_speed": round(float(np.linalg.norm(bv)), 3)})
                    self._goto("DONE")
                    return (0.0, 0.0, 0.0)
                self.ball_rolling = False
                self.log.event({"t": round(t, 3), "event": "BALL_SETTLED",
                                "moved_m": round(float(np.linalg.norm(b - self.ball0)), 3)})
                self.ball0 = b.copy()
                self._enter_fine(t)
                return (0.0, 0.0, 0.0)

        if self.state == "NAVIGATE":
            self.nav.update(est.t, pose, tilt, z)   # the estimate's own stamp feeds the watchdog
            twist = self.nav.command(t)
            if self.nav.state == "DONE":
                if self.camera:     # re-observe before the step policy commits
                    self._start_look(t)
                else:
                    self._enter_fine(t)
                return (0.0, 0.0, 0.0)
            return twist

        if self.state == "PRECISE_PIVOT":   # VelStand turns onto the kick line first
            if tilt > 35 or z < 0.06:
                return self._precise_fail(t, "fall")
            eh = wrap_angle(pose[2] - self.kh)
            if abs(eh) < PRECISE_PIVOT_TOL or t - self.t_state > PRECISE_PIVOT_MAX_S:
                if self.settle_left > 0:
                    self.settle_left -= 1
                    return (0.0, 0.0, 0.0)
                self.precise_entries += 1
                self.on_ticks = 0
                sim.policy.ort_session = self.precision_session
                self.log.event({"t": round(t, 3), "event": "PRECISE_START", "n": self.precise_entries,
                                "offset": [round(v, 4) for v in self.precise_offset(pose, b)]})
                self._goto("PRECISE")
                return self.precise_offset(pose, b)
            self.settle_left = 20
            return (0.0, 0.0, -math.copysign(0.8, eh))

        if self.state == "PRECISE":
            if tilt > 35 or z < 0.06:
                return self._precise_fail(t, "fall")
            off = self.precise_offset(pose, b)
            vb = est.v_xy   # odometry velocity (ground truth's qvel under GroundTruthProvider)
            on = math.hypot(off[0], off[1]) < ALIGN_TOL and abs(off[2]) < ALIGN_HEADING_TOL
            self.on_ticks = self.on_ticks + 1 if on and np.linalg.norm(vb) < PRECISE_STILL_V else 0
            if self.on_ticks >= PRECISE_HOLD_TICKS:
                ex, ey, eh, _ = self.aligner.errors(pose, b, self.kh, sim.site_local(sim.kick_site), sim.home_foot)
                self.log.event({"t": round(t, 3), "event": "PRECISE_ARRIVED", "after_s": round(t - self.t_state, 2)})
                return self._swap_kick(t, pose, b, tilt, ex, ey, eh, aligned=True)
            if t - self.t_state > PRECISE_TIMEOUT_S:
                return self._precise_fail(t, "timeout")
            return off

        if self.state == "ALIGN":
            if tilt > 35 or z < 0.06:  # zero twist while down; VelStand gets up on its own
                if not getattr(self, "down", False):
                    self.down = True
                    self.pulse_left = 0
                    self.result["falls"] = self.result.get("falls", 0) + 1
                    self.log.event({"t": round(t, 3), "event": "FALL", "phase": "ALIGN"})
                self.settle_left = 50
                return (0.0, 0.0, 0.0)
            if getattr(self, "down", False):
                self.down = False
                self.log.event({"t": round(t, 3), "event": "RECOVERED", "phase": "ALIGN"})
            if getattr(self, "backing", False):
                ex, ey, eh, _ = self.aligner.errors(pose, b, self.kh)
                if ex > BACKOFF_TO or t - self.t_back > 3.0:
                    self.backing = False
                    self.settle_left = ALIGN_SETTLE_TICKS
                    self.log.event({"t": round(t, 3), "event": "ALIGN_BACKOFF_STOP", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                    return (0.0, 0.0, 0.0)
                return clamp_align(-0.2, 0.0, 0.0)
            if getattr(self, "approaching", False):
                ex, ey, eh, _ = self.aligner.errors(pose, b, self.kh)  # trunk-relative while walking: the feet swing
                if ex < APPROACH_STOP or t - self.t_approach > 5.0:
                    self.approaching = False
                    self.settle_left = ALIGN_SETTLE_TICKS
                    self.log.event({"t": round(t, 3), "event": "ALIGN_APPROACH_STOP", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                    return (0.0, 0.0, 0.0)
                # walk at cruise, steering at the stance spot (yaw-frame bearing to it)
                st = self.stance_point(b) - np.asarray(pose[:2])
                bearing = wrap_angle(math.atan2(st[1], st[0]) - pose[2])
                return clamp_align(0.2, 0.0, 2.0 * bearing)
            if self.pulse_left > 0:
                self.pulse_left -= 1
                if self.pulse_left == 0:
                    self.settle_left = ALIGN_SETTLE_TICKS
                return clamp_align(*self.pulse_twist)
            if self.settle_left > 0:
                self.settle_left -= 1
                return (0.0, 0.0, 0.0)
            if self.pulse_queue:  # rest of a multi-pulse squaring move
                self.pulse_twist, self.pulse_left = self.pulse_queue.pop(0)
                self.pulse_left -= 1
                return clamp_align(*self.pulse_twist)
            ex, ey, eh, local = self.aligner.errors(pose, b, self.kh, sim.site_local(sim.kick_site), sim.home_foot)
            if ex > APPROACH_FROM and abs(eh) < math.radians(10) and abs(ey) < 0.03:
                self.approaching, self.t_approach = True, t
                self.log.event({"t": round(t, 3), "event": "ALIGN_APPROACH", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                return (0.0, 0.0, 0.0)
            stance = self.stance_point(b)
            if np.linalg.norm(stance - pose[:2]) > ALIGN_ZONE_M:   # drifted out: navigate back in
                self.log.event({"t": round(t, 3), "event": "ALIGN_LOST", "dist": round(float(np.linalg.norm(stance - pose[:2])), 3)})
                k = np.array([math.cos(self.kh), math.sin(self.kh)])
                self.nav = GotoSupervisor([tuple(stance - PRE_STANCE_M * k)], getattr(sim, "arena", ARENA), GotoConfig(arrival_radius=0.06),
                                          log=self.log.event)
                self._goto("NAVIGATE")
                return (0.0, 0.0, 0.0)
            p = self.aligner.pulse(ex, ey, eh, allow_backoff=self.backoffs < MAX_BACKOFFS)
            if p is not None and p[2] == "backoff" and self.pulses < ALIGN_MAX_PULSES:
                self.pulses += 1
                self.backoffs += 1
                self.backing, self.t_back = True, t
                self.log.event({"t": round(t, 3), "event": "ALIGN_BACKOFF", "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)]})
                return (0.0, 0.0, 0.0)
            if p is not None and p[2] == "backoff":
                p = None  # out of pulses: kick from here
            stagger = (sim.site_local(sim.kick_site) - sim.home_foot)[0]
            support = (sim.site_local(sim.support_site) - sim.home_lfoot)[0]
            if (p is None and (abs(stagger) > SQUARE_TOL or abs(support) > SUPPORT_TOL)
                    and self.squares < SQUARE_MAX and self.pulses < ALIGN_MAX_PULSES):
                # Ball placement is fine but a foot sits too far from its HOME spot: the
                # kick only ever started from a square stance. Measured over 133 mission
                # kicks: all 7 whiffs had the feet split front-to-back (kicking foot 22-33 mm
                # behind HOME, or the support foot 13-16 mm behind, where the policy doesn't
                # swing at all). Re-square, then re-check the fit.
                self.squares += 1
                self.pulses += 1
                (self.pulse_twist, self.pulse_left), *self.pulse_queue = SQUARE_MOVE
                self.log.event({"t": round(t, 3), "event": "ALIGN_SQUARE", "n": self.squares,
                                "kick_foot_vs_home_mm": round(stagger * 1000, 1),
                                "support_foot_vs_home_mm": round(support * 1000, 1)})
                self.pulse_left -= 1
                return clamp_align(*self.pulse_twist)
            if p is None or self.pulses >= ALIGN_MAX_PULSES:
                return self._swap_kick(t, pose, b, tilt, ex, ey, eh, aligned=p is None)
            self.pulse_twist, self.pulse_left, axis = p
            self.pulses += 1
            self.log.event({"t": round(t, 3), "event": "ALIGN_PULSE", "n": self.pulses, "axis": axis, "ticks": self.pulse_left,
                            "err_mm": [round(ex * 1000, 1), round(ey * 1000, 1)], "err_heading_deg": round(math.degrees(eh), 2)})
            self.pulse_left -= 1
            return clamp_align(*self.pulse_twist)

        bv = bvt   # KICK/SETTLE are scoring: ground truth
        if self.state == "KICK":
            fwd = float(bv @ np.array([math.cos(self.kh_goal), math.sin(self.kh_goal)]))
            if "kick_contact_t" not in self.result and np.linalg.norm(bv) > 0.05:
                self.result["kick_contact_t"] = round(t - self.t_state, 2)
                self.log.event({"t": round(t, 3), "event": "KICK_FIRE", "after_swap_s": self.result["kick_contact_t"]})
            self.peak = max(self.peak, fwd)
            if t - self.t_state >= KICK_HOLD_S:
                sim.policy.ort_session = self.walk_session
                self.log.event({"t": round(t, 3), "event": "KICK_SWAP_OUT", "peak_ball_fwd": round(self.peak, 3)})
                self.fell = False
                self._goto("SETTLE")
            return (0.0, 0.0, 0.0)

        if self.state == "SETTLE":
            self.peak = max(self.peak, float(bv @ np.array([math.cos(self.kh_goal), math.sin(self.kh_goal)])))
            if (tilt > 35 or z < 0.06) and not self.fell:
                self.fell = True
                self.result["falls"] = self.result.get("falls", 0) + 1
                self.log.event({"t": round(t, 3), "event": "FALL", "phase": "SETTLE"})
            up = tilt < 25 and z > 0.09
            if self.fell and up and not self.result.get("recovered"):
                self.result["recovered"] = True
                self.log.event({"t": round(t, 3), "event": "RECOVERED"})
            if t - self.t_state >= SETTLE_S and up and not self.measured:
                # measured at the same moment as every earlier kick mission (comparable numbers);
                # ball_travel is the distance at that moment, NOT a rest distance (frictionless ball)
                d = bt - self.kick_ball0
                self.measured = True
                self.result.update(peak_ball_fwd=round(self.peak, 3), ball_travel=round(float(np.linalg.norm(d)), 3),
                                   kick_dir_err_deg=round(math.degrees(wrap_angle(math.atan2(d[1], d[0]) - self.kh_goal)), 1)
                                   if np.linalg.norm(d) > 0.02 else None, t_done=round(t, 2))
            if self.measured and (self.referee is None or self.referee.done):
                if self.referee is not None:   # GoalKick: wait for the ball to reach the goal line
                    self.result.update(self.referee.result, t_done=round(t, 2))
                self.log.event({"t": round(t, 3), "event": "MISSION_DONE", **self.result})
                self._goto("DONE")
        return (0.0, 0.0, 0.0)


def draw_mission(scn, m, sim):
    """Stance target (disk = +-15 mm), kick-direction arrow from the ball, pending route."""
    scn.ngeom = 0

    def add(gtype, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def line(a, b, rgba, width=0.004):
        g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, [1, 1, 1], [0, 0, 0], rgba)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))

    if m.goal is not None:   # the goal line between the posts (the posts themselves are real geoms)
        g = m.goal
        c, s = math.cos(g.yaw), math.sin(g.yaw)
        h = g.half_opening + g.post_radius
        line((g.cx - s * h, g.cy + c * h, 0.002), (g.cx + s * h, g.cy - c * h, 0.002), [1, 1, 1, 0.9], 0.003)
    if m.estimated:   # where the supervisor believes the duck is: blue disk + heading tick
        e = m.pose_src.get(sim.t)
        if e.pose is not None:
            ex, ey, eyaw = e.pose
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.025, 0.003, 0], [ex, ey, 0.004], [0.2, 0.4, 1.0, 0.8])
            line((ex, ey, 0.007), (ex + 0.08 * math.cos(eyaw), ey + 0.08 * math.sin(eyaw), 0.007), [0.2, 0.4, 1.0, 0.9], 0.003)
    if m.state == "DONE":
        return
    b, _ = sim.ball()
    k = np.array([math.cos(m.kh_goal), math.sin(m.kh_goal)])   # the intended kick line (the duck aims off it)
    line((*b, 0.004), (*(b + 0.35 * k), 0.004), [1.0, 0.85, 0.1, 0.9], 0.005)
    tip = b + 0.35 * k
    side = np.array([-k[1], k[0]])
    for s in (1, -1):
        line((*tip, 0.004), (*(tip - 0.06 * k + s * 0.035 * side), 0.004), [1.0, 0.85, 0.1, 0.9], 0.005)
    st = m.precise_stance(b) if m.precision_session is not None else m.stance_point(b)
    add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0], [*st, 0.002], [0.1, 0.9, 0.2, 0.6])
    if m.state == "NAVIGATE":
        (px, py, _), _, _ = sim.pose()
        prev = (px, py)
        for _, (x, y) in m.nav.queue:
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0], [x, y, 0.002], [1.0, 0.6, 0.1, 0.5])
            line((*prev, 0.006), (x, y, 0.006), [1.0, 0.6, 0.1, 0.5], 0.002)
            prev = (x, y)


def new_mission(sim, kick, rng, log, ball=None, kick_heading=None, precision=None, foot="right", goal_dist=None,
                pose_src=None, ball_src=None):
    """Put the ball 1.0-1.8 m from the duck (or at `ball`) and start a mission from where the duck stands.
    With goal_dist, the kick heading is the duck's walk-in bearing turned by an approach angle drawn
    from GOAL_APPROACH_DEG, and the goal stands goal_dist m past the ball on that line."""
    (x, y, _), _, _ = sim.pose()
    if ball is None:
        for _ in range(100):
            d, a = rng.uniform(1.0, 1.8), rng.uniform(-math.pi, math.pi)
            ball = (x + d * math.cos(a), y + d * math.sin(a))
            if ARENA[0] + 0.6 < ball[0] < ARENA[1] - 0.6 and ARENA[2] + 0.6 < ball[1] < ARENA[3] - 0.6:
                break
    kh = rng.uniform(-math.pi, math.pi) if kick_heading is None else kick_heading
    goal = None
    if goal_dist is not None:
        if kick_heading is None:
            kh = wrap_angle(math.atan2(ball[1] - y, ball[0] - x) + math.radians(float(rng.choice(GOAL_APPROACH_DEG))))
        goal = place_goal(sim, (ball[0] + goal_dist * math.cos(kh), ball[1] + goal_dist * math.sin(kh)), kh)
    sim.data.qpos[sim.bqa:sim.bqa + 7] = [ball[0], ball[1], 0.035, 1, 0, 0, 0]
    sim.data.qvel[sim.bva:sim.bva + 6] = 0
    mujoco.mj_forward(sim.model, sim.data)
    if pose_src is not None:
        pose_src.reset(sim, seed=int(rng.integers(1 << 30)) if pose_src.name != "gt" else 0)   # a mission starts from a known pose
    if ball_src is not None:   # camera ball: re-anchored on the mission's pose source, starts unseen
        ball_src.pose_src = pose_src or GroundTruthProvider(sim)
        ball_src.reset()
    return Mission(sim, kick, kh, log, foot=foot, precision_session=precision, goal=goal, pose_src=pose_src,
                   ball_src=ball_src)


def setup_goal(sim, name="goal_a"):
    """Cache the goal's mocap slot and the geom sets goal_contacts() needs (goal scene only)."""
    m = sim.model
    body = m.body(name).id
    sim.goal_name, sim.goal_mocap = name, int(m.body_mocapid[body])
    sim.post_geoms = {m.geom(f"{name}_post_left").id, m.geom(f"{name}_post_right").id}
    sim.ball_geom = m.geom("ball_geom").id
    ball_body = m.geom_bodyid[sim.ball_geom]
    sim.robot_geoms = {g for g in range(m.ngeom)
                       if m.body_rootid[m.geom_bodyid[g]] not in (0, ball_body) and m.body_mocapid[m.geom_bodyid[g]] < 0}
    # Render statistics: BAM's MujocoController calls mj_setConst, which recomputes extent/center
    # from the parked goals (+-50 m -> extent 100 m: blocky shadows, black horizon lines). Restore
    # the scene file's pinned <statistic>. extent/center are render-only (the solver uses meaninertia).
    spec = mujoco.MjSpec.from_file(getattr(sim, "scene_path", SCENES["allcollisions_ball_goal"]))
    m.stat.extent, m.stat.center[:] = spec.stat.extent, spec.stat.center


FIELD_INSET_M = 0.5   # the duck's geofence sits this far inside the boards


def field_from_model(m):
    """The field's inner faces (x_min, x_max, y_min, y_max), read back from field.xml's board geoms."""
    def face(name, axis, sign):
        g = m.geom(name)
        return float(g.pos[axis] - sign * g.size[axis])
    return (face("field_board_end_neg_left", 0, -1), face("field_board_end_pos_left", 0, 1),
            face("field_board_side_right", 1, -1), face("field_board_side_left", 1, 1))


def setup_field(sim):
    """Field scene: board geoms for contact checks, both goals parked in their mouths (goal_a at
    +x, goal_b at -x, each facing out of the field). Call after setup_goal (which restores the
    render statistics). Returns the geofence = the field inset by FIELD_INSET_M."""
    m, d = sim.model, sim.data
    sim.board_geoms = {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("field_")}
    fx0, fx1, fy0, fy1 = field_from_model(m)
    for name, x, yaw in (("goal_a", fx1, 0.0), ("goal_b", fx0, math.pi)):
        mid = int(m.body_mocapid[m.body(name).id])
        d.mocap_pos[mid] = [x, 0.0, 0.0]
        d.mocap_quat[mid] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(m, d)
    sim.field = (fx0, fx1, fy0, fy1)
    return (fx0 + FIELD_INSET_M, fx1 - FIELD_INSET_M, fy0 + FIELD_INSET_M, fy1 - FIELD_INSET_M)


def duck_board_contact(sim):
    d = sim.data
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if (g1 in sim.board_geoms and g2 in sim.robot_geoms) or (g2 in sim.board_geoms and g1 in sim.robot_geoms):
            return True
    return False


def place_goal(sim, centre, into_yaw):
    """Move the goal so its goal-line centre is at `centre`, facing `into_yaw`; returns the Goal
    as the compiled model reports it (opening/post radius are read back, never hand-copied)."""
    m, d = sim.model, sim.data
    d.mocap_pos[sim.goal_mocap] = [centre[0], centre[1], 0.0]
    d.mocap_quat[sim.goal_mocap] = [math.cos(into_yaw / 2), 0.0, 0.0, math.sin(into_yaw / 2)]
    mujoco.mj_forward(m, d)
    lp = d.geom_xpos[m.geom(f"{sim.goal_name}_post_left").id]
    rp = d.geom_xpos[m.geom(f"{sim.goal_name}_post_right").id]
    r = float(m.geom_size[m.geom(f"{sim.goal_name}_post_left").id][0])
    return Goal.from_posts(lp[:2], rp[:2], r, into_yaw)


def goal_contacts(sim):
    """(ball touching a post, duck touching a post) this step."""
    d, ball_post, duck_post = sim.data, False, False
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if g1 in sim.post_geoms or g2 in sim.post_geoms:
            other = g2 if g1 in sim.post_geoms else g1
            ball_post |= other == sim.ball_geom
            duck_post |= other in sim.robot_geoms
    return ball_post, duck_post


def set_feet(sim, foot):
    """Kicking/support foot sites and their HOME positions (trunk yaw frame), read at the reset stand."""
    sim.kick_site, sim.support_site = f"{foot}_foot", ("left_foot" if foot == "right" else "right_foot")
    sim.home_foot, sim.home_lfoot = sim.site_local(sim.kick_site), sim.site_local(sim.support_site)


def run_viewer(args):
    import mujoco.viewer
    sim = KickSim(args.velstand, SCENES[args.scene])
    kick = ort.InferenceSession(args.kick)
    precision = ort.InferenceSession(args.precision) if args.precision else None
    sim.reset(0, 0, 0)
    set_feet(sim, args.foot)
    if args.goal_dist is not None or args.scene == "field":
        setup_goal(sim)
    if args.scene == "field":
        sim.arena = setup_field(sim)
    pose_src = None if args.pose == "gt" else make_provider(args.pose, sim, imu_error=not args.no_imu_error, calib=args.odom_calib)
    rng = np.random.default_rng(args.seed)
    out = f"logs/kick_mission/{time.strftime('%Y-%m-%d_%H-%M-%S')}_viewer"
    os.makedirs(out, exist_ok=True)
    ball = tuple(float(v) for v in args.ball.split(",")) if args.ball else None
    kh = math.radians(args.kick_heading) if args.kick_heading is not None else None
    log = RunLog(echo=not args.auto)
    # Duck-cam: the head camera mounted exactly as the perception stack has it (apply_profile: the
    # board's rotation and fovy off the tof site, which also undoes the MJCF camera facing backwards).
    # With --ball-src camera, CameraBall applies the same profile to the same camera; either way the
    # viewer's fixed camera IS the detector's camera.
    from mjlab_microduck.sim import board_profiles
    from mjlab_microduck.sim.camera import apply_profile
    head_cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    ball_cam = None
    if args.ball_src == "camera":
        from camera_ball import CameraBall
        ball_cam = CameraBall(sim, None, profile=args.profile)
    else:
        apply_profile(sim.model, head_cam, board_profiles.get(args.profile))
    m = new_mission(sim, kick, rng, log, ball, kh, precision, args.foot, args.goal_dist, pose_src=pose_src,
                    ball_src=ball_cam)
    n, t_mission, t_done, tally = 0, sim.t, None, {"kicked": 0, "missed": 0, "aborted": 0, "falls": 0}
    goals = {"goal": 0, "post": 0, "miss": 0, "no_cross": 0}

    def report(m):
        r = m.result
        if r.get("aborted") or m.state != "DONE":
            tally["aborted"] += 1
            outcome = f"ABORTED ({r.get('aborted', 'timeout in ' + m.state)})"
        elif (r.get("peak_ball_fwd") or 0) > 0.05:
            tally["kicked"] += 1
            de = r.get("kick_dir_err_deg")
            if m.goal is not None:
                outcome = (f"{r['goal_outcome'].upper():8s} ball {r['peak_ball_fwd']:.2f} m/s"
                           + (f", {r['goal_lateral_mm']:+4.0f} mm at the line in {r['goal_t_line_s']:.1f} s"
                              if r.get("goal_lateral_mm") is not None else "")
                           + (" (post, in)" if r["goal_outcome"] == "post" and r.get("goal_went_in") else ""))
            else:
                outcome = (f"KICKED  ball {r['peak_ball_fwd']:.2f} m/s, rolled {r.get('ball_travel', 0):.2f} m"
                           + (f", {de:+.0f} deg off the arrow" if de is not None else ""))
        else:
            tally["missed"] += 1
            outcome = "MISSED (no kick)"
        if m.goal is not None and r.get("goal_outcome") and not r.get("aborted"):
            goals[r["goal_outcome"]] += 1
        tally["falls"] += r.get("falls", 0)
        mode = {"precision": "step", "precision->pulse fallback": "step->pulse", "pulse": "pulse"}[r["approach_mode"]]
        tail = (f"[goals {goals['goal']}, posts {goals['post']}, misses {goals['miss']}, short {goals['no_cross']}, "
                f"falls {tally['falls']}]" if m.goal is not None else
                f"[tally: {tally['kicked']} kicked, {tally['missed']} missed, {tally['aborted']} aborted, "
                f"{tally['falls']} falls]")
        print(f"  mission {n + 1:2d}: {outcome:58s} [{mode}] aligned to ({r.get('align_ex_mm', float('nan')):+.0f}, "
              f"{r.get('align_ey_mm', float('nan')):+.0f}) mm in {r.get('pulses', '-')} pulses   {tail}", flush=True)

    print(f"\nKeys (type in THIS terminal): N = new mission (ball re-placed), P = push, C = duck-cam ({args.profile}) "
          f"/ V = free view, Q = quit\nball source: {args.ball_src}"
          + ("  (head-down re-observe is camera-mode only)" if args.ball_src != "camera" else "")
          + (f"\nAuto mode: {args.auto} missions back to back\n" if args.auto else "\n"))
    with ip.TerminalInput() as term, \
            mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.trunk_body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.2, -30, 135
        duck_cam = False

        def set_view(on):
            with viewer.lock():
                if on:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = head_cam
                else:   # back to the free (tracking) 3rd-person view, where it was
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = sim.trunk_body
            print(f"  view: {'DUCK CAM (' + args.profile + ', vfov ' + format(sim.model.cam_fovy[head_cam], '.1f') + ' deg)' if on else 'FREE'}",
                  flush=True)
        quit_req = False
        while viewer.is_running() and not quit_req:
            t0 = time.time()
            for key in term.get_keys():
                if key == "q":
                    quit_req = True
                elif key == "c":
                    duck_cam = not duck_cam
                    set_view(duck_cam)
                elif key == "v" and duck_cam:
                    duck_cam = False
                    set_view(False)
                elif key == "p":
                    a = rng.uniform(0, 2 * math.pi)
                    sim.push(1.0 * math.cos(a), 1.0 * math.sin(a))
                    log.event({"t": round(sim.t, 3), "event": "PUSH"})
                elif key == "n":
                    sim.policy.ort_session = m.walk_session
                    log.save(f"{out}/mission{n}")
                    n += 1
                    log = RunLog()
                    m = new_mission(sim, kick, rng, log, precision=precision, foot=args.foot, goal_dist=args.goal_dist,
                                    pose_src=pose_src, ball_src=ball_cam)
            if args.auto:
                over = m.state == "DONE" or sim.t - t_mission > 60.0
                if over and t_done is None:
                    t_done = sim.t
                    sim.policy.ort_session = m.walk_session
                    report(m)
                if t_done is not None and sim.t - t_done > 1.5:  # a beat of standing between missions
                    log.save(f"{out}/mission{n}")
                    n += 1
                    if n >= args.auto:
                        print(f"\n  {args.auto} missions: {tally['kicked']} kicked, {tally['missed']} missed, "
                              f"{tally['aborted']} aborted, {tally['falls']} falls")
                        if args.goal_dist is not None:
                            print(f"  goal at {args.goal_dist:.2f} m: {goals['goal']} goals, {goals['post']} post hits, "
                                  f"{goals['miss']} misses, {goals['no_cross']} short")
                        break
                    log = RunLog(echo=False)
                    m = new_mission(sim, kick, rng, log, precision=precision, foot=args.foot, goal_dist=args.goal_dist,
                                    pose_src=pose_src, ball_src=ball_cam)
                    t_mission, t_done = sim.t, None
            twist = m.tick() if m.state not in ("DONE",) and t_done is None else (0.0, 0.0, 0.0)
            sim.set_twist(twist)
            sim.step()
            if pose_src is not None:
                pose_src.update(sim)
            if duck_cam:
                viewer.user_scn.ngeom = 0     # the detector's frame has no overlays; neither does the duck-cam
            else:
                draw_mission(viewer.user_scn, m, sim)
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
    log.save(f"{out}/mission{n}")
    print(f"logs: {out}/")


def run_headless(args):
    sim = KickSim(args.velstand, SCENES[args.scene])
    kick = ort.InferenceSession(args.kick)
    precision = ort.InferenceSession(args.precision) if args.precision else None
    sim.reset(0, 0, 0)
    set_feet(sim, args.foot)
    if args.goal_dist is not None or args.scene == "field":
        setup_goal(sim)
    if args.scene == "field":
        sim.arena = setup_field(sim)
        print(f"field: boards {sim.field}, geofence {sim.arena}")
    pose_src = None if (args.pose == "gt" and not args.dropout) else make_provider(args.pose, sim, imu_error=not args.no_imu_error, calib=args.odom_calib)
    print(f"pose source: {args.pose}" + (f" (IMU yaw error {'off' if args.no_imu_error else 'on'}, calib {args.odom_calib})" if args.pose != "gt" else "")
          + (f", dropout {args.dropout}" if args.dropout else ""))
    print(f"{args.foot} (kicking) foot at HOME, trunk yaw frame: ({sim.home_foot[0]*1000:.1f}, {sim.home_foot[1]*1000:.1f}) mm")
    rng = np.random.default_rng(args.seed)
    out = unique_dir(f"logs/kick_mission/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.foot}"
                     + (f"_goal{args.goal_dist}" if args.goal_dist is not None else "") + f"_{args.scene}_{args.pose}"
                     + ("_cam" if args.ball_src == "camera" else "") + ("_dropout" if args.dropout else ""))
    print(f"velstand={args.velstand}\nkick={args.kick}\nprecision={args.precision} (pulse aligner = fallback)\nscene={args.scene}  logs -> {out}/")
    print(f"kicking foot {args.foot}: precision stance shift {PRECISE_STANCE_SHIFT_M[args.foot]*1000:+.1f} mm along the kick line; "
          f"aim offset {math.degrees(KICK_AIM_OFFSET[args.foot]):+.1f} deg (+ = left)")
    ball_cam = None
    if args.ball_src == "camera":
        from camera_ball import CameraBall
        ball_cam = CameraBall(sim, None, profile=args.profile)
        print(f"ball source: camera ({args.profile}, ray->plane, K fx={ball_cam.cam.K['fx']:.1f} at "
              f"{ball_cam.cam.w}x{ball_cam.cam.h})" + (f", ball hidden {args.hide_ball} s from the first re-observe" if args.hide_ball else ""))
    rows = []
    for k in range(args.runs):
        sim.reset(0.0, 0.0, rng.uniform(-math.pi, math.pi))
        bd, bb = rng.uniform(1.0, 1.8), rng.uniform(-math.pi, math.pi)
        kh = rng.uniform(-math.pi, math.pi)   # drawn in every mode, so the default rng stream is unchanged
        ball = (bd * math.cos(bb), bd * math.sin(bb))
        goal, approach = None, None
        if args.goal_dist is not None:
            # stratified approach angle (duck walk-in bearing vs ball->goal line), cycled per run
            approach = GOAL_APPROACH_DEG[k % len(GOAL_APPROACH_DEG)]
            kh = wrap_angle(bb + math.radians(approach))
            goal = place_goal(sim, (ball[0] + args.goal_dist * math.cos(kh), ball[1] + args.goal_dist * math.sin(kh)), kh)
        sim.data.qpos[sim.bqa:sim.bqa + 7] = [*ball, 0.035, 1, 0, 0, 0]
        sim.data.qvel[sim.bva:sim.bva + 6] = 0
        mujoco.mj_forward(sim.model, sim.data)
        log = RunLog(echo=args.verbose)
        if pose_src is not None:
            pose_src.reset(sim, seed=k)
            pose_src.dropouts = dropouts_for(args)
        ball_src = None
        if args.ball_src == "camera":
            ball_cam.pose_src = pose_src or GroundTruthProvider(sim)
            ball_cam.reset(hide_s=args.hide_ball)
            ball_src = ball_cam
        m = Mission(sim, kick, kh, log, foot=args.foot, precision_session=precision, goal=goal, pose_src=pose_src,
                    ball_src=ball_src)
        while m.state != "DONE" and sim.t < args.timeout:
            twist = m.tick()
            sim.set_twist(twist)
            (x, y, yaw), tilt, z = sim.pose()
            bxy, _ = sim.ball()
            rf, lf = sim.site_local("right_foot"), sim.site_local("left_foot")
            log.trace.append({"t": round(sim.t, 3), "state": m.state, "x": round(x, 4), "y": round(y, 4),
                              "yaw": round(yaw, 4), "tilt_deg": round(tilt, 1), "trunk_z": round(z, 4),
                              "cmd_vx": round(twist[0], 3), "cmd_vy": round(twist[1], 3), "cmd_wz": round(twist[2], 3),
                              "ball_x": round(float(bxy[0]), 4), "ball_y": round(float(bxy[1]), 4),
                              "rfoot_x": round(float(rf[0]), 4), "rfoot_y": round(float(rf[1]), 4),
                              "lfoot_x": round(float(lf[0]), 4), "lfoot_y": round(float(lf[1]), 4)})
            if pose_src is not None:
                e = pose_src.get(sim.t)
                log.trace[-1].update({"est_x": None if e.pose is None else round(e.pose[0], 4),
                                      "est_y": None if e.pose is None else round(e.pose[1], 4),
                                      "est_yaw": None if e.pose is None else round(e.pose[2], 4),
                                      "est_lost": e.lost, "est_age_s": round(min(e.age_s, 99.0), 3)})
            sim.step()
            if pose_src is not None:
                pose_src.update(sim)
        if m.state != "DONE":
            sim.policy.ort_session = m.walk_session
            m.result["timeout_in"] = m.state
        r = {"run": k, "ball_dist_m": round(bd, 2), **m.result}
        if ball_cam is not None:
            r["cam_looks"] = dict(ball_cam.stats)
            ball_cam.unhide()
        if approach is not None:
            r["approach_deg"] = approach
        rows.append(r)
        log.save(f"{out}/run{k}")
        print(f"  run {k}: ball {bd:.2f} m away, kick heading {math.degrees(kh):+4.0f}deg -> " +
              ", ".join(f"{a}={b}" for a, b in r.items() if a not in ("run", "ball_dist_m")), flush=True)
    al = [r for r in rows if r.get("aligned")]
    print(f"\n  aligned within +-{ALIGN_TOL*1000:.0f} mm / {math.degrees(ALIGN_HEADING_TOL):.0f} deg: {len(al)}/{len(rows)}; "
          f"kicked (peak > 0.05 m/s): {sum((r.get('peak_ball_fwd') or 0) > 0.05 for r in rows)}/{len(rows)}; "
          f"falls: {sum(r.get('falls', 0) for r in rows)}; ball disturbed before kick: "
          f"{sum(r.get('ball_disturbed', 0) > 0 for r in rows)} runs; median mission time "
          f"{np.median([r['t_done'] for r in rows if 't_done' in r]) if any('t_done' in r for r in rows) else float('nan'):.1f} s")
    modes = {}
    for r in rows:
        modes[r["approach_mode"]] = modes.get(r["approach_mode"], 0) + 1
    print(f"  approach mode: {modes}")
    if args.goal_dist is not None:
        print_goal_summary(rows, args.goal_dist, args.foot)
    if pose_src is not None and pose_src.name != "gt":
        print_pose_gap(rows)


def unique_dir(base):
    """A fresh log directory: batteries launched in the same second must not share one."""
    out, n = base, 1
    while True:
        try:
            os.makedirs(out)
            return out
        except FileExistsError:
            n += 1
            out = f"{base}_{n}"


def dropouts_for(args):
    """--dropout T0,DUR: the pose link is down from T0 s after each mission start for DUR s."""
    if not args.dropout:
        return []
    t0, dur = (float(v) for v in args.dropout.split(","))
    return [(t0, t0 + dur)]


def print_pose_gap(rows):
    """Estimated-vs-truth at the kick swap-in, and how the pose behaved over the battery."""
    sw = [r for r in rows if r.get("pose_err_mm") is not None]
    def q(vals, f=lambda v: v):
        vals = [f(v) for v in vals]
        return (f"median {np.median(vals):.1f}, p90 {np.percentile(vals, 90):.1f}, max {max(vals):.1f}"
                if vals else "n/a")
    print(f"\n  POSE GAP at kick swap-in ({len(sw)}/{len(rows)} missions reached it), estimate vs truth:")
    print(f"    position error mm: {q([r['pose_err_mm'] for r in sw])}")
    print(f"    |yaw error| deg:   {q([r['pose_err_yaw_deg'] for r in sw], abs)}")
    print(f"    TRUE ball placement at swap (supervisor believed |ex|<=12/20, |ey|<=12 mm): "
          f"|ex| {q([r['true_align_ex_mm'] for r in sw], abs)}; |ey| {q([r['true_align_ey_mm'] for r in sw], abs)}")
    lost = [r for r in rows if r.get("pose_lost")]
    print(f"    pose lost in {len(lost)} missions; aborted pose_lost {sum(r.get('aborted') == 'pose_lost' for r in rows)}; "
          f"resumed {sum(1 for r in lost if r.get('aborted') != 'pose_lost')}")


def print_goal_summary(rows, goal_dist, foot):
    """The GoalKick brief's per-foot lines: goals, post hits, miss distance, kicks not taken, bumps, falls."""
    def line(rs, label):
        kicked = [r for r in rs if (r.get("peak_ball_fwd") or 0) > 0.05 and not r.get("aborted")]
        oc = {o: sum(r.get("goal_outcome") == o for r in kicked) for o in ("goal", "post", "miss", "no_cross")}
        post_in = sum(r.get("goal_outcome") == "post" and r.get("goal_went_in") for r in kicked)
        lat = [r["goal_lateral_mm"] for r in kicked if r.get("goal_lateral_mm") is not None]
        mb = [r["goal_miss_by_mm"] for r in kicked if r.get("goal_outcome") == "miss"]
        tl = [r["goal_t_line_s"] for r in kicked if r.get("goal_t_line_s") is not None]
        worst = max(lat, key=abs) if lat else float("nan")
        print(f"  {label:14s} n={len(rs):3d}  GOALS {oc['goal']:3d}  posts {oc['post']:2d} ({post_in} in)  "
              f"misses {oc['miss']:2d}  short {oc['no_cross']:2d} | lateral at line median "
              f"{np.median(lat) if lat else float('nan'):+5.0f} mm, worst {worst:+5.0f}"
              + (f", misses outside the post by median {np.median(mb):.0f} mm (max {max(mb):.0f})" if mb else "")
              + f" | time to line {np.median(tl) if tl else float('nan'):.1f} s"
              f" | no kick {sum(1 for r in rs if (r.get('peak_ball_fwd') or 0) <= 0.05 and not r.get('aborted'))}"
              f", aborted {sum(1 for r in rs if r.get('aborted') or r.get('timeout_in'))}"
              f", bumped {sum(r.get('ball_disturbed', 0) > 0 for r in rs)}, falls {sum(r.get('falls', 0) for r in rs)}"
              f", duck hit post {sum(bool(r.get('duck_post_contact')) for r in rs)}")
    print(f"\n  GOAL SUMMARY ({foot} foot, goal {goal_dist:.2f} m past the ball, 210 mm opening):")
    line(rows, "all")
    for a in GOAL_APPROACH_DEG:
        rs = [r for r in rows if r.get("approach_deg") == a]
        if rs:
            line(rs, f"approach {a:+d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("velstand")
    ap.add_argument("kick")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", choices=list(SCENES), default="allcollisions_ball")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--ball", default=None, help='viewer: "x,y" ball position (default: random 1.0-1.8 m away)')
    ap.add_argument("--kick-heading", type=float, default=None, help="viewer: kick direction, deg (default: random)")
    ap.add_argument("--auto", type=int, default=0, help="viewer: run this many missions back to back, with a tally")
    ap.add_argument("--foot", choices=("right", "left"), default="right",
                    help="kicking foot of the kick ONNX (sets the stance side and that foot's calibrations)")
    ap.add_argument("--precision", default=None,
                    help="PrecisionStep ONNX for the final approach (pulse aligner stays as the fallback)")
    ap.add_argument("--calib", choices=tuple(CALIBRATIONS), default="v1",
                    help="which kick generation's stance/aim calibrations to use (v1 = BallKick v1 winners)")
    ap.add_argument("--goal-dist", type=float, default=None,
                    help="GoalKick: a 210 mm goal this many m past the ball on the kick line (switches to the goal "
                         "scene); headless cycles approach angles over GOAL_APPROACH_DEG")
    ap.add_argument("--pose", choices=("gt", "odom"), default="gt",
                    help="supervisor pose source (rung 6): gt = simulator truth (published baselines), odom = contact "
                         "odometry + IMU from a known start (pose_provider.py)")
    ap.add_argument("--no-imu-error", action="store_true", help="odom: perfect IMU heading (the measured yaw error off)")
    ap.add_argument("--ball-src", choices=("gt", "camera"), default="gt",
                    help="ball source: ground truth, or the head camera (relative line-up + re-observe)")
    ap.add_argument("--profile", default="face-board-v1", help="camera board profile (sim/board_profiles.py)")
    ap.add_argument("--hide-ball", type=float, default=0.0,
                    help="camera: grey the ball out for this many seconds from the first re-observe (inf = for good)")
    ap.add_argument("--odom-calib", choices=tuple(ODOM_CALIBS), default="none",
                    help="odom: versioned odometry calibration (stride_v1 = +1.9 %% stride scale)")
    ap.add_argument("--dropout", default=None, help='"T0,DUR": pose link down from T0 s after each mission start, DUR s')
    args = ap.parse_args()
    if args.goal_dist is not None and args.scene not in ("allcollisions_ball_goal", "field"):
        args.scene = "allcollisions_ball_goal"
    apply_calibration(args.calib)
    run_viewer(args) if args.viewer else run_headless(args)


if __name__ == "__main__":
    main()
