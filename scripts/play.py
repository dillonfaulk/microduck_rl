"""Continuous play on the field (brief microduck-continuous-play-brief.md, Workstreams A-C). Frozen policies.

  uv run python scripts/play.py <velstand.onnx> <kick.onnx> --foot right --calib v2 --precision <step.onnx> \
      --pose odom --odom-calib stride_v1 --estimated-only [--runs 30 --time-cap 240] [--viewer]

A. FIXED GOALS. scene_field.xml: boards at +-3 m, goal_a locked in the +x mouth, goal_b in the -x mouth,
   both centred on the centre line, for the whole run and across runs. Only the ball spawns randomly. Each
   run attacks one end (`--attack`, default alternating +x / -x by run). The goals' spots are constants of
   the FIELD MAP: the target assignment (Stage 4, 0.25 m grid, 0.6 m gate) is that spot, the fall protocol
   (C) uses both as landmarks, and an "advance kick" (below) aims at it. Their pose is never read from the
   simulator at runtime; the duck perceives them through the camera.

B. CONTINUOUS PLAY. A loop of Stage 4 missions (camera ball, camera goal, estimated pose): perceive ball ->
   go -> line up -> kick toward the assigned goal -> watch the ball come to rest -> again. The REFEREE
   (ground truth, environment side, like all scoring) watches every tick:
     goal    the ball's centre crosses either goal line between the posts (attacking goal or the other)
     out     the ball touches a board (the boards stand on the boundary lines), or its centre is past a line
             outside the goal mouth (a ball the duck pushed through a board's soft contact)
   and after either it respawns the ball at a fresh random spot. Two more restarts, both counted:
     kick cap   BALL_KICK_CAP kicks at one ball without a decision -> dead ball, respawn
     unplayable the duck reports the ball unplayable (its stance spot is off the field, e.g. pinned in a
                corner) -> referee's drop ball, respawn
   The run ends at --time-cap (sim s), or cleanly on an abort the play loop cannot recover from.
   Advance kicks: the camera sees the 20 mm posts to ~2.5 m (measured: fit error <= 28 mm at 2.5 m,
   detection collapses past 3 m) but the field is 6 m long. A ball farther than GOAL_CAM_RANGE_M from the
   goal (estimated) is kicked toward the goal's FIELD-MAP spot through the estimated pose, and so is a
   "tap-in" (ball within 0.6 m of the goal: from the line-up point the posts overflow the frame); in
   between, the Stage 4 path (the goal seen by the camera at the pre-stance look, the heading from the
   perceived goal; not seen after the sweep = abort goal_not_found) is unchanged. SCAN looks for the ball
   only; the route is planned on the map heading until the goal look replaces it.
   Ball physics: the sim ball has no rolling resistance (MuJoCo rolling friction barely slows it: a 0.4 m/s
   ball still rolls 3+ m in 10 s and never stops), so "re-approach where it stopped" never happens.
   BallDrag adds a rolling-resistance deceleration (--ball-decel, default 0.03 m/s^2; measured roll per kick
   in the report). Play only; every earlier battery is untouched.
   Routine landmark fix (beyond the brief, --no-routine-fix turns it off): odometry + IMU drift grows
   without bound over a multi-minute run (a 180 s smoke run ended 834 mm off, past the 0.6 m target gate),
   so the fall protocol's landmark fix is also taken while standing in WATCH after each kick, when a fixed
   goal is in view within ROUTINE_MAX_RANGE.

C. FALL RE-LOCALIZATION. A fall (estimated tilt > 35 deg or trunk < 6 cm: IMU, never sim state) marks the
   duck NOT LOCALIZED and cancels the mission (a fall during KICK/SETTLE lets that kick finish first). When
   VelStand has it up (upright 1 s): stop; face the nearer goal (by the suspect estimate; walk toward it
   if farther than RELOC_LOOK_RANGE, allowed under the staleness rules); head sweep 0/+0.5/-0.5 rad with
   goal looks. Each sighting of a FIXED goal (identified by its facing through the IMU heading; within
   RELOC_MAX_RANGE) gives a position: landmark - R(yaw) * measured centre. Enough sightings -> the median
   is the fix, applied as a layer over the odometry (and over mapd's offset, which keeps working). mapd
   (--pose mapd) is the secondary source: it breaks its chain on a fall ("fell") and, if a stop matches
   its pre-fall keyframes, relocalizes; that pose is absolute and supersedes the landmark layer. No fix
   after RELOC_MAX_ATTEMPTS (alternating goals) = the run ends "not_localized". No mission (so no kick)
   runs while not localized.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kick_mission as km  # noqa: E402
from goto_point import CONTROL_DT, RunLog  # noqa: E402
from goto_supervisor import GotoConfig, GotoSupervisor, wrap_angle  # noqa: E402
from kick_battery import SCENES, KickSim  # noqa: E402
from pose_provider import ODOM_CALIBS, make_provider, pose_error  # noqa: E402
from soccer_referee import Goal  # noqa: E402
from truth_guard import TruthGuard  # noqa: E402

BALL_DECEL = 0.03            # m/s^2 rolling resistance (play only): a ~0.4 m/s kick rolls ~1.2 m
PLAY_INSET_M = 0.20          # the duck's geofence in play: the field inset by this (Stage 4 missions: 0.5)
SPAWN_INSET_M = 0.60         # a respawned ball lands this far inside the boards (a ball nearer a board than
                             # ~0.5 m may need a stance spot outside the geofence) ...
SPAWN_CLEAR_M = 0.60         # ... and this far from the duck
TIME_CAP_S = 240.0
BALL_KICK_CAP = 6
MISSION_CAP_S = 90.0         # one mission (scan .. kick .. settle) longer than this is cancelled and restarted
GOAL_CAM_RANGE_M = 2.5
WATCH_MAX_S, WATCH_GAP_S, WATCH_STILL_M = 10.0, 0.6, 0.02
WATCH_LOST_S = 2.0
FALL_TILT, FALL_Z, UP_TILT, UP_Z, UP_SETTLE_S = 35.0, 0.06, 25.0, 0.09, 1.0   # GotoConfig's fall rule
RELOC_SETTLE_S = 0.5
RELOC_LOOK_RANGE = 1.3       # farther than this from the landmark (estimated): walk to 0.2 m inside it
RELOC_MAX_RANGE = 2.6        # sightings farther than this are not used
RELOC_HEAD_YAWS = (0.0, 0.5, -0.5)
RELOC_HEAD_SETTLE_S = 0.4
RELOC_LOOKS = 3
RELOC_MIN_SIGHTINGS = 2
RELOC_FACING_TOL = math.radians(40)
RELOC_GATE_M = 1.2           # a sighting's position must be within this of the suspect estimate
RELOC_MAX_ATTEMPTS = 4
RELOC_FACE_TOL = math.radians(12)
RELOC_YAW_RANGE = 1.3        # the goal's measured facing also fixes the heading: sightings this close ...
                             # (measured facing error max 1.3 deg at 1.2 m, 6.3 at 1.6, 7.7 at 2.0; and one
                             # viewpoint's sightings share their bias: 9 looks at 2.19 m agreed on a 5 deg error)
RELOC_YAW_MAD = math.radians(1.5)   # ... at least 3, agreeing (median absolute deviation) within this
ROUTINE_MAX_RANGE = 2.0      # routine fix (standing in WATCH after a kick): only sightings this close
ROUTINE_MIN_SIGHTINGS = 3
ROUTINE_HEADING_RANGE = 3.0  # heading-only routine fix: the BEARING to a fixed goal through the map spot (any
                             # goal the detector finds; the post columns are the precise part of the fit). The
                             # heading error it leaves is ~position error / range; IMU drift reached 16 deg.
SIGHT_MAX_TILT, SIGHT_MAX_V = 8.0, 0.05   # sightings only from a duck standing upright and still
MAX_YAW_JUMP = math.radians(10)   # a fix may not turn the heading more than this (the IMU drifts ~2 deg/min and a
                             # fall does not corrupt it; measured: 4 agreeing looks at 0.95 m put the facing 36 deg off)
ROUTINE_MAX_JUMP_M = 0.5     # a routine (no-fall) fix may not move the position more than this
SIGHT_MIN_RANGE = 0.9        # and not too close: below ~0.9 m a post sits near the frame edge and the fitted
                             # facing goes wrong (measured: 4 of 25 fixes from 0.58-0.87 m moved the estimate
                             # to 156-340 mm off; Stage 4 ranging: 2/15 not found at 0.6 m)
WATCH_HEAD_YAWS = (0.0, -0.6, 0.6)   # the kicked ball can leave the view: sweep while watching
WATCH_HEAD_STEP_S = 0.8


@dataclass
class PlayConfig:
    goal_map: tuple                      # the attacked goal's field-map spot (goal-line centre)
    goal_cam_range: float = GOAL_CAM_RANGE_M
    tap_in_m: float = 0.60


# --- field, environment side -----------------------------------------------------------------------

def fixed_goals(sim):
    """The two goals as setup_field locked them (goal_a +x, goal_b -x, facing out of the field)."""
    m, d = sim.model, sim.data
    out = {}
    for key, name, into in (("+x", "goal_a", 0.0), ("-x", "goal_b", math.pi)):
        lp, rp = d.geom_xpos[m.geom(f"{name}_post_left").id], d.geom_xpos[m.geom(f"{name}_post_right").id]
        r = float(m.geom_size[m.geom(f"{name}_post_left").id][0])
        out[key] = Goal.from_posts(lp[:2], rp[:2], r, into)
    return out


def field_landmarks(sim):
    """The FIELD MAP (what the duck knows): each goal-line centre and its facing, from the field's own
    definition (field.xml's board faces), never from the goal bodies."""
    fx0, fx1, _, _ = km.field_from_model(sim.model)
    return {"+x": (fx1, 0.0, 0.0), "-x": (fx0, 0.0, math.pi)}


class BallDrag:
    """Rolling resistance the sim ball lacks: a constant deceleration opposing its planar velocity
    (linear below 2 cm/s so it comes to rest without chatter). Force at the centre of a rolling hollow
    sphere decelerates it by F / (m (1 + 2/3))."""

    def __init__(self, sim, decel):
        self.sim, self.decel = sim, decel
        self.body = sim.model.geom_bodyid[sim.model.geom("ball_geom").id]
        m = float(sim.model.body_mass[self.body])
        self.force = m * (5.0 / 3.0) * decel

    def apply(self):
        d, s = self.sim.data, self.sim
        v = d.qvel[s.bva:s.bva + 2]
        sp = float(np.linalg.norm(v))
        f = -self.force * v / sp if sp > 0.02 else -(self.force / 0.02) * v
        d.xfrc_applied[self.body, :2] = f if self.decel > 0 else 0.0


class PlayReferee:
    """Ground truth, environment side: goals at either end, out of bounds (ball on a board)."""

    def __init__(self, sim, goals):
        self.sim, self.goals = sim, goals
        m = sim.model
        self.boards = {g for g in sim.board_geoms if not (m.geom(g).name or "").startswith("field_goalbox")}
        self.ball_geom = m.geom("ball_geom").id
        self.field = sim.field

    def update(self):
        b, _ = self.sim.ball()
        for key, g in self.goals.items():
            along, lat = g.to_frame(float(b[0]), float(b[1]))
            if along >= 0.0 and abs(lat) < g.half_opening + g.post_radius:
                return {"event": "goal", "end": key, "lateral_mm": round(lat * 1000, 1)}
        fx0, fx1, fy0, fy1 = self.field
        if not (fx0 <= b[0] <= fx1 and fy0 <= b[1] <= fy1):      # pushed through a board (soft contact)
            return {"event": "out", "board": "past the line", "at": [round(float(v), 3) for v in b]}
        d = self.sim.data
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if (g1 == self.ball_geom and g2 in self.boards) or (g2 == self.ball_geom and g1 in self.boards):
                other = g2 if g1 == self.ball_geom else g1
                return {"event": "out", "board": self.sim.model.geom(other).name,
                        "at": [round(float(v), 3) for v in b]}
        return None


def spawn_ball(sim, rng, field):
    """A fresh ball anywhere in the field (inset), clear of the duck. Environment side: truth is fine."""
    fx0, fx1, fy0, fy1 = field
    (x, y, _), _, _ = sim.pose()
    for _ in range(200):
        b = (rng.uniform(fx0 + SPAWN_INSET_M, fx1 - SPAWN_INSET_M), rng.uniform(fy0 + SPAWN_INSET_M, fy1 - SPAWN_INSET_M))
        if math.hypot(b[0] - x, b[1] - y) >= SPAWN_CLEAR_M:
            break
    sim.data.qpos[sim.bqa:sim.bqa + 7] = [b[0], b[1], 0.035, 1, 0, 0, 0]
    sim.data.qvel[sim.bva:sim.bva + 6] = 0
    mujoco.mj_forward(sim.model, sim.data)
    return b


# --- the duck's supervisor: estimated inputs only ---------------------------------------------------

class PlaySupervisor:
    """Runs one Stage 4 mission at a time, watches the ball between kicks, and owns the fall protocol.
    Reads pose / ball / goal through providers; ground truth only inside _truth() (scoring, logs)."""

    def __init__(self, sim, kick, precision, foot, pose_src, ball_src, goal_src, attack_goal, attack_key,
                 landmarks, log, estimated_only=True, hide_goal=None, routine_fix=True, single=False, target=None,
                 mission_play=True):
        """single (Stage 4 replay, kick_mission.py --reloc): one ball, one kick, then OVER; the missions are
        Stage 4 missions (mission_play=False) aimed at `target` (the placed goal's assignment)."""
        self.sim, self.kick, self.precision, self.foot = sim, kick, precision, foot
        self.pose_src, self.ball_src, self.goal_src = pose_src, ball_src, goal_src
        self.attack_goal, self.attack_key, self.landmarks = attack_goal, attack_key, landmarks
        self.cfg = PlayConfig(goal_map=landmarks[attack_key][:2] if attack_key in landmarks else
                              (attack_goal.cx, attack_goal.cy))   # single (replay): the placed goal
        self.log, self.estimated_only, self.hide_goal = log, estimated_only, hide_goal
        self.routine_fix, self.routine = routine_fix, []
        self.pending_refs = []           # (GoalReferee, kick record): fed by the ENVIRONMENT (play loop), not here
        self.walk_session = sim.policy.ort_session
        self.localized = True
        self.need_reloc = False
        self.over = None                 # why the run ended early (None = still playing / time cap)
        self.kicks, self.kicks_unlocalized, self.ball_kicks = [], 0, 0
        self.falls, self.reloc_falls, self.relocs = 0, 0, []
        self.request = None              # to the referee: "unplayable" | "kick_cap"
        self.missions = 0
        self.mission_aborts = {}
        self.searches = 0
        self._was_kick = False
        self.single, self.target, self.mission_play = single, target, mission_play
        self.last_result = None
        self._new_mission(survey=True)

    def _truth(self):
        allow = getattr(self.sim, "allow_truth", None)
        return allow() if allow is not None else km._NoGuard()

    def _ev(self, event, **kw):
        self.log.event({"t": round(self.sim.t, 3), "event": event, **kw})

    def _new_mission(self, survey=False):
        from camera_goal import assign_target
        self.sim.policy.ort_session = self.walk_session
        self.sim.policy.head_offset[:] = 0.0
        self.ball_src.reset()
        self.goal_src.reset(target=self.target or assign_target(self.cfg.goal_map), hide=self.hide_goal)
        self.mission = km.Mission(self.sim, self.kick, None, self.log, foot=self.foot, precision_session=self.precision,
                                  goal=self.attack_goal, pose_src=self.pose_src, ball_src=self.ball_src,
                                  goal_src=self.goal_src, estimated_only=self.estimated_only,
                                  play=self.cfg if self.mission_play else None,
                                  survey=survey and getattr(self.pose_src, "name", "") == "mapd")
        self.missions += 1
        self.t_mission = self.sim.t
        self._was_kick = False
        self.state = "MISSION"

    def _cancel_mission(self):
        self.sim.policy.ort_session = self.walk_session
        self.sim.policy.head_offset[:] = 0.0
        self.mission = None

    def whistle(self, event):
        """The referee restarted play (goal / out / dead ball): the ball is somewhere new."""
        self.ball_kicks = 0
        self._ev("WHISTLE", **event)
        if self.state in ("MISSION", "WATCH"):
            self._cancel_mission()
            self._new_mission()
        # DOWN / RELOC: the protocol finishes first, then a fresh mission starts anyway

    # --- per tick ---
    def tick(self):
        sim, t = self.sim, self.sim.t
        est = self.pose_src.get(t)
        if est.pose is None and (self.state != "MISSION" or self.mission.state in ("KICK", "SETTLE")):
            # (a mission applies its own, identical rule -- except in KICK/SETTLE, where it would wait forever
            # for an "up" it cannot see)
            self._pose_lost_t = getattr(self, "_pose_lost_t", None) or t
            if t - self._pose_lost_t > km.POSE_LOST_ABORT_S:
                self._ev("POSE_LOST_ABORT", phase=self.state, after_s=round(t - self._pose_lost_t, 2))
                self.over, self.state = "pose_lost", "OVER"
            return (0.0, 0.0, 0.0)
        self._pose_lost_t = None
        if est.pose is not None and est.tilt_deg is not None:
            fallen = est.tilt_deg > FALL_TILT or est.trunk_z < FALL_Z
            if fallen and self.state != "DOWN" and not self.need_reloc:
                in_kick = self.state == "MISSION" and self.mission.state in ("KICK", "SETTLE")
                self.falls += 1
                if self.state == "RELOC":
                    self.reloc_falls += 1
                with self._truth():
                    true_pose, _, _ = sim.pose()
                err = pose_error(est.pose, true_pose)
                self.localized = False
                self._fall = {"t_fall": round(t, 2), "phase": self.mission.state if self.state == "MISSION" else self.state,
                              "err_at_fall_mm": round(err[0] * 1000, 1)}
                self._ev("PLAY_FALL", **self._fall)
                if in_kick:
                    self.need_reloc = True       # the kick already happened; Mission scores it, then the protocol
                else:
                    self._cancel_mission()
                    self.state, self._up_since = "DOWN", None
        if self.state == "MISSION":
            return self._tick_mission(t)
        if self.state == "WATCH":
            return self._tick_watch(t)
        if self.state == "DOWN":
            return self._tick_down(t, est)
        if self.state == "RELOC":
            return self._tick_reloc(t, est)
        if self.state == "SEARCH":
            return self._tick_search(t, est)
        return (0.0, 0.0, 0.0)

    def _tick_mission(self, t):
        m = self.mission
        if t - self.t_mission > MISSION_CAP_S and m.state not in ("KICK", "SETTLE"):
            self.mission_aborts["mission_cap"] = self.mission_aborts.get("mission_cap", 0) + 1
            self._ev("MISSION_CAP", phase=m.state)
            self._cancel_mission()
            self._new_mission()
            return (0.0, 0.0, 0.0)
        twist = m.tick()
        if m.state == "KICK" and not self._was_kick:
            self._was_kick = True
            if not self.localized:
                self.kicks_unlocalized += 1
        if m.state != "DONE":
            return twist
        r = m.result
        self.last_result = r
        kicked = "t_swap" in r
        if kicked:
            self.ball_kicks += 1
            rec = {k: r.get(k) for k in ("t_swap", "aim_mode", "aim_why", "aim_err_deg", "pose_err_yaw_deg", "pose_err_mm", "kick_dir_err_deg",
                                         "peak_ball_fwd", "approach_mode", "falls")}
            self.kicks.append(rec)
            if m.referee is not None:
                if m.referee.done:
                    rec.update(m.referee.result)
                else:
                    self.pending_refs.append((m.referee, rec))
        self._cancel_mission()
        if self.single:
            self.over = r.get("aborted") or "done"
            self.state = "OVER"
            return (0.0, 0.0, 0.0)
        if self.need_reloc:
            self.need_reloc = False
            self.state, self._up_since = "DOWN", None
            return (0.0, 0.0, 0.0)
        why = r.get("aborted")
        if why:
            self.mission_aborts[why] = self.mission_aborts.get(why, 0) + 1
            if why == "ball_unplayable":
                self.request = "unplayable"
                self._new_mission()
            elif why == "ball_not_found":
                if self.searches >= 1:
                    self.over = "ball_not_found"
                    self.state = "OVER"
                else:
                    self.searches += 1
                    self._start_search(t)
            elif why in ("goal_not_found", "pose_lost"):
                self.over = why
                self.state = "OVER"
            else:                                # ball_lost (still creeping), anything else: try again
                self._new_mission()
            return (0.0, 0.0, 0.0)
        self.searches = 0
        if self.ball_kicks >= BALL_KICK_CAP:
            self.request = "kick_cap"
            self._new_mission()
            return (0.0, 0.0, 0.0)
        self.state, self.t_watch, self.watch, self.watch_sightings = "WATCH", t, [], []
        return (0.0, 0.0, 0.0)

    def _tick_watch(self, t):
        """Standing after a kick: 5 Hz looks (head yaw swept) until the ball is at rest or out of view; the
        same looks collect fixed-goal sightings for the routine landmark fix."""
        i = int((t - self.t_watch) / WATCH_HEAD_STEP_S) % len(WATCH_HEAD_YAWS)
        self.sim.policy.head_offset[:] = [0.0, 0.0, WATCH_HEAD_YAWS[i], 0.0]
        if t - self.t_watch - i * WATCH_HEAD_STEP_S < 0.3 and int((t - self.t_watch) / WATCH_HEAD_STEP_S) > 0:
            return (0.0, 0.0, 0.0)                          # head still moving
        est = self.pose_src.get(t)
        if self.ball_src.look(t):
            b, _ = self.ball_src.get(t)
            self.watch.append((t, b))
            if self.routine_fix and est.pose is not None and est.tilt_deg < SIGHT_MAX_TILT \
                    and (est.v_xy is None or np.linalg.norm(est.v_xy) < SIGHT_MAX_V):
                self._collect(est.pose, self.watch_sightings, ROUTINE_HEADING_RANGE)
        seen = [w for w in self.watch if t - w[0] <= 2 * WATCH_GAP_S]
        still = False
        if len(seen) >= 2 and seen[-1][0] - seen[0][0] >= WATCH_GAP_S - 1e-9:
            r = 1.0 if est.pose is None else math.hypot(seen[-1][1][0] - est.pose[0], seen[-1][1][1] - est.pose[1])
            still = np.linalg.norm(seen[-1][1] - seen[0][1]) < WATCH_STILL_M + 0.02 * r
        lost = (not self.watch and t - self.t_watch > WATCH_LOST_S + WATCH_HEAD_STEP_S * len(WATCH_HEAD_YAWS)) or \
               (self.watch and t - self.watch[-1][0] > WATCH_LOST_S + WATCH_HEAD_STEP_S * len(WATCH_HEAD_YAWS))
        if still or lost or t - self.t_watch > WATCH_MAX_S:
            self.sim.policy.head_offset[:] = 0.0
            self._ev("WATCH_DONE", why="still" if still else ("lost" if lost else "cap"), after_s=round(t - self.t_watch, 2))
            if self.routine_fix and est.pose is not None:
                self._landmark_fix(t, est, self.watch_sightings, ROUTINE_MIN_SIGHTINGS, "routine")
            self._new_mission()
        return (0.0, 0.0, 0.0)

    # --- landmark fixes (C, and the routine fix) ---
    def _heading_fix(self, t, est, S):
        """Routine, no close goal: heading only, from the bearings to a fixed goal through its map spot
        (median of >= ROUTINE_MIN_SIGHTINGS agreeing within RELOC_YAW_MAD). Position unchanged."""
        if len(S) < ROUTINE_MIN_SIGHTINGS:
            return None
        ys = np.unwrap([s_["yaw_b"] for s_ in S])
        med = float(np.median(ys))
        x, y, yaw0 = est.pose
        if float(np.median(np.abs(ys - med))) > RELOC_YAW_MAD or abs(wrap_angle(med - yaw0)) > MAX_YAW_JUMP:
            return None
        with self._truth():
            true_pose, _, _ = self.sim.pose()
        before = pose_error(est.pose, true_pose)
        self.pose_src.apply_fix((x, y, float(wrap_angle(med))))
        after = pose_error(self.pose_src.get(t).pose, true_pose)
        rec = {"t": round(t, 1), "kind": "heading", "landmark": S[0]["key"], "n": len(S),
               "range_m": round(float(np.median([s_["rng"] for s_ in S])), 2), "yaw_fixed": True, "jump_mm": 0.0,
               "err_before_mm": round(before[0] * 1000, 1), "err_after_mm": round(after[0] * 1000, 1),
               "yaw_err_before_deg": round(math.degrees(before[1]), 2), "yaw_err_after_deg": round(math.degrees(after[1]), 2)}
        self.routine.append(rec)
        self._ev("ROUTINE_FIX", **rec)
        return rec

    def _collect(self, pose, out, max_range, only=None):
        """One goal look: every FIXED goal in view (identified by its facing through the IMU heading),
        as (landmark key, the duck pose it implies, range, yaw implied by the goal's facing)."""
        x, y, yaw = pose
        for g in self.goal_src.landmark_sightings():
            cx, cy = g["centre"]
            rng = math.hypot(cx, cy)
            if rng > max_range or rng < SIGHT_MIN_RANGE:
                continue
            into_w = wrap_angle(yaw + g["into"])
            for key, L in self.landmarks.items():
                if (only is not None and key != only) or abs(wrap_angle(into_w - L[2])) > RELOC_FACING_TOL:
                    continue
                c, s_ = math.cos(yaw), math.sin(yaw)
                p = (L[0] - (c * cx - s_ * cy), L[1] - (s_ * cx + c * cy))
                if math.hypot(p[0] - x, p[1] - y) <= RELOC_GATE_M:
                    out.append({"key": key, "p": p, "rng": rng, "yaw_g": wrap_angle(L[2] - g["into"]), "rel": (cx, cy),
                                "yaw_est": yaw, "yaw_b": wrap_angle(math.atan2(L[1] - y, L[0] - x) - math.atan2(cy, cx))})

    def _landmark_fix(self, t, est, S, min_n, kind):
        """Median of the sightings of the most-seen landmark -> the fix: heading from the goal's measured
        facing (median, when >= 3 close sightings agree), position = landmark - R(heading) * measured centre.
        Without a heading fix the IMU heading is used -- after a fall only: a ROUTINE fix needs the heading
        (measured: a position-only routine fix under a 3.8 deg IMU drift at 1.85 m moved the estimate
        40 -> 110 mm off). Returns the record or None."""
        if not S:
            return None
        key = max({s_["key"] for s_ in S}, key=lambda k: sum(1 for s_ in S if s_["key"] == k))
        S = [s_ for s_ in S if s_["key"] == key]
        if len(S) < min_n:
            return None
        yaw = est.pose[2]
        near = [s_ for s_ in S if s_["rng"] <= RELOC_YAW_RANGE]
        yaw_fixed = False
        if len(near) >= 3:
            ys = np.unwrap([s_["yaw_g"] for s_ in near])
            med = float(np.median(ys))
            if float(np.median(np.abs(ys - med))) <= RELOC_YAW_MAD and abs(wrap_angle(med - yaw)) <= MAX_YAW_JUMP:
                # heading from the goal's facing; positions re-derived with it (the duck stood still)
                yaw = float(wrap_angle(med))
                yaw_fixed = True
        if kind == "routine" and not yaw_fixed:
            return self._heading_fix(t, est, S)
        if kind == "routine":
            S = [s_ for s_ in S if s_["rng"] <= ROUTINE_MAX_RANGE]
        L = self.landmarks[key]
        ps = []
        for s_ in S:
            cx, cy = s_["rel"]
            c, sn = math.cos(yaw), math.sin(yaw)
            ps.append((L[0] - (c * cx - sn * cy), L[1] - (sn * cx + c * cy)))
        fx, fy = float(np.median([p[0] for p in ps])), float(np.median([p[1] for p in ps]))
        if kind == "routine" and math.hypot(fx - est.pose[0], fy - est.pose[1]) > ROUTINE_MAX_JUMP_M:
            self._ev("ROUTINE_FIX_REFUSED", jump_mm=round(1000 * math.hypot(fx - est.pose[0], fy - est.pose[1]), 1))
            return None
        with self._truth():
            true_pose, _, _ = self.sim.pose()
        before = pose_error(est.pose, true_pose)
        jump = math.hypot(fx - est.pose[0], fy - est.pose[1])
        self.pose_src.apply_fix((fx, fy, yaw))
        after = pose_error(self.pose_src.get(t).pose, true_pose)
        rec = {"t": round(t, 1), "kind": kind, "landmark": key, "n": len(S), "range_m": round(float(np.median([s_["rng"] for s_ in S])), 2),
               "yaw_fixed": yaw_fixed, "jump_mm": round(jump * 1000, 1),
               "err_before_mm": round(before[0] * 1000, 1), "err_after_mm": round(after[0] * 1000, 1),
               "yaw_err_before_deg": round(math.degrees(before[1]), 2), "yaw_err_after_deg": round(math.degrees(after[1]), 2)}
        if kind == "routine":
            self.routine.append(rec)
            self._ev("ROUTINE_FIX", **rec)
        return rec

    def _start_search(self, t):
        """Ball not found by a full turn: walk toward the field centre (estimated) and look again."""
        est = self.pose_src.get(t)
        self._ev("SEARCH_START")
        self.nav = GotoSupervisor([(0.0, 0.0)], self.sim.arena, GotoConfig(arrival_radius=0.3), log=self.log.event)
        self.state = "SEARCH"
        self.t_search = t

    def _tick_search(self, t, est):
        if est.pose is None:
            return (0.0, 0.0, 0.0)
        self.nav.update(est.t, est.pose, est.tilt_deg, est.trunk_z)
        tw = self.nav.command(t)
        if self.nav.state == "DONE" or t - self.t_search > 30.0:
            self._new_mission()
            return (0.0, 0.0, 0.0)
        return tw

    def _tick_down(self, t, est):
        if est.pose is None:
            return (0.0, 0.0, 0.0)
        up = est.tilt_deg < UP_TILT and est.trunk_z > UP_Z
        self._up_since = (self._up_since if self._up_since is not None else t) if up else None
        if self._up_since is not None and t - self._up_since >= UP_SETTLE_S:
            self._start_reloc(t, est)
        return (0.0, 0.0, 0.0)

    # --- C. fall re-localization ---
    def _start_reloc(self, t, est):
        with self._truth():
            true_pose, _, _ = self.sim.pose()
        err = pose_error(est.pose, true_pose)
        self.reloc = {"t0": t, "attempt": 0, "phase": "settle", "t_phase": t, "falls0": self.reloc_falls,
                      "mapd0": getattr(self.pose_src, "relocalizations", 0),
                      "rec": {**getattr(self, "_fall", {}), "err_before_mm": round(err[0] * 1000, 1),
                              "yaw_err_before_deg": round(math.degrees(err[1]), 2)}}
        self.state = "RELOC"
        self._ev("RELOC_START", err_before_mm=self.reloc["rec"]["err_before_mm"])

    def _reloc_phase(self, phase, t):
        self.reloc["phase"], self.reloc["t_phase"] = phase, t

    def _reloc_done(self, t, est, source, n=0, fix=None):
        with self._truth():
            true_pose, _, _ = self.sim.pose()
        err = pose_error(self.pose_src.get(t).pose, true_pose)
        rec = self.reloc["rec"]
        rec.update(source=source, sightings=n, err_after_mm=round(err[0] * 1000, 1),
                   yaw_err_after_deg=round(math.degrees(err[1]), 2), attempts=self.reloc["attempt"] + 1,
                   protocol_s=round(t - self.reloc["t0"], 1),
                   falls_during=self.reloc_falls - self.reloc["falls0"],
                   yaw_fixed=None if fix is None else fix["yaw_fixed"], range_m=None if fix is None else fix["range_m"])
        self.relocs.append(rec)
        self.localized = True
        self.sim.policy.head_offset[:] = 0.0
        self._ev("RELOC_DONE", **rec)
        self._new_mission()

    def _target_landmark(self, est):
        """Nearer goal first (by the suspect estimate), then alternate."""
        x, y, _ = est.pose
        keys = sorted(self.landmarks, key=lambda k: math.hypot(self.landmarks[k][0] - x, self.landmarks[k][1] - y))
        return keys[self.reloc["attempt"] % 2]

    def _tick_reloc(self, t, est):
        R = self.reloc
        if est.pose is None:
            return (0.0, 0.0, 0.0)
        if getattr(self.pose_src, "relocalizations", 0) > R["mapd0"]:      # secondary: mapd recognised the place
            self._reloc_done(t, est, "mapd")
            return (0.0, 0.0, 0.0)
        x, y, yaw = est.pose
        if R["phase"] == "settle":
            if t - R["t_phase"] >= RELOC_SETTLE_S:
                R["target"] = self._target_landmark(est)
                R["approached"] = False
                self._reloc_phase("face", t)
            return (0.0, 0.0, 0.0)
        L = self.landmarks[R["target"]]
        if R["phase"] == "face":
            err = wrap_angle(math.atan2(L[1] - y, L[0] - x) - yaw)
            if abs(err) > RELOC_FACE_TOL and t - R["t_phase"] < 8.0:
                return (0.0, 0.0, math.copysign(0.8, err))
            dist = math.hypot(L[0] - x, L[1] - y)
            if dist > RELOC_LOOK_RANGE and not R["approached"]:
                R["approached"] = True
                k = (np.array([x, y]) - np.array(L[:2])) / max(dist, 1e-6)
                wp = np.array(L[:2]) + (RELOC_LOOK_RANGE - 0.2) * k
                fx0, fx1, fy0, fy1 = self.sim.arena
                wp = (float(np.clip(wp[0], fx0, fx1)), float(np.clip(wp[1], fy0, fy1)))
                self.nav = GotoSupervisor([wp], self.sim.arena, GotoConfig(arrival_radius=0.15), log=self.log.event)
                self._ev("RELOC_APPROACH", target=R["target"], to=[round(v, 2) for v in wp])
                self._reloc_phase("approach", t)
                return (0.0, 0.0, 0.0)
            R["head_i"], R["sightings"], R["looks"] = 0, [], 0
            self.sim.policy.head_offset[:] = [0.0, 0.0, RELOC_HEAD_YAWS[0], 0.0]
            self._reloc_phase("survey_settle", t)
            return (0.0, 0.0, 0.0)
        if R["phase"] == "approach":
            self.nav.update(est.t, est.pose, est.tilt_deg, est.trunk_z)
            tw = self.nav.command(t)
            if self.nav.state == "DONE" or t - R["t_phase"] > 25.0:
                self._reloc_phase("face", t)
                return (0.0, 0.0, 0.0)
            return tw
        if R["phase"] == "survey_settle":
            if t - R["t_phase"] >= RELOC_HEAD_SETTLE_S:
                self._reloc_phase("survey_look", t)
                R["looks"] = 0
            return (0.0, 0.0, 0.0)
        if R["phase"] == "survey_look":
            if t - R["t_phase"] >= 0.1 * R["looks"] - 1e-9:
                R["looks"] += 1
                self._collect(est.pose, R["sightings"], RELOC_MAX_RANGE, only=R["target"])
            if R["looks"] < RELOC_LOOKS:
                return (0.0, 0.0, 0.0)
            R["head_i"] += 1
            if R["head_i"] < len(RELOC_HEAD_YAWS):
                self.sim.policy.head_offset[:] = [0.0, 0.0, RELOC_HEAD_YAWS[R["head_i"]], 0.0]
                self._reloc_phase("survey_settle", t)
                return (0.0, 0.0, 0.0)
            self.sim.policy.head_offset[:] = 0.0
            fix = self._landmark_fix(t, est, R["sightings"], RELOC_MIN_SIGHTINGS, "reloc")
            if fix is not None:
                self._reloc_done(t, est, f"goal{R['target']}", fix["n"], fix)
                return (0.0, 0.0, 0.0)
            R["attempt"] += 1
            self._ev("RELOC_NO_FIX", target=R["target"], sightings=len(R["sightings"]), attempt=R["attempt"])
            if R["attempt"] >= RELOC_MAX_ATTEMPTS:
                R["rec"].update(source=None, attempts=R["attempt"], protocol_s=round(t - R["t0"], 1),
                                falls_during=self.reloc_falls - R["falls0"])
                self.relocs.append(R["rec"])
                self.over = "not_localized"
                self.state = "OVER"
                self._ev("NOT_LOCALIZED_ABORT", **R["rec"])
                return (0.0, 0.0, 0.0)
            self._reloc_phase("settle", t)
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.0)


# --- runs ------------------------------------------------------------------------------------------

class FallScript:
    """Scripted falls (Workstream C acceptance): once per run, when the supervisor's mission has been in
    `phase` for `after_s` (or the play state is `phase`), push the trunk sideways at push_v until it falls
    (re-pushing up to 3 times). Environment side: truth is fine."""

    def __init__(self, phase, push_v=2.0, after_s=0.5, not_before_s=5.0):
        self.phase, self.push_v, self.after_s, self.not_before = phase, push_v, after_s, not_before_s
        self.done, self.t_in, self.pushes, self.t_push = False, None, 0, None

    def tick(self, sim, sup):
        if self.done or sim.t < self.not_before:
            return
        ph = sup.mission.state if sup.state == "MISSION" and sup.mission is not None else sup.state
        if self.t_push is not None:
            if sup.falls > 0 or sup.state == "DOWN":
                self.done = True
            elif sim.t - self.t_push > 1.5 and self.pushes < 3:
                self._push(sim)
            elif self.pushes >= 3 and sim.t - self.t_push > 1.5:
                self.done = True
            return
        if ph != self.phase:
            self.t_in = None
            return
        self.t_in = self.t_in if self.t_in is not None else sim.t
        if sim.t - self.t_in >= self.after_s:
            self._push(sim)

    def _push(self, sim):
        (_, _, yaw), _, _ = sim.pose()
        a = yaw + math.pi / 2 * (1 if self.pushes % 2 == 0 else -1)
        sim.push(self.push_v * math.cos(a), self.push_v * math.sin(a))
        self.pushes += 1
        self.t_push = sim.t


def setup_play(args):
    sim = KickSim(args.velstand, SCENES["field"])
    sim.reset(0, 0, 0)
    km.set_feet(sim, args.foot)
    km.setup_goal(sim)
    km.setup_field(sim)                                  # goal_a -> +x mouth, goal_b -> -x mouth: LOCKED
    fx0, fx1, fy0, fy1 = sim.field
    sim.arena = (fx0 + PLAY_INSET_M, fx1 - PLAY_INSET_M, fy0 + PLAY_INSET_M, fy1 - PLAY_INSET_M)
    m = sim.model                                       # post contacts: both goals' posts
    sim.post_geoms = {m.geom(f"{g}_post_{s}").id for g in ("goal_a", "goal_b") for s in ("left", "right")}
    goals = fixed_goals(sim)
    lm = field_landmarks(sim)
    for k in goals:                                     # the map and the placed goals agree (set by setup_field)
        assert abs(goals[k].cx - lm[k][0]) < 1e-6 and abs(goals[k].cy - lm[k][1]) < 1e-6, (k, goals[k], lm[k])
    return sim, goals, lm


def run_one(args, k, sim, goals, lm, kick, precision, pose_src, ball_cam, goal_cam, rng, drag, out, viewer=None):
    """One continuous-play run. Returns its record."""
    from camera_goal import CameraGoal  # noqa: F401  (imported for the side effect of the module check)
    attack = args.attack if args.attack in ("+x", "-x") else ("+x" if k % 2 == 0 else "-x")
    sim.reset(0.0, 0.0, rng.uniform(-math.pi, math.pi))   # kickoff: field centre, random facing
    for _ in range(25):                                    # stand up straight before the clock starts
        sim.set_twist((0.0, 0.0, 0.0))
        sim.step()
    sim.t = 0.0
    b0 = spawn_ball(sim, rng, sim.field)
    pose_src.reset(sim, seed=args.seed * 1000 + k)
    pose_src.dropouts = [] if not args.dropout else [tuple(float(v) for v in args.dropout.split(","))]
    ball_cam.pose_src = pose_src
    goal_cam.pose_src = pose_src
    ball_geom = sim.model.geom("ball_geom").id
    ball_rgba = sim.model.geom_rgba[ball_geom].copy()
    log = RunLog(echo=args.verbose)
    guard = TruthGuard(sim) if args.estimated_only else sim
    sup = PlaySupervisor(guard, kick, precision, args.foot, pose_src, ball_cam, goal_cam, goals[attack], attack, lm, log,
                         estimated_only=args.estimated_only, hide_goal=args.hide_goal, routine_fix=not args.no_routine_fix)
    ref = PlayReferee(sim, goals)
    falls_script = FallScript(args.fall_in, args.push_v) if args.fall_in else None
    rec = {"run": k, "attack": attack, "goals_for": 0, "goals_against": 0, "out": 0, "kick_cap": 0, "unplayable": 0,
           "t_first_goal": None, "balls": 1}
    spawns = [b0]
    events = []
    log.event({"t": 0.0, "event": "PLAY_START", "attack": attack, "ball": [round(v, 3) for v in b0]})
    t_wall = time.time()
    while sim.t < args.time_cap and sup.state != "OVER":
        drag.apply()
        if falls_script is not None:
            falls_script.tick(sim, sup)
        twist = sup.tick()
        sim.set_twist(twist)
        if int(round(sim.t / CONTROL_DT)) % 5 == 0:            # 10 Hz trace: truth (logs) + the estimate
            (x, y, yaw), tilt, _ = sim.pose()
            bxy, _ = sim.ball()
            e = pose_src.get(sim.t)
            log.trace.append({"t": round(sim.t, 2), "sup": sup.state, "mission": sup.mission.state if sup.mission else "",
                              "x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3), "tilt": round(tilt, 1),
                              "est_x": None if e.pose is None else round(e.pose[0], 3),
                              "est_y": None if e.pose is None else round(e.pose[1], 3),
                              "est_yaw": None if e.pose is None else round(e.pose[2], 3),
                              "ball_x": round(float(bxy[0]), 3), "ball_y": round(float(bxy[1]), 3),
                              "cmd_vx": round(twist[0], 2), "cmd_vy": round(twist[1], 2), "cmd_wz": round(twist[2], 2),
                              "localized": sup.localized})
        sim.step()
        pose_src.update(sim)
        if sup.pending_refs:                                   # finished kicks still rolling: scored here
            bt, _ = sim.ball()
            bp = km.goal_contacts(sim)[0]
            for item in list(sup.pending_refs):
                if item[0].update(sim.t, float(bt[0]), float(bt[1]), bp) is not None:
                    item[1].update(item[0].result)
                    sup.pending_refs.remove(item)
        ev = ref.update()
        if ev is None and sup.request:
            ev = {"event": sup.request}
        sup.request = None
        if ev is not None:
            ev["t"] = round(sim.t, 2)
            ev["kicks_at_ball"] = sup.ball_kicks
            if ev["event"] == "goal":
                if ev["end"] == attack:
                    rec["goals_for"] += 1
                    if rec["t_first_goal"] is None:
                        rec["t_first_goal"] = round(sim.t, 1)
                else:
                    rec["goals_against"] += 1
            else:
                rec[ev["event"]] += 1
            for item in sup.pending_refs:                      # the ball is gone: a kick still pending is decided
                item[1].update(item[0].result or {"goal_outcome": "goal" if ev["event"] == "goal" and ev["end"] == attack
                                                  else "no_cross"})
            sup.pending_refs.clear()
            events.append(ev)
            log.event({"t": round(sim.t, 3), "event": "REFEREE", **ev})
            b = spawn_ball(sim, rng, sim.field)
            spawns.append(b)
            rec["balls"] += 1
            if args.hide_ball_after_respawn:
                sim.model.geom_rgba[ball_geom] = (0.5, 0.5, 0.5, 1.0)     # the camera's not-found test
            sup.whistle({"why": ev["event"], **({"end": ev["end"]} if "end" in ev else {})})
        if viewer is not None and not viewer(sim, sup, rec):
            break
    sim.model.geom_rgba[ball_geom] = ball_rgba
    sim.policy.ort_session = sup.walk_session
    sim.policy.head_offset[:] = 0.0
    sim.data.xfrc_applied[:] = 0.0
    with sup._truth():
        tp, _, _ = sim.pose()
    e = pose_src.get(sim.t)
    end_err = None if e.pose is None else round(pose_error(e.pose, tp)[0] * 1000, 1)
    gm = lm[attack][:2]
    rec.update(t_end=round(sim.t, 1), over=sup.over or "time_cap", kicks=len(sup.kicks), kicks_unlocalized=sup.kicks_unlocalized,
               falls=sup.falls, reloc_falls=sup.reloc_falls, relocs=sup.relocs, routine=sup.routine, missions=sup.missions,
               mission_aborts=sup.mission_aborts, pose_err_end_mm=end_err,
               kicks_per_ball=[ev["kicks_at_ball"] for ev in events],
               decisions=[(ev["event"] if ev["event"] != "goal" else ("GOAL" if ev["end"] == attack else "own_end"))
                          for ev in events],
               spawn_dist_to_goal=[round(math.hypot(b[0] - gm[0], b[1] - gm[1]), 2) for b in spawns],
               aim_modes={m: sum(1 for kk in sup.kicks if kk.get("aim_mode") == m) for m in ("camera", "map")},
               wall_s=round(time.time() - t_wall, 0))
    if hasattr(pose_src, "summary"):
        rec.update(pose_src.summary())
    rec["kick_list"] = sup.kicks
    rec["events"] = events
    log.save(f"{out}/run{k}")
    return rec


def fmt_run(r):
    return (f"  run {r['run']:2d} attack {r['attack']}: GOALS {r['goals_for']}  other-end {r['goals_against']}  out {r['out']}  "
            f"kick-cap {r['kick_cap']}  unplayable {r['unplayable']} | kicks {r['kicks']} (cam {r['aim_modes']['camera']}, "
            f"map {r['aim_modes']['map']}) | first goal {r['t_first_goal']} s | falls {r['falls']} relocs {len(r['relocs'])} "
            f"unlocalized kicks {r['kicks_unlocalized']} routine fixes {len(r['routine'])} | end {r['over']} at {r['t_end']} s, pose err {r['pose_err_end_mm']} mm "
            f"| decisions {r['decisions']} kicks/ball {r['kicks_per_ball']} | aborts {r['mission_aborts']}"
            + (f" | mapd relocs {r.get('mapd_relocalizations')}" if "mapd_relocalizations" in r else "")
            + "".join(f"\n      reloc: {x}" for x in r["relocs"]))


def summarize(rows, args):
    n = len(rows)
    gf = [r["goals_for"] for r in rows]
    kpg = [kk for r in rows for kk, d in zip(r["kicks_per_ball"], r["decisions"]) if d == "GOAL"]
    print(f"\nPLAY SUMMARY: {n} runs x {args.time_cap:.0f} s cap, {args.foot} foot, pose {args.pose}, ball decel "
          f"{args.ball_decel} m/s^2, kick cap {BALL_KICK_CAP}/ball, mission cap {MISSION_CAP_S:.0f} s")
    print(f"  goals (attacked end): total {sum(gf)}, per run median {np.median(gf):.1f} (min {min(gf)}, max {max(gf)}); "
          f"runs with >= 1 goal {sum(g > 0 for g in gf)}/{n}")
    print(f"  goals in the OTHER end: {sum(r['goals_against'] for r in rows)}; out of bounds {sum(r['out'] for r in rows)}; "
          f"kick-cap restarts {sum(r['kick_cap'] for r in rows)}; unplayable restarts {sum(r['unplayable'] for r in rows)}")
    if kpg:
        print(f"  kicks to score (per scored ball): " + ", ".join(f"{v}: {kpg.count(v)}" for v in sorted(set(kpg)))
              + f" (median {np.median(kpg):.0f})")
    tf = [r["t_first_goal"] for r in rows if r["t_first_goal"] is not None]
    print(f"  time to first goal: median {np.median(tf) if tf else float('nan'):.0f} s over {len(tf)} runs; "
          f"runs hitting the time cap without a goal {sum(1 for r in rows if r['goals_for'] == 0 and r['over'] == 'time_cap')}")
    kk = [x for r in rows for x in r["kick_list"]]
    for mode in ("camera", "map"):
        ks = [x for x in kk if x.get("aim_mode") == mode]
        ae = [abs(x["aim_err_deg"]) for x in ks if x.get("aim_err_deg") is not None]
        print(f"  {mode:6s} kicks {len(ks)}: |aim err| median {np.median(ae) if ae else float('nan'):.1f}, max "
              f"{max(ae) if ae else float('nan'):.1f} deg; per-kick outcomes "
              + str({o: sum(1 for x in ks if x.get('goal_outcome') == o) for o in ('goal', 'post', 'miss', 'no_cross')}))
    print(f"  falls {sum(r['falls'] for r in rows)} (during the protocol {sum(r['reloc_falls'] for r in rows)}); "
          f"re-localizations {sum(len(r['relocs']) for r in rows)}; kicks while NOT localized "
          f"{sum(r['kicks_unlocalized'] for r in rows)}")
    rl = [x for r in rows for x in r["relocs"]]
    if rl:
        ok = [x for x in rl if x.get("source")]
        print(f"  protocol: fixed {len(ok)}/{len(rl)} (sources " + str({s: sum(1 for x in ok if x['source'] == s)
                                                                         for s in sorted({x['source'] for x in ok})}) + ")"
              + (f"; error before fix median {np.median([x['err_before_mm'] for x in rl]):.0f} mm (max "
                 f"{max(x['err_before_mm'] for x in rl):.0f}), after median "
                 f"{np.median([x['err_after_mm'] for x in ok]) if ok else float('nan'):.0f} mm (max "
                 f"{max([x['err_after_mm'] for x in ok]) if ok else float('nan'):.0f}); protocol time median "
                 f"{np.median([x['protocol_s'] for x in rl]):.0f} s"))
    for kind in ("routine", "heading"):
        ro = [x for r in rows for x in r["routine"] if x.get("kind", "routine") == kind]
        if not ro:
            continue
        print(f"  routine {'FULL' if kind == 'routine' else 'HEADING-ONLY'} fixes {len(ro)}: error before median {np.median([x['err_before_mm'] for x in ro]):.0f} mm (max "
              f"{max(x['err_before_mm'] for x in ro):.0f}), after median {np.median([x['err_after_mm'] for x in ro]):.0f} mm "
              f"(max {max(x['err_after_mm'] for x in ro):.0f}); heading fixed too in {sum(x['yaw_fixed'] for x in ro)}; "
              f"|yaw err| before median {np.median([abs(x['yaw_err_before_deg']) for x in ro]):.1f} -> after "
              f"{np.median([abs(x['yaw_err_after_deg']) for x in ro]):.1f} deg")
    ends = {}
    for r in rows:
        ends[r["over"]] = ends.get(r["over"], 0) + 1
    pe = [r["pose_err_end_mm"] for r in rows if r["pose_err_end_mm"] is not None]
    print(f"  run ends: {ends}" + (f"; pose error at the end median {np.median(pe):.0f} mm, max {max(pe):.0f} mm" if pe else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("velstand")
    ap.add_argument("kick")
    ap.add_argument("--foot", choices=("right", "left"), default="right")
    ap.add_argument("--calib", choices=tuple(km.CALIBRATIONS), default="v2")
    ap.add_argument("--precision", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--only-runs", default=None, help="comma list of run indices to play (the rng stream is kept)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-cap", type=float, default=TIME_CAP_S)
    ap.add_argument("--attack", choices=("alt", "+x", "-x"), default="alt")
    ap.add_argument("--pose", choices=("odom", "mapd"), default="odom")
    ap.add_argument("--odom-calib", choices=tuple(ODOM_CALIBS), default="stride_v1")
    ap.add_argument("--no-imu-error", action="store_true")
    ap.add_argument("--profile", default="face-board-v1")
    ap.add_argument("--estimated-only", action="store_true")
    ap.add_argument("--ball-decel", type=float, default=BALL_DECEL)
    ap.add_argument("--fall-in", default=None, help="scripted fall once per run in this phase (NAVIGATE, PRECISE, LOOK, WATCH, ...)")
    ap.add_argument("--push-v", type=float, default=2.0)
    ap.add_argument("--hide-goal", choices=("all", "look"), default=None)
    ap.add_argument("--hide-ball-after-respawn", action="store_true")
    ap.add_argument("--no-routine-fix", action="store_true", help="landmark fixes only after a fall (the brief, literally)")
    ap.add_argument("--dropout", default=None, help='"T0,DUR": pose link down from T0 s into each run for DUR s')
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    km.apply_calibration(args.calib)
    if args.viewer:
        import play_viewer
        return play_viewer.run(args)
    sim, goals, lm = setup_play(args)
    kick, precision = ort.InferenceSession(args.kick), ort.InferenceSession(args.precision)
    from camera_ball import CameraBall
    from camera_goal import CameraGoal
    pose_src = make_provider(args.pose, sim, imu_error=not args.no_imu_error, calib=args.odom_calib)
    ball_cam, goal_cam = CameraBall(sim, pose_src, profile=args.profile), CameraGoal(sim, pose_src, profile=args.profile)
    drag = BallDrag(sim, args.ball_decel)
    out = km.unique_dir(args.out or f"logs/play/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.foot}_{args.pose}{args.tag}")
    print(f"continuous play: field {sim.field}, geofence {sim.arena}, goals " +
          ", ".join(f"{k} ({g.cx:.3f}, {g.cy:.3f}) into {math.degrees(g.yaw):.0f} deg" for k, g in goals.items()) +
          f"\n  foot {args.foot} calib {args.calib}, pose {args.pose} ({args.odom_calib}), estimated-only {args.estimated_only}, "
          f"ball decel {args.ball_decel} m/s^2, time cap {args.time_cap:.0f} s, kick cap {BALL_KICK_CAP}/ball"
          + (f", scripted fall in {args.fall_in} (push {args.push_v} m/s)" if args.fall_in else "")
          + (f", goal hidden: {args.hide_goal}" if args.hide_goal else "")
          + (", ball hidden after the first respawn" if args.hide_ball_after_respawn else "")
          + (f", pose dropout {args.dropout}" if args.dropout else "")
          + (", routine landmark fix OFF" if args.no_routine_fix else ", routine landmark fix on") + f"\n  logs -> {out}/", flush=True)
    rng = np.random.default_rng(args.seed)
    only = None if args.only_runs is None else {int(v) for v in args.only_runs.split(",")}
    rows = []
    import json
    for k in range(args.runs):
        if only is not None and k not in only:
            continue
        rng_k = np.random.default_rng([args.seed, k])     # each run its own stream: --only-runs replays exactly
        r = run_one(args, k, sim, goals, lm, kick, precision, pose_src, ball_cam, goal_cam, rng_k, drag, out)
        rows.append(r)
        print(fmt_run(r), flush=True)
        with open(f"{out}/runs.jsonl", "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
    if rows:
        summarize(rows, args)
    if hasattr(pose_src, "shim"):
        pose_src.shim.close()


if __name__ == "__main__":
    main()
