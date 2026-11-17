"""Continuous play (play.py): fixed goals, referee, ball drag, the landmark fix, play-mode aiming."""
import math
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import play as pl  # noqa: E402
import pose_provider as pp  # noqa: E402

VELSTAND = os.path.join(REPO, "logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx")
KICK_R = os.path.join(REPO, "logs/rsl_rl/ball_kick_v2_right/2026-11-13_18-51-32_v2_right_chunk1/"
                            "2026-11-13_18-51-32_v2_right_chunk1_model_1250.onnx")
PRECISION = os.path.join(REPO, "logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/"
                               "2026-11-13_16-23-21_v2_chunk1_model_2500.onnx")
HAVE_POLICIES = all(os.path.exists(p) for p in (VELSTAND, KICK_R, PRECISION))


def _render_ok():
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def field():
    if not HAVE_POLICIES:
        pytest.skip("policies not present")
    args = SimpleNamespace(velstand=VELSTAND, foot="right")
    return pl.setup_play(args)


def test_goals_are_locked_at_the_ends_and_the_map_agrees(field):
    sim, goals, lm = field
    assert (goals["+x"].cx, goals["+x"].cy, goals["+x"].yaw) == pytest.approx((3.0, 0.0, 0.0), abs=1e-6)
    assert (goals["-x"].cx, goals["-x"].cy) == pytest.approx((-3.0, 0.0), abs=1e-6)
    assert lm["-x"] == pytest.approx((-3.0, 0.0, math.pi))
    assert sim.arena == pytest.approx((-2.8, 2.8, -2.8, 2.8))


def test_referee_goal_out_and_play_on(field):
    sim, goals, _ = field
    ref = pl.PlayReferee(sim, goals)

    def put(x, y):
        sim.data.qpos[sim.bqa:sim.bqa + 7] = [x, y, 0.035, 1, 0, 0, 0]
        sim.data.qvel[sim.bva:sim.bva + 6] = 0
        mujoco.mj_forward(sim.model, sim.data)
        return ref.update()

    assert put(1.0, 1.0) is None
    ev = put(-3.05, 0.02)
    assert ev["event"] == "goal" and ev["end"] == "-x"
    assert put(3.04, -0.05)["end"] == "+x"
    ev = put(1.0, 2.97)                              # touching the side board (inner face at 3.0)
    assert ev is not None and ev["event"] == "out"
    assert put(2.0, 2.90) is None                    # near, not touching


def test_ball_drag_brings_a_rolling_ball_to_rest(field):
    sim, _, _ = field
    sim.reset(-2.5, -2.5, 0.0)
    sim.data.qpos[sim.bqa:sim.bqa + 7] = [-2.0, 0.5, 0.035, 1, 0, 0, 0]
    sim.data.qvel[sim.bva:sim.bva + 6] = [0.4, 0, 0, 0, 0.4 / 0.035, 0]
    mujoco.mj_forward(sim.model, sim.data)
    drag = pl.BallDrag(sim, 0.03)
    for _ in range(int(20 / sim.model.opt.timestep / 4)):   # 20 s at the control rate
        drag.apply()
        for _ in range(4):
            mujoco.mj_step(sim.model, sim.data)
    b, v = sim.ball()
    assert np.linalg.norm(v) < 0.01
    assert 1.0 < b[0] + 2.0 < 3.5                    # rolled, then stopped (the frictionless ball goes 7+ m in 20 s)
    sim.data.xfrc_applied[:] = 0


def test_apply_fix_reanchors_and_composes_with_the_offset():
    class Odo(pp.OdometryProvider):
        def __init__(self):
            pass
    o = Odo()
    o.offset = (0.1, -0.2, 0.05)
    o.fix = None
    xy, yaw = pp.apply_offset(o.offset, np.array([1.0, 2.0]), 0.3)
    o._pre_fix = (float(xy[0]), float(xy[1]), float(yaw))
    o._est = pp.PoseEstimate(5.0, (o._pre_fix[0], o._pre_fix[1], yaw), None, 0.0, 1.0, 0.12, np.zeros(2))
    from collections import deque
    o.hist = deque()
    o.apply_fix((0.5, 0.5, 1.0))
    assert o._est.pose == pytest.approx((0.5, 0.5, 1.0))
    xy2, yaw2 = pp.apply_offset(o.fix, np.array(o._pre_fix[:2]), o._pre_fix[2])   # what the next tick computes
    assert (xy2[0], xy2[1], yaw2) == pytest.approx((0.5, 0.5, 1.0), abs=1e-12)


def _sup(true_pose, est_pose):
    """A PlaySupervisor shell for the landmark-fix maths: fake pose source and truth."""
    s = pl.PlaySupervisor.__new__(pl.PlaySupervisor)
    applied = {}

    class PS:
        def apply_fix(self, p):
            applied["p"] = p

        def get(self, t):
            return SimpleNamespace(pose=applied.get("p", est_pose))
    s.pose_src, s.landmarks, s.routine = PS(), {"+x": (3.0, 0.0, 0.0), "-x": (-3.0, 0.0, math.pi)}, []
    s.sim = SimpleNamespace(pose=lambda: (true_pose, 0.0, 0.12), t=0.0)
    s.log = SimpleNamespace(event=lambda e: None)
    return s, applied


def _sighting(true_pose, L, key, noise=(0.0, 0.0, 0.0)):
    """What the goal camera reports (duck yaw frame) for landmark L seen from the TRUE pose."""
    x, y, yaw = true_pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = (L[0] - x, L[1] - y)
    rel = (c * d[0] + s * d[1] + noise[0], -s * d[0] + c * d[1] + noise[1])
    return {"centre": rel, "into": pl.wrap_angle(L[2] - yaw + noise[2])}


def test_landmark_fix_recovers_position_and_heading_from_a_fixed_goal():
    true = (2.0, 0.5, 0.2)                           # 1.1 m from the +x goal: close enough for a heading fix
    est = (2.35, 0.25, 0.2 + math.radians(4))       # a fall's worth of position error + IMU drift
    sup, applied = _sup(true, est)
    S = []
    rng = np.random.default_rng(0)
    for _ in range(9):
        g = _sighting(true, sup.landmarks["+x"], "+x", (rng.normal(0, 0.008), rng.normal(0, 0.008), rng.normal(0, 0.004)))
        sup.goal_src = SimpleNamespace(landmark_sightings=lambda g=g: [g])
        sup._collect(est, S, pl.RELOC_MAX_RANGE)
    assert len(S) == 9 and {s["key"] for s in S} == {"+x"}     # identified by its facing via the IMU heading
    rec = sup._landmark_fix(0.0, SimpleNamespace(pose=est), S, 2, "reloc")
    assert rec["yaw_fixed"]
    p = applied["p"]
    assert math.hypot(p[0] - true[0], p[1] - true[1]) < 0.03 and abs(p[2] - true[2]) < math.radians(1)


def test_routine_fix_without_a_close_goal_corrects_the_heading_only():
    true = (0.0, 1.0, 0.1)
    est = (0.1, 1.0, 0.1 + math.radians(4))         # IMU heading drift
    sup, applied = _sup(true, est)
    S = []
    for k in range(4):                                # 3.2 m away: no facing-based (full) fix
        g = _sighting(true, sup.landmarks["+x"], "+x", (0, 0, math.radians(4 * (-1) ** k)))
        sup.goal_src = SimpleNamespace(landmark_sightings=lambda g=g: [g])
        sup._collect(est, S, 5.0)
    rec = sup._landmark_fix(0.0, SimpleNamespace(pose=est), S, 3, "routine")
    assert rec["kind"] == "heading"
    p = applied["p"]
    assert p[:2] == est[:2]                           # position untouched
    assert abs(p[2] - true[2]) < math.radians(2.5)    # 4 deg drift -> ~position error / range (0.1 m / 3.2 m)


def test_play_aims_at_the_map_when_the_goal_is_too_far_or_too_close_for_the_camera():
    import kick_mission as km
    m = km.Mission.__new__(km.Mission)
    m.play = pl.PlayConfig(goal_map=(3.0, 0.0))
    assert m._goal_far(np.array([0.0, 0.0]))         # 3 m: beyond the 2.5 m camera range -> advance kick
    assert not m._goal_far(np.array([1.5, 0.3]))     # 1.5 m: the camera goal
    assert m._goal_far(np.array([2.5, 0.05]))        # 0.5 m: tap-in
    m.play = None
    assert not m._goal_far(np.array([0.0, 0.0]))     # Stage 4: never


@pytest.mark.skipif(not (_render_ok() and HAVE_POLICIES), reason="needs offscreen GL and the policies")
def test_a_guarded_play_run_kicks_on_estimated_inputs_and_survives_a_fall():
    import onnxruntime as ort
    import kick_mission as km
    from camera_ball import CameraBall
    from camera_goal import CameraGoal
    km.apply_calibration("v2")
    args = SimpleNamespace(velstand=VELSTAND, foot="right", attack="+x", seed=0, dropout=None, verbose=False,
                           estimated_only=True, hide_goal=None, no_routine_fix=False, fall_in="NAVIGATE", push_v=2.0,
                           time_cap=70.0, hide_ball_after_respawn=False)
    sim, goals, lm = pl.setup_play(args)
    pose = pp.make_provider("odom", sim, calib="stride_v1")
    rec = pl.run_one(args, 1, sim, goals, lm, ort.InferenceSession(KICK_R), ort.InferenceSession(PRECISION), pose,
                     CameraBall(sim, pose), CameraGoal(sim, pose), np.random.default_rng([0, 1]),
                     pl.BallDrag(sim, 0.03), os.path.join(REPO, "logs", "play", "_pytest"))
    assert rec["kicks"] >= 1 and rec["kicks_unlocalized"] == 0
    assert rec["falls"] >= 1 and rec["relocs"] and rec["relocs"][0]["source"]
    assert rec["relocs"][0]["err_after_mm"] < 100 and rec["reloc_falls"] == 0


def test_a_fix_may_not_turn_the_heading_implausibly():
    """4 agreeing looks at ~1 m once put the goal's facing 36 deg off (C_fall_PRECISE_right run 1): refused."""
    true = (2.0, 0.0, 0.0)
    est = (2.0, 0.0, 0.0)
    sup, applied = _sup(true, est)
    S = []
    for _ in range(4):
        g = _sighting(true, sup.landmarks["+x"], "+x", (0, 0, math.radians(-36)))   # a consistent, wrong facing
        sup.goal_src = SimpleNamespace(landmark_sightings=lambda g=g: [g])
        sup._collect(est, S, 5.0)
    rec = sup._landmark_fix(0.0, SimpleNamespace(pose=est), S, 3, "routine")
    assert rec is None or abs(applied["p"][2] - est[2]) <= pl.MAX_YAW_JUMP + 1e-9
