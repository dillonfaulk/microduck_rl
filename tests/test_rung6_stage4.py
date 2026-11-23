"""Rung 6 Stage 4: goal perception, the real mapd in the loop, and the supervisor cutover (truth = scoring only)."""
import json
import math
import os
import sys

import mujoco
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import mapd_shim as ms  # noqa: E402
import pose_provider as pp  # noqa: E402
from truth_guard import TruthAccess, TruthGuard  # noqa: E402

# shipped VelStand (HeadJamRecover 4250, 2026-11-23); the neck clip turns on automatically for it
VELSTAND = os.path.join(REPO, "logs/rsl_rl/velstand/2026-11-22_19-47-54_headjamB/"
                              "2026-11-22_19-47-54_headjamB_model_4250.onnx")
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


# --- Workstream C: truth wired to scoring only ---------------------------------------------------

class _FakeSim:
    def __init__(self):
        self.t, self.policy = 1.5, "p"

    def pose(self):
        return (1.0, 2.0, 0.3), 1.0, 0.12

    def ball(self):
        return np.array([0.5, 0.5]), np.zeros(2)


def test_truth_guard_blocks_truth_outside_scoring_and_passes_the_rest():
    g = TruthGuard(_FakeSim())
    assert g.t == 1.5 and g.policy == "p"          # everything else passes through
    g.t = 2.0
    assert g._sim.t == 2.0
    for read in (g.pose, g.ball):
        with pytest.raises(TruthAccess):
            read()
    with g.allow_truth():
        assert g.pose()[0] == (1.0, 2.0, 0.3)
        assert g.ball()[0].tolist() == [0.5, 0.5]
    assert g.scoring_reads == 2
    with pytest.raises(TruthAccess):               # and the door shuts again
        g.pose()


def test_estimated_only_mission_refuses_ground_truth_inputs():
    import kick_mission as km

    class Src:
        def __init__(self, name):
            self.name = name
    with pytest.raises(TruthAccess):               # no sources given = ground truth defaults
        km.Mission(None, None, None, None, estimated_only=True)
    with pytest.raises(TruthAccess):               # a truth ball
        km.Mission(None, None, None, None, pose_src=Src("odom"), ball_src=Src("gt"), goal_src=Src("camera"),
                   estimated_only=True)
    with pytest.raises(TruthAccess):               # a hand-fed kick heading is the goal, known
        km.Mission(None, None, 0.3, None, pose_src=Src("mapd"), ball_src=Src("camera"), goal_src=Src("camera"),
                   estimated_only=True)


# --- Workstream B: the mapd shim -------------------------------------------------------------------

def test_shim_messages_carry_every_field_mapd_requires():
    st = ms.robot_state(1_000_000_000, 0.0, (0.1, 0.2, 0.12), 0.3, (1, 0, 0, 0), (0, 0, 0), (0.05, 0, 0.1),
                        (1, 0, 0, 0), (0.3, 0, 0), False)
    for k in ("t", "move", "head", "policy", "safety", "loop", "joints", "targets", "odom", "t_ns", "imu", "frames"):
        assert k in st, k                          # RobotState's non-default fields + what maploc reads
    assert st["move"]["requested"] == [0.3, 0.0, 0.0] and st["frames"]["tof"]["pos"] == [0.05, 0.0, 0.1]
    f = ms.tof_frame(5, 1, [40000] * 64, [5] * 64)
    assert f["rows"] == f["cols"] == 8 and max(f["distance_mm"]) == 32767   # i16 on the wire
    line = json.loads(ms.notify("tof.frame", f))
    assert line["jsonrpc"] == "2.0" and "id" not in line


def test_offset_composes_back_to_the_map_pose():
    odom, mp = (0.7, -0.2, 0.4), (0.75, -0.1, 0.43)
    off = ms.offset_from(mp, odom)
    xy, yaw = pp.apply_offset(off, np.array(odom[:2]), odom[2])
    assert xy == pytest.approx(mp[:2], abs=1e-12) and yaw == pytest.approx(mp[2], abs=1e-12)


@pytest.mark.skipif(not os.path.exists(ms.MAPD_BIN), reason="mapd not built (cargo build --release -p mapd)")
def test_real_mapd_subscribes_through_the_autonomy_gate_and_runs_in_lockstep():
    shim = ms.MapdShim()
    try:
        assert "robot.mode" in shim.robot.calls and "robot.subscribe" in shim.robot.calls
        assert shim.tof.calls == ["tof.stream"]
        for k in range(5):
            t_ns = shim.clock(0.04 * k)
            st = shim.send_state(ms.robot_state(t_ns, 0.04 * k, (0, 0, 0.12), 0.0, (1, 0, 0, 0), (0, 0, 0),
                                                (0.05, 0, 0.1), (1, 0, 0, 0), (0, 0, 0), False))
            assert st["state"]["t_ns"] == t_ns                         # processed before we go on
            assert st["state"]["pose"]["x"] == pytest.approx(0.0)      # fresh map: map = odometry
        shim.new_run()
        assert shim.status()["state"]["keyframes"] == 0
    finally:
        shim.close()


# --- Workstream A: goal perception ------------------------------------------------------------------

def test_white_mask_takes_posts_not_floor_lines_or_boards():
    from camera_goal import white_mask
    px = np.array([[[216, 216, 216], [140, 141, 142],      # lit post, its shaded side
                    [130, 131, 133], [216, 184, 132],      # floor edge line (0.8 grey, shaded), tan board
                    [60, 90, 120], [40, 40, 40]]], dtype=np.uint8)
    assert white_mask(px).tolist() == [[True, True, False, False, False, False]]


def test_target_assignment_is_coarse():
    from camera_goal import TARGET_GATE_M, TARGET_GRID_M, assign_target
    assert assign_target((0.61, -1.37)) == pytest.approx((0.5, -1.25))
    assert TARGET_GRID_M == 0.25 and TARGET_GATE_M > TARGET_GRID_M     # the assignment says WHICH, not WHERE


@pytest.mark.skipif(not (_render_ok() and HAVE_POLICIES), reason="needs offscreen GL and the policies")
def test_goal_fit_ranges_a_placed_goal_within_2cm_and_ignores_the_far_field_goal():
    from camera_goal import TARGET_GATE_M, GoalCamera
    from kick_battery import SCENES, KickSim
    from kick_mission import place_goal, setup_field, setup_goal
    sim = KickSim(VELSTAND, SCENES["field"])
    setup_goal(sim)
    setup_field(sim)
    sim.reset(0.0, 0.0, math.pi)                   # facing -x: the field's own goal at x = -3 is in view too
    sim.data.qpos[sim.bqa:sim.bqa + 3] = [9.0, 9.0, 0.035]
    for _ in range(40):
        sim.step()
    (x, y, yaw), _, _ = sim.pose()
    centre = (x + 1.2 * math.cos(yaw + 0.2), y + 1.2 * math.sin(yaw + 0.2))
    cam = GoalCamera(sim.model)                    # applies the board profile to the camera: before mj_forward
    place_goal(sim, centre, yaw + 0.2 + 0.4)
    mujoco.mj_forward(sim.model, sim.data)
    assert len(cam.measure_all(sim.data)) == 2     # both goals seen
    c, s = math.cos(yaw), math.sin(yaw)
    near = (c * (centre[0] - x) + s * (centre[1] - y), -s * (centre[0] - x) + c * (centre[1] - y))
    m = cam.measure(sim.data, near_rel=near, gate=TARGET_GATE_M)
    est = (x + c * m["centre"][0] - s * m["centre"][1], y + s * m["centre"][0] + c * m["centre"][1])
    assert math.hypot(est[0] - centre[0], est[1] - centre[1]) < 0.02
    assert abs(math.degrees(m["into"] + yaw - (yaw + 0.6))) < 3.0


# --- the cutover, end to end -------------------------------------------------------------------------

@pytest.mark.skipif(not (_render_ok() and HAVE_POLICIES), reason="needs offscreen GL and the policies")
def test_a_guarded_mission_on_estimated_inputs_kicks_and_truth_is_read_only_for_scoring():
    import onnxruntime as ort
    import kick_mission as km
    from camera_ball import CameraBall
    from camera_goal import CameraGoal, assign_target
    from goto_point import RunLog
    km.apply_calibration("v2")
    sim = km.KickSim(VELSTAND, km.SCENES["field"])
    km.set_feet(sim, "right")
    km.setup_goal(sim)
    sim.arena = km.setup_field(sim)
    sim.reset(0.0, 0.0, 0.0)
    ball = (1.1, 0.0)
    goal = km.place_goal(sim, (ball[0] + 0.8, 0.0), 0.0)
    sim.data.qpos[sim.bqa:sim.bqa + 7] = [*ball, 0.035, 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    pose = pp.make_provider("odom", sim, calib="stride_v1")
    pose.reset(sim, seed=0)
    bc = CameraBall(sim, pose)
    gc = CameraGoal(sim, pose)
    gc.reset(target=assign_target((goal.cx, goal.cy)))
    guard = TruthGuard(sim)
    m = km.Mission(guard, ort.InferenceSession(KICK_R), None, RunLog(echo=False), foot="right",
                   precision_session=ort.InferenceSession(PRECISION), goal=goal, pose_src=pose, ball_src=bc,
                   goal_src=gc, estimated_only=True)
    while m.state != "DONE" and sim.t < 45.0:
        sim.set_twist(m.tick())
        sim.step()
        pose.update(sim)
    r = m.result
    assert m.state == "DONE" and not r.get("aborted"), r
    assert (r.get("peak_ball_fwd") or 0) > 0.05 and abs(r["aim_err_deg"]) < 3.0
    assert guard.scoring_reads > 0                 # truth WAS read: for scoring, inside Mission._truth()
    # and truth sources (the defaults) on the guarded sim fail loudly, before the duck takes a step
    sim.reset(0.0, 0.0, 0.0)
    with pytest.raises(TruthAccess):
        km.Mission(guard, ort.InferenceSession(KICK_R), 0.0, RunLog(echo=False), foot="right").tick()
