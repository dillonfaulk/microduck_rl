"""Rung 6 Stage 2: ToF flaw switches, board/camera profiles, field boards, IMU yaw error, pose provider."""
import math
import os
import sys

import mujoco
import numpy as np
import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

from mjlab_microduck.sim import board_profiles as bp  # noqa: E402
from mjlab_microduck.sim.camera import apply_profile, pinhole_intrinsics  # noqa: E402
from mjlab_microduck.sim.tof import (MAX_RANGE, STATUS_INVALID, STATUS_NO_TARGET, STATUS_VALID, Tof,  # noqa: E402
                                     TofFlaws, valid_probability)
import pose_provider as pp  # noqa: E402

ROBOT = os.path.join(REPO, "src/mjlab_microduck/robot/microduck")
FIELD_SCENE = os.path.join(ROBOT, "scene_field.xml")
GOAL_SCENE = os.path.join(ROBOT, "scene_allcollisions_ball_goal.xml")


def load(scene=FIELD_SCENE):
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def place_trunk(m, d, x, y, yaw):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    a = m.jnt_qposadr[j]
    d.qpos[a:a + 2] = [x, y]
    d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(m, d)


def tof_of(m, flaws, seed=0):
    return Tof(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tof"), seed=seed, flaws=flaws)


# --- ToF ----------------------------------------------------------------------------------------

def test_ideal_sensor_matches_the_old_mj_ray_loop():
    """TofFlaws.none() + mj_multiRay == the pre-Stage-2 per-zone mj_ray loop, noise draws included."""
    m, d = load()
    place_trunk(m, d, 2.4, 0.3, 0.0)                     # facing the +x board ~0.5 m away
    new = tof_of(m, TofFlaws.none(), seed=3).frame(d)
    t = tof_of(m, TofFlaws.none(), seed=3)
    origin = d.site_xpos[t.site].copy()
    world = d.site_xmat[t.site].reshape(3, 3) @ t.directions.T
    rng = np.random.default_rng(3)
    dist, stat = [0] * 64, [STATUS_NO_TARGET] * 64
    g = np.zeros(1, dtype=np.int32)
    for z in range(64):
        hit = mujoco.mj_ray(m, d, origin, np.ascontiguousarray(world[:, z]), None, 1, -1, g)
        if hit < 0 or hit > MAX_RANGE:
            continue
        sigma = 0.003 + 0.02 * (hit / MAX_RANGE)
        dist[z] = int(max(0.0, hit + rng.normal(0.0, sigma)) * 1000.0)
        stat[z] = STATUS_VALID
    assert new == (dist, stat)


def test_axial_range_reads_the_corners_short():
    m, d = load()
    place_trunk(m, d, 2.4, 0.0, 0.0)
    noflaw = TofFlaws(axial_range=False, latency_s=0.0, pitch_offset_deg=0.0, dropout=False)
    slant, _ = tof_of(m, noflaw).frame(d)
    axial, _ = tof_of(m, TofFlaws(axial_range=True, latency_s=0.0, pitch_offset_deg=0.0, dropout=False)).frame(d)
    corner, centre = 3 * 8 + 0, 3 * 8 + 3           # row 3: level-ish rows that hit the board
    assert slant[corner] > 0 and axial[corner] > 0
    shrink_corner = axial[corner] / slant[corner]
    shrink_centre = axial[centre] / slant[centre]
    assert 0.90 < shrink_corner < 0.95                   # cos(off-axis) at the edge column
    assert shrink_centre > 0.98


def test_pitch_offset_tips_the_beams_down():
    m, _ = load()
    level = tof_of(m, TofFlaws.none())._beams
    tipped = tof_of(m, TofFlaws(pitch_offset_deg=3.0))._beams
    assert np.allclose(np.degrees(np.arcsin(level[:, 2]) - np.arcsin(tipped[:, 2])).mean(), 3.0, atol=0.2)


def test_dropout_follows_the_measured_range_curve_and_uses_real_status_codes():
    assert valid_probability(0.3) > 0.9 and valid_probability(0.9) < 0.5 and valid_probability(3.5) < 0.4
    m, d = load()
    place_trunk(m, d, 2.4, 0.0, 0.0)
    t = tof_of(m, TofFlaws(latency_s=0.0), seed=1)
    statuses = np.array([t.frame(d)[1] for _ in range(200)])
    assert STATUS_NO_TARGET not in statuses          # the real recordings contain no 255
    assert set(np.unique(statuses)) <= {STATUS_VALID, STATUS_INVALID}
    legacy = tof_of(m, TofFlaws.none()).frame(d)[1]
    assert STATUS_NO_TARGET in legacy                 # upper rows over the board: legacy "no target"


def test_latency_hands_out_the_capture_60ms_old():
    m, d = load()
    t = tof_of(m, TofFlaws())
    assert t.frame_at(d, 0.0) is None
    assert t.frame_at(d, 1 / 15) is not None          # the first capture is now 67 ms old
    got = t.frame_at(d, 2 / 15)
    assert got is not None and abs(got[2] - 1 / 15) < 1e-9
    zero = tof_of(m, TofFlaws.none())
    assert zero.frame_at(d, 5.0)[2] == 5.0


# --- board / camera profiles ----------------------------------------------------------------------

def test_board_profiles_carry_the_family_solves():
    p = bp.get()
    assert p.name == "face-board-v1" and p.camera.sensor == "GC2093" and p.tof.chip == "VL53L8CX"
    assert p.camera.rotate_deg == 0
    assert p.camera.hfov_deg == pytest.approx(85.2, abs=0.1) and p.camera.vfov_deg == pytest.approx(54.7, abs=0.1)
    z = bp.get("zero3")
    assert z.camera.sensor == "IMX219" and z.camera.rotate_deg == 90 and z.tof.chip == "VL53L5CX"
    assert z.camera.hfov_deg == pytest.approx(62.2, abs=0.1) and z.camera.vfov_deg == pytest.approx(37.4, abs=0.1)
    assert "L5CX" in p.tof.flaws_measured_on          # the L8CX carries the L5CX values, said so


@pytest.mark.parametrize("board", ["face-board-v1", "zero3"])
def test_camera_profile_mount_and_field(board):
    m, d = load()
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    apply_profile(m, cam, bp.get(board))
    mujoco.mj_forward(m, d)
    R = d.cam_xmat[cam].reshape(3, 3)                 # columns: camera x (right), y (up), z (back)
    tof = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tof")
    fwd = d.site_xmat[tof].reshape(3, 3)[:, 0]
    assert -R[:, 2] @ fwd > 0.999                     # looks where the ToF looks
    if board == "zero3":
        assert R[:, 0] @ np.array([0, 0, -1.0]) > 0.99     # image right = world down: rotate 90
    else:
        assert R[:, 1] @ np.array([0, 0, 1.0]) > 0.99      # upright
    assert m.cam_fovy[cam] == pytest.approx(bp.get(board).camera.vfov_deg)
    k = pinhole_intrinsics(m, cam, 1280, 720)
    assert 2 * math.degrees(math.atan(640 / k["fx"])) == pytest.approx(bp.get(board).camera.hfov_deg, abs=0.2)


# --- field ----------------------------------------------------------------------------------------

def test_field_boards_are_the_geofence():
    import kick_mission as km
    m, _ = load()
    f = km.field_from_model(m)
    assert f == pytest.approx((-3.0, 3.0, -3.0, 3.0))
    inset = (f[0] + km.FIELD_INSET_M, f[1] - km.FIELD_INSET_M, f[2] + km.FIELD_INSET_M, f[3] - km.FIELD_INSET_M)
    assert inset == pytest.approx(km.ARENA)           # the geofence go-to-point always used


def test_tof_sees_a_board_and_the_goal_scene_is_untouched():
    m, d = load()
    place_trunk(m, d, 2.5, 1.0, 0.0)                  # 0.5 m from the +x board, beside the goal mouth
    dist, stat = tof_of(m, TofFlaws(latency_s=0.0), seed=0).frame(d)
    level = [dist[r * 8 + c] for r in (2, 3, 4) for c in range(8) if stat[r * 8 + c] == STATUS_VALID]
    assert level and 0.35 < np.median(level) / 1000 < 0.6
    g = mujoco.MjModel.from_xml_path(GOAL_SCENE)
    assert not any((g.geom(i).name or "").startswith("field_") for i in range(g.ngeom))


# --- IMU yaw error + provider ---------------------------------------------------------------------

def test_imu_yaw_error_is_seeded_and_reproduces_the_measured_median():
    a, b = pp.ImuYawError(seed=4), pp.ImuYawError(seed=4)
    assert [a.step(0.02) for _ in range(100)] == [b.step(0.02) for _ in range(100)]
    off = pp.ImuYawError(seed=4, enabled=False)
    assert all(off.step(0.02) == 0.0 for _ in range(50))
    errs = []
    for s in range(400):
        e = pp.ImuYawError(seed=s)
        for _ in range(1000):                         # 20 s
            v = e.step(0.02)
        errs.append(abs(math.degrees(v)))
    assert 0.9 < np.median(errs) < 1.5                # measured median 1.1 deg per 20 s (3.3 deg/min)


class _Sim:
    """Just enough of DuckSim for the providers: model, data, t, pose(), va."""

    def __init__(self):
        self.model, self.data = load()
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qa, self.va = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        self.t = 0.0

    def pose(self):
        q = self.data.qpos
        w, x, y, z = q[self.qa + 3:self.qa + 7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        return (float(q[self.qa]), float(q[self.qa + 1]), yaw), tilt, float(q[self.qa + 2])


def test_ground_truth_provider_is_the_old_read():
    s = _Sim()
    e = pp.GroundTruthProvider(s).get(0.0)
    assert (e.pose, e.tilt_deg, e.trunk_z) == s.pose()
    assert np.shares_memory(e.v_xy, s.data.qvel)


def test_odometry_provider_anchors_at_the_start_and_goes_absent_when_stale():
    s = _Sim()
    place_trunk(s.model, s.data, 0.7, -0.4, 1.0)
    src = pp.OdometryProvider(s, dropouts=[(0.1, 0.3)])
    e = src.get(0.0)
    err, eyaw = pp.pose_error(e.pose, s.pose()[0])
    assert err < 0.005 and abs(eyaw) < 1e-6 and e.lost is None
    for k in range(1, 20):                            # standing still, link down 0.1-0.3 s
        s.t = 0.02 * k
        src.update(s)
        e = src.get(s.t)
        if 0.1 <= s.t < 0.3:
            age = s.t - 0.08
            assert (e.pose is None) == (age > pp.STALE_S + 1e-9), (s.t, e)
            if e.pose is None:
                assert e.lost == "stale" and "pose" not in e.map_state()
        else:
            assert e.pose is not None and e.age_s == 0.0
    assert e.map_state()["pose"]["x"] == pytest.approx(0.7, abs=0.01)
