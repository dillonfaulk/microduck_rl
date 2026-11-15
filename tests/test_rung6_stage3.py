"""Rung 6 Stage 3: stride calibration, the camera ball channel, the re-observe mission states."""
import math
import os
import sys

import mujoco
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import pose_provider as pp  # noqa: E402
from test_rung6_stage2 import _Sim, place_trunk  # noqa: E402


def test_stride_calibration_is_named_versioned_and_scales_displacement_only():
    assert pp.ODOM_CALIBS == {"none": 1.0, "stride_v1": 1.019}
    s = _Sim()
    place_trunk(s.model, s.data, 0.7, -0.4, 1.0)
    raw, cal = pp.OdometryProvider(s), pp.OdometryProvider(s, calib="stride_v1")
    assert cal.stride_scale == 1.019 and raw.stride_scale == 1.0
    # at the anchor (zero displacement) the calibration changes nothing
    assert cal.get(0.0).pose == pytest.approx(raw.get(0.0).pose, abs=1e-9)
    with pytest.raises(SystemExit):
        pp.OdometryProvider(s, calib="stride_v9")


def test_map_offset_composes_like_autod():
    xy, yaw = pp.apply_offset((0.0, 0.0, 0.0), np.array([0.3, -0.2]), 0.4)
    assert xy == pytest.approx([0.3, -0.2]) and yaw == pytest.approx(0.4)
    xy, yaw = pp.apply_offset((1.0, 2.0, math.pi / 2), np.array([1.0, 0.0]), 0.0)
    assert xy == pytest.approx([1.0, 3.0]) and yaw == pytest.approx(math.pi / 2)


def test_orange_mask_takes_the_ball_and_leaves_the_field():
    from camera_ball import orange_mask
    px = np.array([[[255, 140, 0], [200, 105, 10],     # ball, lit and shaded
                    [205, 180, 140], [128, 128, 128],  # light wood boards, grey (hidden) ball
                    [40, 60, 40], [250, 250, 250]]], dtype=np.uint8)
    assert orange_mask(px).tolist() == [[True, True, False, False, False, False]]


def _render_ok():
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _render_ok(), reason="no offscreen GL (set MUJOCO_GL=egl)")
def test_ray_to_plane_ranges_a_ball_within_5mm_from_the_profile_intrinsics():
    from camera_ball import BallCamera
    s = _Sim()
    place_trunk(s.model, s.data, 0.0, 0.0, 0.5)
    cam = BallCamera(s.model)
    assert cam.K["fy"] == pytest.approx(180 / math.tan(math.radians(s.model.cam_fovy[cam.cam]) / 2))
    b = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qa = s.model.jnt_qposadr[b]
    trunk = s.data.xpos[cam.trunk]
    for d in (0.6, 0.9):
        s.data.qpos[qa:qa + 7] = [trunk[0] + d * math.cos(0.5), trunk[1] + d * math.sin(0.5), 0.035, 1, 0, 0, 0]
        mujoco.mj_forward(s.model, s.data)
        m = cam.measure(s.data)
        assert m is not None and not m["edge"]
        assert math.hypot(m["ray"][0] - d, m["ray"][1]) < 0.005
    s.model.geom_rgba[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")] = (0.5, 0.5, 0.5, 1)
    assert cam.measure(s.data) is None          # a grey ball is not found: the not-found case is real


def test_mission_camera_constants_are_sane():
    import kick_mission as km
    assert km.RESCAN_MAX >= 2 and km.RESCAN_HEAD_YAWS[0] == 0.0
    assert abs(km.LOOK_HEAD[1]) <= 1.10 and abs(km.LOOK_HEAD[0]) <= 1.10   # inside VelStand's trained head range
    assert km.CAM_REBASE_TOL < km.PRECISE_MAX_R
