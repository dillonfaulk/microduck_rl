"""HeadJamRecover (2026-11-22): head-under spawn sampler, MICRODUCK_HEADJAM switch, runtime neck clip."""
import importlib
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
LO = np.array([-0.436, -0.384, -1.571, -1.571, -1.571, -1.571, -1.571, -2.967, -0.436,
               -0.524, -0.384, -1.571, -1.571, -1.571])
HI = np.array([0.524, 0.384, 1.571, 1.571, 1.571, 1.047, 1.571, 2.967, 0.436,
               0.436, 0.384, 1.571, 1.571, 1.571])


def _rpy(q):
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(roll), np.degrees(pitch)


def test_sampler_ranges():
    from mjlab_microduck.tasks.mdp import sample_head_under_np
    q, z, j = sample_head_under_np(np.random.default_rng(0), 2000, LO, HI)
    assert q.shape == (2000, 4) and z.shape == (2000,) and j.shape == (2000, 14)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)
    roll, pitch = _rpy(q)
    assert pitch.min() >= 25.0 - 1e-6 and pitch.max() <= 60.0 + 1e-6
    assert np.abs(roll).max() <= 20.0 + 1e-6
    assert z.min() >= 0.13 and z.max() <= 0.14
    assert (j >= LO - 1e-12).all() and (j <= HI + 1e-12).all()
    assert j[:, 5].min() >= -1.571 and j[:, 5].max() <= -1.10
    assert np.abs(j[:, 7]).max() <= 1.1


def test_sampler_mirror_is_symmetric():
    from mjlab_microduck.tasks.mdp import sample_head_under_np
    _, _, j = sample_head_under_np(np.random.default_rng(1), 4000, LO, HI)
    # the stall's left hip_roll is +0.27; its mirror puts -0.27 on the right: both halves present
    assert (j[:, 1] > 0.15).mean() > 0.3 and (j[:, 10] < -0.15).mean() > 0.3


def test_headjam_switch_off_by_default(monkeypatch):
    monkeypatch.delenv("MICRODUCK_HEADJAM", raising=False)
    import mjlab_microduck.tasks.microduck_standup_env_cfg as s
    s = importlib.reload(s)
    cfg = s.make_microduck_standup_env_cfg()
    assert cfg.actions["joint_pos"].clip is None
    assert "set_head_under_state" not in cfg.events


def test_headjam_switch_on(monkeypatch):
    monkeypatch.setenv("MICRODUCK_HEADJAM", "1")
    import mjlab_microduck.tasks.microduck_standup_env_cfg as s
    s = importlib.reload(s)
    try:
        cfg = s.make_microduck_standup_env_cfg()
        lo, hi = cfg.actions["joint_pos"].clip["neck_pitch"]
        assert lo == pytest.approx(-math.pi / 2 + 0.05) and hi == pytest.approx(math.radians(60) - 0.05)
        assert cfg.events["set_head_under_state"].params["prob"] == 0.10
    finally:
        monkeypatch.delenv("MICRODUCK_HEADJAM")
        importlib.reload(s)


def test_runtime_clip_matches_training_clip():
    spec = importlib.util.spec_from_file_location("infer_policy", REPO / "scripts" / "infer_policy.py")
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)
    from mjlab_microduck.tasks.microduck_standup_env_cfg import HEADJAM_NECK_CLIP
    assert ip.NECK_CLIP_RANGE == pytest.approx(HEADJAM_NECK_CLIP)
    assert ip.NECK_CLIP_INDEX == 5


def _load_ip(monkeypatch, env=None):
    if env is None:
        monkeypatch.delenv("MICRODUCK_NECK_CLIP", raising=False)
    else:
        monkeypatch.setenv("MICRODUCK_NECK_CLIP", env)
    spec = importlib.util.spec_from_file_location("infer_policy_t", REPO / "scripts" / "infer_policy.py")
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)
    return ip


def test_neck_clip_auto_for_headjam_policies(monkeypatch):
    ip = _load_ip(monkeypatch)
    assert ip.neck_clip_auto("logs/x/2026-11-22_19-47-54_headjamB_model_4250.onnx")
    assert ip.neck_clip_auto(None, "microduck/velstand/velstand_headjam_model_4250.onnx")
    assert not ip.neck_clip_auto("logs/x/velstand_round3_model_3999.onnx", None)
    assert not ip.neck_clip_auto("logs/headjam_dir/velstand_round3_model_3999.onnx")   # file name only
    # an explicit MICRODUCK_NECK_CLIP (0 or 1) always wins over the automatic switch
    assert not _load_ip(monkeypatch, "0").neck_clip_auto("velstand_headjam_model_4250.onnx")
    ip1 = _load_ip(monkeypatch, "1")
    assert ip1.NECK_CLIP_ENABLED and not ip1.neck_clip_auto("velstand_round3_model_3999.onnx")


def test_soccer_default_velstand_is_shipped_4250():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import kick_mission as km
    assert km.DEFAULT_VELSTAND.endswith("2026-11-22_19-47-54_headjamB_model_4250.onnx")
    assert os.path.exists(km.DEFAULT_VELSTAND) and os.path.exists(km.ROUND3_VELSTAND)
