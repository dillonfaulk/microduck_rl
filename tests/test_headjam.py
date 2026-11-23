"""HeadJamRecover (2026-11-22): head-under spawn sampler, MICRODUCK_HEADJAM switch, runtime neck clip."""
import importlib
import importlib.util
import math
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
