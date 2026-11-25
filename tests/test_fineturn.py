"""FineTurn (2026-11-24): slow pure-channel command bucket, anchor dead-band mask, MICRODUCK_FINETURN switch."""
import importlib
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.distill import anchor_dead_band_mask

TW = (0, 3)
BANDS = {"wz": (-0.12, 0.12), "vx": (-0.13, 0.145), "vy": (-0.06, 0.06)}


def test_dead_band_mask_pure_channels_only():
    obs = torch.tensor([
        [0.0, 0.0, 0.10],    # in-place slow yaw -> masked
        [0.0, 0.0, 0.20],    # anchor live -> anchored
        [0.10, 0.0, 0.0],    # slow pure forward -> masked
        [0.15, 0.0, 0.0],    # anchor live
        [-0.10, 0.0, 0.0],   # slow back -> masked
        [0.0, 0.05, 0.0],    # slow sideways -> masked
        [0.0, 0.08, 0.0],    # live
        [0.10, 0.0, 0.10],   # mixed -> not pure -> anchored
        [0.0, 0.0, 0.0],     # standing -> anchored (never masked)
    ])
    got = anchor_dead_band_mask(obs, TW, BANDS).tolist()
    assert got == [True, False, True, False, True, True, False, False, False]


def test_fine_sampler_pure_and_in_range():
    cfg = SimpleNamespace(fine_wz_range=(0.05, 0.4), fine_vx_range=(0.03, 0.2), fine_vy_range=(0.03, 0.12))
    n = 20000
    fake = SimpleNamespace(cfg=cfg, device="cpu", vel_command_b=torch.full((n, 3), 9.0),
                           vel_command_w=torch.zeros(n, 3), is_standing_env=torch.ones(n, dtype=torch.bool))
    torch.manual_seed(0)
    mdp.VelocityCommandCommandOnly._resample_fine(fake, torch.arange(n), 0.15)
    hit = fake.vel_command_b[:, 0] != 9.0
    frac = hit.float().mean().item()
    assert 0.13 < frac < 0.17
    c = fake.vel_command_b[hit]
    assert ((c != 0).sum(dim=1) == 1).all()                       # exactly one channel
    for col, (lo, hi) in ((2, cfg.fine_wz_range), (0, cfg.fine_vx_range), (1, cfg.fine_vy_range)):
        v = c[c[:, col] != 0, col]
        assert len(v) > 500 and (v.abs() >= lo).all() and (v.abs() <= hi).all()
        assert (v > 0).any() and (v < 0).any()
    assert not fake.is_standing_env[hit].any() and fake.is_standing_env[~hit].all()
    assert torch.equal(fake.vel_command_w[hit], fake.vel_command_b[hit])


def test_fineturn_switch(monkeypatch):
    import mjlab_microduck.tasks.microduck_velstand_env_cfg as v
    monkeypatch.delenv("MICRODUCK_FINETURN", raising=False)
    v = importlib.reload(v)
    assert not v.FINETURN_ENABLED and v.MicroduckVelStandRlCfg.algorithm.bc_cfg.get("anchor_dead_bands") is None
    assert v.make_microduck_velstand_env_cfg().commands["twist"].rel_fine_envs == 0.0
    monkeypatch.setenv("MICRODUCK_FINETURN", "1")
    try:
        v = importlib.reload(v)
        assert v.MicroduckVelStandRlCfg.algorithm.bc_cfg["anchor_dead_bands"] == BANDS
        tw = v.make_microduck_velstand_env_cfg().commands["twist"]
        assert tw.rel_fine_envs == 0.15 and tw.turn_in_place_min_frac == 0.05
    finally:
        monkeypatch.delenv("MICRODUCK_FINETURN")
        importlib.reload(v)
