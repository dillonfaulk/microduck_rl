import importlib

from mjlab_microduck.tasks import microduck_ball_kick_env_cfg as bk


def test_target_speed_default_is_upstream(monkeypatch):
    monkeypatch.delenv("MICRODUCK_BALL_TARGET_SPEED", raising=False)
    mod = importlib.reload(bk)
    assert mod.BALL_TARGET_SPEED == 1.0
    cfg = mod.make_microduck_ball_kick_env_cfg()
    assert cfg.rewards["ball_forward_velocity"].params["max_speed"] == 1.0
    assert cfg.rewards["ball_speed_overshoot"].params["target_speed"] == 1.0
    assert cfg.rewards["ball_forward_velocity"].weight == 12.0
    assert cfg.rewards["ball_speed_overshoot"].weight == -4.0


def test_target_speed_env_override_reaches_both_kick_terms(monkeypatch):
    monkeypatch.setenv("MICRODUCK_BALL_TARGET_SPEED", "0.25")
    try:
        mod = importlib.reload(bk)
        cfg = mod.make_microduck_ball_kick_env_cfg()
        assert cfg.rewards["ball_forward_velocity"].params["max_speed"] == 0.25
        assert cfg.rewards["ball_speed_overshoot"].params["target_speed"] == 0.25
    finally:
        monkeypatch.delenv("MICRODUCK_BALL_TARGET_SPEED")
        importlib.reload(bk)


def test_actor_stays_ball_blind_and_61d():
    cfg = bk.make_microduck_ball_kick_env_cfg()
    actor = cfg.observations["actor"].terms
    assert not any("ball" in name for name in actor)
    assert list(actor) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
                           "command", "head_command", "body_command"]
    assert "ball_position" in cfg.observations["critic"].terms


def test_kick_foot_default_is_right(monkeypatch):
    monkeypatch.delenv("MICRODUCK_KICK_FOOT", raising=False)
    mod = importlib.reload(bk)
    assert mod.KICK_FOOT == "right"
    cfg = mod.make_microduck_ball_kick_env_cfg()
    assert cfg.events["reset_ball"].params["offset"] == (0.09, -0.042)


def test_kick_foot_env_left_mirrors_ball_support_and_experiment(monkeypatch):
    monkeypatch.setenv("MICRODUCK_KICK_FOOT", "left")
    try:
        mod = importlib.reload(bk)
        assert mod.KICK_FOOT == "left"
        cfg = mod.make_microduck_ball_kick_env_cfg()
        assert cfg.events["reset_ball"].params["offset"] == (0.09, 0.042)   # +y = left toe
        support = [s for s in cfg.scene.sensors if s.name == "support_foot_ground_contact"][0]
        assert "right_foot_collision" in support.primary.pattern               # right leg supports
        assert mod.MicroduckBallKickRlCfg.experiment_name == "ball_kick_left"
    finally:
        monkeypatch.delenv("MICRODUCK_KICK_FOOT")
        importlib.reload(bk)


def test_v2_default_off_keeps_v1_recipe(monkeypatch):
    monkeypatch.delenv("MICRODUCK_BALLKICK_V2", raising=False)
    mod = importlib.reload(bk)
    cfg = mod.make_microduck_ball_kick_env_cfg()
    assert cfg.rewards["ball_speed_overshoot"].weight == -4.0
    assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.05, 0.05)
    assert mod.MicroduckBallKickRlCfg.experiment_name == "ball_kick_right"


def test_v2_env_sets_overshoot_start_noise_and_experiment(monkeypatch):
    monkeypatch.setenv("MICRODUCK_BALLKICK_V2", "1")
    monkeypatch.setenv("MICRODUCK_KICK_FOOT", "left")
    try:
        mod = importlib.reload(bk)
        cfg = mod.make_microduck_ball_kick_env_cfg()
        w = cfg.rewards["ball_speed_overshoot"].weight
        assert w == -8.0 and abs(w) < cfg.rewards["ball_forward_velocity"].weight   # kicking still pays
        assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.08, 0.08)
        assert cfg.events["reset_ball"].params["offset"] == (0.09, 0.042)
        assert mod.MicroduckBallKickRlCfg.experiment_name == "ball_kick_v2_left"
    finally:
        monkeypatch.delenv("MICRODUCK_BALLKICK_V2")
        monkeypatch.delenv("MICRODUCK_KICK_FOOT")
        importlib.reload(bk)


def test_v3_default_off_has_no_lateral_term(monkeypatch):
    monkeypatch.delenv("MICRODUCK_BALLKICK_V3", raising=False)
    monkeypatch.setenv("MICRODUCK_BALLKICK_V2", "1")
    try:
        mod = importlib.reload(bk)
        cfg = mod.make_microduck_ball_kick_env_cfg()
        assert "ball_lateral_velocity" not in cfg.rewards
        assert mod.MicroduckBallKickRlCfg.experiment_name == "ball_kick_v2_right"
        assert mod.MicroduckBallKickRlCfg.save_interval == 250
    finally:
        monkeypatch.delenv("MICRODUCK_BALLKICK_V2")
        importlib.reload(bk)


def test_v3_env_is_v2_plus_only_the_lateral_penalty(monkeypatch):
    monkeypatch.setenv("MICRODUCK_BALLKICK_V2", "1")
    v2 = importlib.reload(bk).make_microduck_ball_kick_env_cfg()
    monkeypatch.delenv("MICRODUCK_BALLKICK_V2")
    monkeypatch.setenv("MICRODUCK_BALLKICK_V3", "1")
    try:
        mod = importlib.reload(bk)
        cfg = mod.make_microduck_ball_kick_env_cfg()
        assert set(cfg.rewards) - set(v2.rewards) == {"ball_lateral_velocity"}
        assert set(v2.rewards) - set(cfg.rewards) == set()
        for k in v2.rewards:
            assert cfg.rewards[k].weight == v2.rewards[k].weight, k
        t = cfg.rewards["ball_lateral_velocity"]
        assert t.weight == -8.0 and t.func.__name__ == "ball_lateral_velocity_penalty"
        assert cfg.rewards["ball_speed_overshoot"].weight == -8.0
        assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.08, 0.08)
        assert mod.MicroduckBallKickRlCfg.experiment_name == "ball_kick_v3_right"
        assert mod.MicroduckBallKickRlCfg.save_interval == 125
    finally:
        monkeypatch.delenv("MICRODUCK_BALLKICK_V3")
        importlib.reload(bk)


def test_lateral_penalty_is_the_perpendicular_speed():
    import torch
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp
    env = SimpleNamespace(num_envs=3, device="cpu")
    env._ball_kick_dir_w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    vel = torch.tensor([[0.3, 0.05, 0.0], [0.1, 0.4, 0.0], [0.6, 0.8, 0.0]])
    ball = SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=vel))
    env.scene = {"ball": ball}
    out = mdp.ball_lateral_velocity_penalty(env, "ball")
    assert torch.allclose(out, torch.tensor([0.05, 0.1, 0.0]), atol=1e-6)
