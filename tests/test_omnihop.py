"""OmniHop (2026-11-24): cfg contract, target sampler, drift algebra, hop state machine (stub env)."""
import importlib
import math
from types import SimpleNamespace as NS

import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks import microduck_omnihop_env_cfg as oh


# ── cfg contract ──────────────────────────────────────────────────────────────

def test_actor_is_the_61d_layout_with_pulse_in_body_slot():
    cfg = oh.make_microduck_omnihop_env_cfg()
    actor = cfg.observations["actor"].terms
    assert list(actor) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
                           "command", "head_command", "body_command"]
    assert actor["body_command"].func is m.hop_body_command
    assert actor["head_command"].params == {"dim": 4}
    assert isinstance(cfg.commands["twist"], m.HopTargetCommandCfg)
    assert actor["base_ang_vel"].func is m.base_ang_vel_imu_biased


def test_robot_sensors_terminations_and_event_order():
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    cfg = oh.make_microduck_omnihop_env_cfg()
    assert cfg.scene.entities == {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    assert {s.name for s in cfg.scene.sensors} >= {"feet_ground_contact", "nonfoot_ground_contact"}
    assert "nonfoot_contact" in cfg.terminations
    ev = list(cfg.events)
    assert ev.index("reset_hop_target") > ev.index("set_ground_state")
    assert cfg.events["reset_hop_target"].params["in_place_prob"] == 1.0      # in place first
    assert oh.make_microduck_omnihop_env_cfg(play=True).events["reset_hop_target"].params["in_place_prob"] == 0.25


def test_nonfoot_pattern_excludes_only_the_ankles():
    import re
    p = re.compile(oh.NONFOOT_BODIES_PATTERN)
    assert not p.match("ankle_left") and not p.match("ankle_right")
    assert all(p.match(b) for b in ("trunk_base", "leg", "leg_2", "jaw_soft", "neck_pitch", "upper_leg_left"))


def test_self_negating_terms_have_positive_weights_and_symmetry_on():
    cfg = oh.make_microduck_omnihop_env_cfg()
    for name in ("hop_extra", "nonfoot_contact"):
        assert cfg.rewards[name].weight > 0
    assert oh.MicroduckOmniHopRlCfg.algorithm.symmetry_cfg is not None
    assert oh.MicroduckOmniHopRlCfg.experiment_name == "omnihop"


def test_drift_switch(monkeypatch):
    monkeypatch.setenv("MICRODUCK_HOP_DRIFT", "0")
    try:
        mod = importlib.reload(oh)
        assert mod.make_microduck_omnihop_env_cfg().events["reset_hop_target"].params["drift"] is False
    finally:
        monkeypatch.delenv("MICRODUCK_HOP_DRIFT")
        importlib.reload(oh)
    assert oh.make_microduck_omnihop_env_cfg().events["reset_hop_target"].params["drift"] is True


# ── stub env ──────────────────────────────────────────────────────────────────

class _Sensors(dict):
    pass


def _stub_env(n, yaw0=0.0):
    qpos = torch.zeros(n, 7)
    qpos[:, 2] = 0.115
    qpos[:, 3] = math.cos(yaw0 / 2)
    qpos[:, 6] = math.sin(yaw0 / 2)
    data = NS(root_link_pos_w=qpos[:, 0:3].clone(), root_link_quat_w=qpos[:, 3:7].clone(),
              projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
              root_link_lin_vel_b=torch.zeros(n, 3), root_link_ang_vel_b=torch.zeros(n, 3),
              site_pos_w=torch.zeros(n, 2, 3))
    robot = NS(indexing=NS(free_joint_q_adr=torch.arange(7)), data=data,
               find_sites=lambda names, preserve_order=False: ([0, 1], list(names)))
    sensors = {"feet_ground_contact": NS(data=NS(found=torch.ones(n, 2))),
               "nonfoot_ground_contact": NS(data=NS(found=torch.zeros(n, 5, 1)))}
    env = NS(device="cpu", num_envs=n, sim=NS(data=NS(qpos=qpos)), step_dt=0.02,
             common_step_counter=0, episode_length_buf=torch.zeros(n, dtype=torch.long))
    env.scene = type("S", (), {"__getitem__": lambda self, k: {"robot": robot}[k], "sensors": sensors})()
    return env, robot


def _step(env, robot, feet_down: bool, foot_z: float, tilt_deg: float = 0.0):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    env.scene.sensors["feet_ground_contact"].data.found[:] = 1.0 if feet_down else 0.0
    robot.data.site_pos_w[:, :, 2] = foot_z
    t = math.radians(tilt_deg)
    robot.data.projected_gravity_b[:] = torch.tensor([math.sin(t), 0.0, -math.cos(t)])
    return m.hop_update(env)


# ── sampler ───────────────────────────────────────────────────────────────────

def test_in_place_stage_targets_are_zero_and_no_hop_bucket():
    torch.manual_seed(0)
    env, _ = _stub_env(5000)
    m.reset_hop_target(env, torch.arange(5000), no_hop_prob=0.15, in_place_prob=1.0)
    st = env._hop
    assert torch.all(st["target_b"] == 0)
    assert 0.13 < (~st["req"]).float().mean().item() < 0.17
    assert ((st["t0"] >= 0.0) & (st["t0"] <= 0.5)).all()


def test_widened_targets_are_single_axis_and_in_range():
    torch.manual_seed(1)
    env, _ = _stub_env(8000)
    m.reset_hop_target(env, torch.arange(8000), no_hop_prob=0.0, in_place_prob=0.25, max_fwd=0.04, max_side=0.03)
    tb = env._hop["target_b"]
    moving = tb.abs().sum(dim=1) > 0
    assert 0.72 < moving.float().mean().item() < 0.78
    assert ((tb[moving] != 0).sum(dim=1) == 1).all()
    x, y = tb[:, 0], tb[:, 1]
    assert x.abs().max() <= 0.04 + 1e-6 and y.abs().max() <= 0.03 + 1e-6
    assert (x[x != 0].abs() >= 0.01 - 1e-6).all() and (y[y != 0].abs() >= 0.0075 - 1e-6).all()
    for v in (x > 0, x < 0, y > 0, y < 0):
        assert v.sum() > 1000


def test_drift_draws_and_off_switch():
    torch.manual_seed(2)
    env, _ = _stub_env(4000)
    m.reset_hop_target(env, torch.arange(4000), drift=True)
    st = env._hop
    assert abs(st["scale"].mean().item() + 0.019) < 0.002
    assert st["jump"].norm(dim=1).max() <= 0.02 + 1e-6 and st["gyro_bias"].abs().max() <= 0.03
    m.reset_hop_target(env, torch.arange(4000), drift=False)
    assert torch.all(st["scale"] == 0) and torch.all(st["jump"] == 0) and torch.all(st["gyro_bias"] == 0)


# ── drift algebra ─────────────────────────────────────────────────────────────

def test_offset_is_target_at_start_and_zero_on_arrival_any_heading():
    env, robot = _stub_env(1, yaw0=0.7)
    m.reset_hop_target(env, torch.arange(1), no_hop_prob=0.0, in_place_prob=0.0, drift=False)
    st = env._hop
    st["target_b"][:] = torch.tensor([[0.04, 0.0]])
    assert torch.allclose(m.hop_est_offset(env), torch.tensor([[0.04, 0.0]]), atol=1e-6)
    gx, gy = m._yaw_frame_to_world(torch.tensor([0.04]), torch.tensor([0.0]), torch.tensor([0.7]))
    robot.data.root_link_pos_w[:, 0] += gx
    robot.data.root_link_pos_w[:, 1] += gy
    assert torch.allclose(m.hop_est_offset(env), torch.zeros(1, 2), atol=1e-6)
    st["scale"][:] = -0.02            # odometry under-reads: 2 % of the hop still "remains"
    assert torch.allclose(m.hop_est_offset(env), torch.tensor([[0.0008, 0.0]]), atol=1e-6)


# ── state machine ─────────────────────────────────────────────────────────────

def _hop_env(req=True, t0=0.0):
    env, robot = _stub_env(1)
    m.reset_hop_target(env, torch.arange(1), no_hop_prob=0.0 if req else 1.0, pulse_delay=(t0, t0), drift=False)
    for _ in range(3):
        _step(env, robot, True, 0.015)
    return env, robot


def test_clean_hop_counts_once_and_lands():
    env, robot = _hop_env()
    assert m.hop_pulse(env).item()
    for z in (0.020, 0.025, 0.027, 0.025, 0.020):
        st = _step(env, robot, False, z)
        assert m.hop_flight_reward(env).item() > 0.2
    st = _step(env, robot, True, 0.015)
    assert st["hops"].item() == 1 and st["landed"].item() and st["ev_land"].item() == 1.0
    assert abs(st["best_clear"].item() - 0.012) < 1e-4 and st["flight_steps"].item() == 5
    assert m.hop_landing_reward(env).item() > 0.99          # in place, landed on target
    assert m.hop_extra_penalty(env).item() == 0.0
    st = _step(env, robot, True, 0.015)
    assert st["ev_land"].item() == 0.0 and m.hop_stand_reward(env).item() > 0.9


def test_second_hop_is_penalised_and_short_flights_do_not_count():
    env, robot = _hop_env()
    _step(env, robot, False, 0.03)
    _step(env, robot, False, 0.03)                           # 2 steps: too short
    st = _step(env, robot, True, 0.015)
    assert st["hops"].item() == 0
    for _ in range(4):
        _step(env, robot, False, 0.03)
    _step(env, robot, True, 0.015)
    for _ in range(4):
        _step(env, robot, False, 0.03)
    st = _step(env, robot, True, 0.015)
    assert st["hops"].item() == 2 and m.hop_extra_penalty(env).item() == -1.0


def test_unrequested_hop_is_penalised_and_stand_reward_kept_when_no_request():
    env, robot = _hop_env(req=False)
    assert not m.hop_pulse(env).item() and m.hop_stand_reward(env).item() > 0.9
    for _ in range(4):
        _step(env, robot, False, 0.03)
    st = _step(env, robot, True, 0.015)
    assert st["ev_unreq"].item() == 1.0 and not st["landed"].item()
    assert m.hop_extra_penalty(env).item() == -1.0


def test_requested_but_no_hop_forfeits_standing_after_the_pulse():
    env, robot = _hop_env(t0=0.1)
    for _ in range(4):
        _step(env, robot, True, 0.015)                       # t = 0.14 s: pulse on
    assert m.hop_pulse(env).item() and m.hop_stand_reward(env).item() == 0.0


def test_tilted_liftoff_is_not_a_hop():
    env, robot = _hop_env()
    for _ in range(5):
        _step(env, robot, False, 0.03, tilt_deg=40)
        assert m.hop_flight_reward(env).item() == 0.0
    st = _step(env, robot, True, 0.015, tilt_deg=40)
    assert st["hops"].item() == 0


def test_body_command_carries_the_pulse_in_slot_0_only():
    env, robot = _hop_env()
    b = m.hop_body_command(env)
    assert b.shape == (1, 6) and b[0, 0].item() == 1.0 and torch.all(b[0, 1:] == 0)
    for _ in range(40):                                      # past 0.6 s
        _step(env, robot, True, 0.015)
    assert m.hop_body_command(env)[0, 0].item() == 0.0


# ── conditioning guard (IsaacLab PR #9173) ────────────────────────────────────

def test_direction_command_is_in_every_actor_frame_no_goal_encoder():
    """The hop target rides in the twist slots of the flat 61-d actor obs each step, into a
    plain MLP: no separate goal/latent encoder that could saturate to one value for all commands."""
    cfg = oh.make_microduck_omnihop_env_cfg()
    actor = cfg.observations["actor"]
    assert actor.terms["command"].params["command_name"] == "twist"
    assert getattr(actor, "history_length", 0) in (0, 1, None)
    rl = oh.MicroduckOmniHopRlCfg
    assert rl.actor.hidden_dims == (512, 256, 128)
    assert not any("encoder" in k.lower() for k in vars(rl.actor))
    assert set(rl.obs_groups["actor"]) == {"actor"}


def _rows(disp_by_dir, n=5):
    import sys
    sys.path.insert(0, "scripts")
    import hop_battery as hb
    tg = {"in_place": (0, 0), "fwd": (30, 0), "back": (-30, 0), "left": (0, 20), "right": (0, -20)}
    rows = [{"drift": True, "dir": d, "target_mm": list(tg[d]), "first": {"disp_mm": list(disp_by_dir[d])}}
            for d in tg for _ in range(n)]
    return hb, rows


def test_direction_check_flags_one_hop_for_every_command():
    hb, rows = _rows({d: (2.0, 1.0) for d in ("in_place", "fwd", "back", "left", "right")})
    out = "\n".join(hb.direction_check(rows))
    assert "SAME HOP REGARDLESS OF DIRECTION" in out and "DIRECTION NOT FOLLOWED: fwd/back" in out


def test_direction_check_passes_a_policy_that_follows_commands():
    hb, rows = _rows({"in_place": (1, 0), "fwd": (25, 2), "back": (-22, -1), "left": (1, 16), "right": (0, -15)})
    out = "\n".join(hb.direction_check(rows))
    assert "flags: none" in out and "FLAG" not in out


# ── v2 shaping (after chunk 1a locked into standing) ──────────────────────────

def test_v2_standing_terms_are_gated_and_shaping_exists():
    cfg = oh.make_microduck_omnihop_env_cfg()
    r = cfg.rewards
    assert "upright" not in r
    assert r["pose_stand_legs"].func is m.hop_gated_pose and r["upright_stand"].func is m.hop_gated_upright
    assert r["height_stand"].func is m.hop_gated_height
    assert r["hop_reference"].func is m.hop_reference_prior and r["hop_reference"].weight > 0
    assert r["hop_push"].weight > 0
    for p in (oh.HOP_REF_CROUCH, oh.HOP_REF_EXTEND, oh.HOP_REF_TUCK):
        assert len(p) == 14 and p[9 + 2] == -p[2] and p[9 + 3] == -p[3]      # mirrored legs


def test_standing_gate_off_from_pulse_until_landing():
    env, robot = _hop_env(t0=0.1)
    assert m._hop_standing_gate(env).item() == 1.0                 # before the pulse
    for _ in range(4):
        _step(env, robot, True, 0.015)
    assert m._hop_standing_gate(env).item() == 0.0                 # requested, not hopped
    for z in (0.02, 0.025, 0.025, 0.02):
        _step(env, robot, False, z)
    _step(env, robot, True, 0.015)
    assert env._hop["landed"].item() and m._hop_standing_gate(env).item() == 1.0


def test_reference_phases_follow_the_pulse_and_stop_after_the_hop():
    env, robot = _hop_env(t0=0.0)                                   # t = 0.06 s after _hop_env
    c, e, k, T = oh.HOP_REF_CROUCH, oh.HOP_REF_EXTEND, oh.HOP_REF_TUCK, oh.HOP_REF_T
    ref, act = m.hop_reference_targets(env, c, e, k, T)
    assert act.item() and torch.allclose(ref[0], torch.tensor(c))
    for _ in range(13):                                             # t = 0.32 s: extend
        _step(env, robot, True, 0.015)
    ref, act = m.hop_reference_targets(env, c, e, k, T)
    assert act.item() and torch.allclose(ref[0], torch.tensor(e))
    for _ in range(4):                                              # t = 0.40 s: tuck
        _step(env, robot, True, 0.015)
    ref, _ = m.hop_reference_targets(env, c, e, k, T)
    assert torch.allclose(ref[0], torch.tensor(k))
    for z in (0.02, 0.025, 0.025, 0.02):
        _step(env, robot, False, z)
    _step(env, robot, True, 0.015)
    _, act = m.hop_reference_targets(env, c, e, k, T)
    assert not act.item()                                           # hopped: prior off


def test_push_reward_only_in_window_before_hop():
    env, robot = _hop_env(req=False)
    robot.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.4]])
    assert m.hop_push_reward(env).item() == 0.0
    env, robot = _hop_env()
    robot.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.25]])
    assert abs(m.hop_push_reward(env).item() - 0.5) < 1e-6


def test_v3_fall_costs_and_alive_upright():
    cfg = oh.make_microduck_omnihop_env_cfg()
    assert cfg.rewards["fell"].func is m.hop_fell_penalty and cfg.rewards["fell"].weight == 30.0
    assert cfg.rewards["fell"].params["limit_angle"] == cfg.terminations["fell_over"].params["limit_angle"]
    assert cfg.rewards["upright_alive"].weight == 0.5
    assert cfg.rewards["upright_alive"].weight < cfg.rewards["hop_stand"].weight      # no stand-still trap
    env, robot = _hop_env()
    _step(env, robot, True, 0.015, tilt_deg=59)
    assert m.hop_fell_penalty(env, math.radians(60)).item() == 0.0
    _step(env, robot, True, 0.015, tilt_deg=61)
    assert m.hop_fell_penalty(env, math.radians(60)).item() == -1.0
    assert m.hop_upright(env, math.sqrt(0.05)).item() < 0.01


# ── v4 ────────────────────────────────────────────────────────────────────────

def test_v4_cfg_height_swap_and_schedule():
    cfg = oh.make_microduck_omnihop_env_cfg()
    assert cfg.rewards["hop_flight"].params["clear_full"] == 0.04
    assert cfg.observations["actor"].terms["actions"].func is m.hop_last_action
    assert list(cfg.observations["actor"].terms).index("actions") == 4          # 61-d layout unchanged
    p = cfg.events["reset_hop_target"].params
    assert p["swap_prob"] == 0.5 and p["in_place_prob"] == 1.0
    assert oh.target_stages()[1]["step"] == 2000 * 24                           # v3 model_999 resume stays in place


def test_swap_action_only_on_the_first_frame():
    from types import SimpleNamespace as NS2
    env, robot = _stub_env(2000)
    torch.manual_seed(3)
    m.reset_hop_target(env, torch.arange(2000), swap_prob=0.5, swap_noise=0.15)
    st = env._hop
    env.action_manager = NS2(action=torch.full((2000, 14), 0.3))
    env.episode_length_buf[:] = 0
    a0 = m.hop_last_action(env)
    assert 0.45 < st["swap"].float().mean().item() < 0.55
    assert torch.equal(a0[st["swap"]], st["swap_act"][st["swap"]]) and a0[st["swap"]].abs().max() <= 0.15
    assert torch.all(a0[~st["swap"]] == 0.3)
    env.episode_length_buf[:] = 1
    assert torch.all(m.hop_last_action(env) == 0.3)


def test_battery_default_pulse_delay_is_the_runtime_settle():
    import sys
    sys.path.insert(0, "scripts")
    import hop_battery as hb
    assert hb.SETTLE_S == 0.25
