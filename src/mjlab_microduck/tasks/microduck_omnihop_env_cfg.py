"""Microduck OmniHop task: one two-foot hop on request (in place / forward / back / left /
right), land on the feet, stand at HOME. Swap-in policy (BallKick / PrecisionStep mold).

Brief: A2A microduck-omnihop-brief.md (2026-11-19); run plan
microduck-fineturn-omnihop-run-plan-2026-11-23.md, Dillon approved 2026-11-24: ONE
direction-conditioned policy, in-place first (gate after the in-place phase), drift model
ON in training (battery ON vs OFF), flat hops v1 (10-20 mm thresholds = v2).

Feasibility (logs/evals/hop/feasibility_2026-11-23/PROBE.txt, open loop, BAM): a TUCK hop
(push, then snap the feet up) clears 10-38 mm for 100-150 ms; the CoM itself barely
launches (peak push speed 0.48 m/s); the head/neck swing is worth 13 -> 38 mm, so the
policy keeps the head joints.

Command contract (mdp.py OmniHop section): twist = estimated remaining offset to the hop
target [dx, dy, 0] (trunk yaw frame), body slot 0 = hop-request pulse (0.6 s), every
other head/body slot 0. 61-d obs / 14 actions / 50 Hz.
"""

import math
import os
from copy import deepcopy

ENABLE_SYMMETRY = True       # hops are bilateral; VelStand without it grew a left/right yaw split
DRIFT_ENABLED = os.environ.get("MICRODUCK_HOP_DRIFT", "1") == "1"

ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True    # persistent per-env tilt offset (drift item 3b)
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)   # gentler than PrecisionStep: a hop needs a settled start
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task constants ────────────────────────────────────────────────────────────
EPISODE_LENGTH_S = 4.0            # <= 0.5 s delay + 0.6 s pulse + flight + >= 2.5 s standing after
NO_HOP_PROB      = 0.15           # pulse never comes: stand still (a hop there is penalised)
PULSE_DELAY_S    = (0.0, 0.5)
MAX_FWD          = 0.04           # m, forward/back target (final stage)
MAX_SIDE         = 0.03           # m, left/right target (final stage)
IN_PLACE_ITERS   = 1000           # in-place only until here (gate), then directions widen
STAND_Z          = 0.115
NONFOOT_BODIES_PATTERN = r"^(?!ankle_left$|ankle_right$).*$"

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]
_IT = 24                          # env steps per PPO iteration

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def target_stages() -> list:
    """reset_hop_target params by PPO iteration: in place only, then widen."""
    return [
        {"step": 0, "params": {"in_place_prob": 1.0, "max_fwd": MAX_FWD, "max_side": MAX_SIDE}},
        {"step": IN_PLACE_ITERS * _IT, "params": {"in_place_prob": 0.4, "max_fwd": 0.5 * MAX_FWD, "max_side": 0.5 * MAX_SIDE}},
        {"step": (IN_PLACE_ITERS + 500) * _IT, "params": {"in_place_prob": 0.25, "max_fwd": MAX_FWD, "max_side": MAX_SIDE}},
    ]


def make_microduck_omnihop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(mode="body", pattern=NONFOOT_BODIES_PATTERN, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",), reduce="none", num_slots=1,
    )

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in ["track_linear_velocity", "track_angular_velocity", "air_time",
                 "foot_clearance", "foot_swing_height", "foot_slip", "pose", "soft_landing"]:
        cfg.rewards.pop(name, None)

    # ── Hop rewards (gated; see mdp.py OmniHop section) ───────────────────────
    # Flight: <= ~8 airborne steps x 5 = up to ~40 per hop. Landing: one-off 30 (half for
    # landing upright in the window, half for hitting the target). Standing: ~2/step,
    # forfeited from the pulse until a successful landing -> not hopping costs ~5 per
    # remaining second.
    cfg.rewards["hop_flight"] = RewardTermCfg(func=microduck_mdp.hop_flight_reward, weight=5.0)
    cfg.rewards["hop_landing"] = RewardTermCfg(func=microduck_mdp.hop_landing_reward, weight=30.0)
    cfg.rewards["hop_stand"] = RewardTermCfg(func=microduck_mdp.hop_stand_reward, weight=2.0)
    cfg.rewards["hop_extra"] = RewardTermCfg(func=microduck_mdp.hop_extra_penalty, weight=20.0)
    cfg.rewards["nonfoot_contact"] = RewardTermCfg(func=microduck_mdp.hop_nonfoot_penalty, weight=30.0)

    # ── Posture (loose: the hop crouches and tucks) ───────────────────────────
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=0.5,
        params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None})
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=0.5,
        params={"std": 0.5, "joint_indices": _NECK_JOINTS, "target_overrides": None})
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian, weight=0.5,
        params={"std": 0.04, "target_height": STAND_Z,
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # ── Regularisers (action rate capped at -0.4: the hop needs fast actions) ─
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})
    # Peak trunk spin above 6 rad/s (the 10.7 rad/s bar is reported by the battery).
    cfg.rewards["spin_cap"] = RewardTermCfg(
        func=microduck_mdp.trunk_ang_vel_excess, weight=-0.01, params={"w_thresh": 6.0})

    # ── Observations (unified 61-d actor layout) ──────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    g = "projected_gravity"
    cfg.observations["actor"].terms[g] = deepcopy(cfg.observations["actor"].terms[g])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])
    for term in ("base_ang_vel", g):
        cfg.observations["actor"].terms[term].delay_min_lag = 0
        cfg.observations["actor"].terms[term].delay_max_lag = 1
        cfg.observations["actor"].terms[term].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[g].noise              = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise    = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise    = Unoise(n_min=-0.25, n_max=0.25)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_biased       # misalignment + per-episode gyro bias
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        gv = cfg.observations["actor"].terms[g]
        gv.func = microduck_mdp.projected_gravity_imu_misaligned
        gv.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.hop_body_command)

    # ── Command: one fixed target per episode ─────────────────────────────────
    cfg.commands["twist"] = microduck_mdp.HopTargetCommandCfg(
        resampling_time_range=(EPISODE_LENGTH_S * 10, EPISODE_LENGTH_S * 10))

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)
    cfg.terminations["nonfoot_contact"] = TerminationTermCfg(
        func=microduck_mdp.hop_nonfoot_termination, time_out=False)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(func=microduck_mdp.expand_bam_friction_fields, mode="startup")
    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ("left_foot_collision", "right_foot_collision")
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.08, 0.08)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state, mode="reset",
        params={"face_down_prob": 0.0, "face_up_prob": 0.0, "sitting_prob": 0.0, "standing_prob": 1.0,
                "sitting_tilt_max": math.radians(5), "standing_z_min": 0.11, "standing_z_max": 0.12})
    # MUST come after set_ground_state (reads the final robot pose).
    stages = target_stages()
    cfg.events["reset_hop_target"] = EventTermCfg(
        func=microduck_mdp.reset_hop_target, mode="reset",
        params={"no_hop_prob": NO_HOP_PROB, "pulse_delay": PULSE_DELAY_S, "drift": DRIFT_ENABLED,
                **(stages[-1]["params"] if play else stages[0]["params"])})

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity, mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={"velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                    "asset_cfg": SceneEntityCfg("robot")})
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "operation": "add",
                    "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE)})
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES), "operation": "add",
                    "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE)})
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "operation": "scale",
                    "ranges": ARMATURE_RANDOMIZATION_RANGE})
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        lo, hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                    "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)})
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE})

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["hop_targets"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_stages, params={"event_name": "reset_hop_target", "stages": stages})
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "action_rate_l2", "weight_stages": [
            {"step": 0, "weight": -0.1}, {"step": 1000 * _IT, "weight": -0.2},
            {"step": 1500 * _IT, "weight": -0.3}, {"step": 2000 * _IT, "weight": -0.4}]})
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_com", "range_stages": [
                {"step": 0, "range": 0.003}, {"step": 500 * _IT, "range": 0.005},
                {"step": 1000 * _IT, "range": 0.01}]})
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_head_com", "range_stages": [
                {"step": 0, "range": 0.003}, {"step": 500 * _IT, "range": 0.005},
                {"step": 1000 * _IT, "range": 0.01}]})
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={"event_name": "push_robot", "push_stages": [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 750 * _IT, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1500 * _IT, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}}]})

    return cfg


MicroduckOmniHopRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128), activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="omnihop",
    run_name="omnihop",
    save_interval=250,
    num_steps_per_env=_IT,
    max_iterations=3000,
)
