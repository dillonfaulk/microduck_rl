"""Headless CPU rehearsal of an exported microduck ONNX using infer_policy.py's own code paths.

Run from the microduck_rl repo root:
  python logs/headless_rehearsal.py <onnx> [--kind walking|standing] [--suite walk|fallen|all]
                                          [--trials N] [--no-bam]
"""
import argparse
import sys

import mujoco
import numpy as np

sys.path.insert(0, "scripts")
import infer_policy as ip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("--kind", choices=["walking", "standing"], default="walking")
ap.add_argument("--suite", choices=["walk", "fallen", "all"], default="walk")
ap.add_argument("--trials", type=int, default=8)
ap.add_argument("--no-bam", action="store_true")
ap.add_argument("--train-solver", action="store_true",
                help="match training MuJoCo options: implicitfast, iterations 10, ls_iterations 20 "
                     "(scene.xml loads Euler / 100 / 50)")
ap.add_argument("--delay", type=int, default=0,
                help="action delay in 20 ms control steps (training BAM: 3-6 x 5 ms physics steps ~ 1)")
ap.add_argument("--scene", default=None,
                help="scene XML (VelStand trains on scene_allcollisions.xml; default is ground-contact scene.xml)")
args = ap.parse_args()
if args.scene:
    ip.MICRODUCK_XML = args.scene

FALL_TILT_DEG = 70.0          # mjlab fell_over limit_angle
UP_TILT_DEG, UP_Z = 25.0, 0.09  # velstand RECOVERED_UP_TILT_DEG / RECOVERED_UP_Z

WALK_SCENARIOS = [  # name, (vx, vy, wz), seconds
    ("stand (all-zero cmd)", (0.0, 0.0, 0.0), 10.0),
    ("forward 0.15 m/s", (0.15, 0.0, 0.0), 10.0),
    ("forward 0.30 m/s (max)", (0.30, 0.0, 0.0), 10.0),
    ("forward 0.10 m/s", (0.10, 0.0, 0.0), 10.0),
    ("backward 0.10 m/s", (-0.10, 0.0, 0.0), 10.0),
    ("backward 0.20 m/s", (-0.20, 0.0, 0.0), 10.0),
    ("turn left 0.5 rad/s", (0.0, 0.0, 0.5), 10.0),
    ("turn right 0.5 rad/s", (0.0, 0.0, -0.5), 10.0),
    ("strafe left 0.15 m/s", (0.0, 0.15, 0.0), 10.0),
    ("turn left 1.0 rad/s", (0.0, 0.0, 1.0), 10.0),
    ("turn right 1.0 rad/s", (0.0, 0.0, -1.0), 10.0),
    ("fwd 0.2 + turn 0.5", (0.20, 0.0, 0.5), 10.0),
]
FALLEN_SCENARIOS = [  # name, (roll, pitch) rad, spawn z
    ("upright (control)", (0.0, 0.0), 0.125),
    ("face-down", (0.0, np.pi / 2), 0.08),
    ("face-up", (0.0, -np.pi / 2), 0.08),
    ("left side", (np.pi / 2, 0.0), 0.08),
    ("right side", (-np.pi / 2, 0.0), 0.08),
    ("sitting", (0.0, 0.0), 0.07),
]
# microduck_standup_env_cfg.SITTING_JOINT_OVERRIDES (servo index -> rad), the sit keyframe
SITTING_JOINTS = {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0, 10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0}


def build():
    if args.no_bam:
        model = mujoco.MjModel.from_xml_path(ip.MICRODUCK_XML)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        bam_ctrl = None
    else:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(
            ip.MICRODUCK_XML, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    if args.train_solver:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.iterations, model.opt.ls_iterations = 10, 20
    kw = {"walking_onnx_path": args.onnx} if args.kind == "walking" else {"standing_onnx_path": args.onnx}
    # use_projected_gravity=True: infer_policy's CLI default (training obs is projected_gravity);
    # the class default (False) feeds the raw accelerometer instead.
    policy = ip.PolicyInference(model, data, bam_ctrl=bam_ctrl, new_cmd_obs=True, use_projected_gravity=True,
                                delay_min_lag=args.delay, delay_max_lag=args.delay, **kw)
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    return model, data, bam_ctrl, policy, model.jnt_qposadr[fj], model.jnt_dofadr[fj]


def reset(model, data, bam_ctrl, policy, qa, z=0.125, quat=(1, 0, 0, 0), joint_noise=0.0, rng=None):
    data.qpos[qa:qa + 3] = [0, 0, z]
    data.qpos[qa + 3:qa + 7] = quat
    for i, q in enumerate(policy.joint_qpos_indices):
        data.qpos[q] = policy.default_pose[i] + (rng.uniform(-joint_noise, joint_noise) if rng else 0.0)
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    assert policy.get_observations().size == 61


def step(model, data, bam_ctrl, policy):
    policy.apply_action(policy.infer())
    for _ in range(4):
        if bam_ctrl is not None:
            bam_ctrl.update()
        mujoco.mj_step(model, data)


def tilt_deg(policy):
    g = policy.get_projected_gravity()
    return float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))


def run_walk(name, cmd, seconds):
    model, data, bam_ctrl, policy, qa, va = build()
    reset(model, data, bam_ctrl, policy, qa)
    policy.set_vel_cmd(*cmd)
    vel, zs, tilts, fell_at = [], [], [], None
    start_xy = data.qpos[qa:qa + 2].copy()
    for k in range(int(seconds * 50)):
        step(model, data, bam_ctrl, policy)
        quat = data.qpos[qa + 3:qa + 7].astype(np.float32)
        vb = policy.quat_rotate_inverse(quat, data.qvel[va:va + 3].astype(np.float32))
        if k >= 100:
            vel.append((vb[0], vb[1], data.qvel[va + 5]))
        zs.append(float(data.qpos[qa + 2]))
        tilts.append(tilt_deg(policy))
        if fell_at is None and (tilts[-1] > FALL_TILT_DEG or zs[-1] < 0.06):
            fell_at = k / 50
            break
    v = np.array(vel) if vel else np.zeros((1, 3))
    disp = np.linalg.norm(data.qpos[qa:qa + 2] - start_xy)
    print(f"{name:26s} cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})  "
          f"achieved=({v[:,0].mean():+.2f},{v[:,1].mean():+.2f},{v[:,2].mean():+.2f})  "
          f"trunk_z={np.mean(zs)*1000:5.1f}mm  max_tilt={max(tilts):4.1f}deg  "
          f"disp={disp:4.2f}m  {'FELL @ %.1fs' % fell_at if fell_at is not None else 'no fall'}",
          flush=True)


def euler_quat(roll, pitch, yaw):
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def run_fallen(name, rp, z0, trials, seconds=12.0, hold_s=1.0):
    rng = np.random.default_rng(0)
    times = []
    for _ in range(trials):
        model, data, bam_ctrl, policy, qa, _ = build()
        quat = euler_quat(rp[0], rp[1], rng.uniform(-np.pi, np.pi))
        reset(model, data, bam_ctrl, policy, qa, z=z0, quat=quat, joint_noise=0.1, rng=rng)
        if name == "sitting":
            for i, q in SITTING_JOINTS.items():
                data.qpos[policy.joint_qpos_indices[i]] = q
            if bam_ctrl is not None:
                bam_ctrl.reset(data.qpos)
            mujoco.mj_forward(model, data)
        policy.set_vel_cmd(0.0, 0.0, 0.0)
        up_since, rec_t = None, None
        for k in range(int(seconds * 50)):
            step(model, data, bam_ctrl, policy)
            up = tilt_deg(policy) < UP_TILT_DEG and data.qpos[qa + 2] > UP_Z
            up_since = (up_since if up_since is not None else k) if up else None
            if up_since is not None and (k - up_since) / 50 >= hold_s:
                rec_t = up_since / 50
                break
        times.append(rec_t)
    ok = [t for t in times if t is not None]
    med = f"median {np.median(ok):.1f}s, max {max(ok):.1f}s" if ok else "-"
    print(f"{name:18s} recovered {len(ok)}/{trials}  ({med})", flush=True)


print(f"onnx={args.onnx}  kind={args.kind}  "
      f"mode={'--no-bam (XML PD, NOT the training actuator)' if args.no_bam else 'BAM M6 (training actuator)'}")
if args.suite in ("walk", "all"):
    for s in WALK_SCENARIOS:
        run_walk(*s)
if args.suite in ("fallen", "all"):
    print(f"fallen spawns: HOME joints +-0.1 rad noise, random yaw, {args.trials} trials, 12 s, "
          f"recovered = tilt<{UP_TILT_DEG:.0f}deg & z>{UP_Z*1000:.0f}mm held 1 s")
    for s in FALLEN_SCENARIOS:
        run_fallen(*s, trials=args.trials)
