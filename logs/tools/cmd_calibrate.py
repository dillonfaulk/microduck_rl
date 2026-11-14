"""Measure achieved velocity vs FED command per axis on the CPU BAM harness (projected gravity,
delay 1, training solver, all-collisions scene). Run: python logs/tools/cmd_calibrate.py <onnx>"""
import sys
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], "--suite", "walk", "--delay", "1", "--train-solver",
            "--scene", "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"]
src = open("logs/tools/headless_rehearsal.py").read().split('print(f"onnx={args.onnx}')[0]
exec(compile(src, "headless_rehearsal.py", "exec"))


def achieved(cmd, seconds=8.0, yaw0=0.0):
    model, data, bam_ctrl, policy, qa, va = build()
    reset(model, data, bam_ctrl, policy, qa, quat=euler_quat(0, 0, yaw0))
    policy.set_vel_cmd(*cmd)
    vel, fell = [], False
    for k in range(int(seconds * 50)):
        step(model, data, bam_ctrl, policy)
        if k >= 100:
            quat = data.qpos[qa + 3:qa + 7].astype(np.float32)
            vb = policy.quat_rotate_inverse(quat, data.qvel[va:va + 3].astype(np.float32))
            vel.append((vb[0], vb[1], data.qvel[va + 5]))
        if tilt_deg(policy) > FALL_TILT_DEG:
            fell = True
            break
    return np.mean(vel, axis=0) if vel else np.zeros(3), fell


GRID = {0: [-0.6, -0.5, -0.4, -0.35, -0.3, -0.25, -0.2, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
        1: [-0.45, -0.35, -0.3, -0.25, -0.2, 0.2, 0.25, 0.3, 0.35, 0.45],
        2: [-1.4, -1.0, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 1.0, 1.4]}
for axis, vals in GRID.items():
    for v in vals:
        c = [0.0, 0.0, 0.0]; c[axis] = v
        rs = [achieved(c, yaw0=y) for y in (0.0, 1.5)]
        a = np.mean([r[0] for r in rs], axis=0)
        print(f"axis {axis} fed {v:+.2f} -> achieved ({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f}) falls {sum(r[1] for r in rs)}/2",
              flush=True)
