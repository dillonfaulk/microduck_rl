"""Camera ball ranging vs ground truth (rung 6 Stage 3, Workstream B).

    MUJOCO_GL=egl python scripts/camera_ranging_check.py <velstand.onnx> [--profile face-board-v1]

The duck stands (VelStand, zero twist) on the field scene; the ball is placed at 0.3 / 0.5 / 0.8 /
1.1 m ahead, at lateral offsets across the field of view, at several duck headings. For each
placement the head camera frame is rendered and both rangers are scored against the truth in the
duck's yaw frame: ray -> plane (primary) and apparent size (cross-check). Head neutral, plus the
mission's re-observe head pose (kick_mission.LOOK_HEAD) for the near distances.
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kick_mission as km  # noqa: E402
from camera_ball import BallCamera  # noqa: E402
from kick_battery import SCENES, KickSim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("velstand")
    ap.add_argument("--profile", default="face-board-v1")
    args = ap.parse_args()
    sim = KickSim(args.velstand, SCENES["field"])
    sim.reset(0, 0, 0)
    km.setup_goal(sim)
    sim.arena = km.setup_field(sim)
    cam = BallCamera(sim.model, args.profile)
    print(f"profile {args.profile}: fx {cam.K['fx']:.1f} at {cam.w}x{cam.h} (from the profile's fovy, pinhole)")
    heads = {"neutral": (0.0, 0.0, 0.0, 0.0), "look": km.LOOK_HEAD}
    print(f"{'head':8s} {'dist':>5s} | {'found':>6s} | {'ray |err| med / max mm':>24s} | {'size |err| med / max mm':>24s} | ray err range-only med")
    for hname, head in heads.items():
        for dist in (0.15, 0.21, 0.3, 0.5, 0.8, 1.1):
            ray_e, size_e, ray_r, n, found = [], [], [], 0, 0
            for yaw in (0.0, 1.3, -2.2):
                sim.reset(0, 0, yaw)
                sim.policy.head_offset[:] = head
                for _ in range(50):
                    sim.set_twist((0, 0, 0))
                    sim.step()
                R = sim.data.xmat[cam.trunk].reshape(3, 3)
                ty = math.atan2(R[1, 0], R[0, 0])
                c, s = math.cos(ty), math.sin(ty)
                pt = sim.data.xpos[cam.trunk]
                for lat_frac in (-0.6, -0.3, 0.0, 0.3, 0.6):
                    ly = lat_frac * dist * math.tan(math.radians(cam.profile.camera.hfov_deg / 2))
                    n += 1
                    sim.data.qpos[sim.bqa:sim.bqa + 7] = [pt[0] + c * dist - s * ly, pt[1] + s * dist + c * ly, 0.035, 1, 0, 0, 0]
                    mujoco.mj_forward(sim.model, sim.data)
                    m = cam.measure(sim.data)
                    if m is None or m["edge"] or "ray" not in m:
                        continue
                    found += 1
                    ray_e.append(math.hypot(m["ray"][0] - dist, m["ray"][1] - ly))
                    size_e.append(math.hypot(m["size"][0] - dist, m["size"][1] - ly))
                    ray_r.append(math.hypot(*m["ray"]) - math.hypot(dist, ly))
            sim.policy.head_offset[:] = 0.0
            if not found:
                print(f"{hname:8s} {dist:5.2f} | {found:2d}/{n:2d}  | {'not in view':>24s} |")
                continue
            print(f"{hname:8s} {dist:5.2f} | {found:2d}/{n:2d}  | {np.median(ray_e)*1000:10.1f} / {max(ray_e)*1000:6.1f}      "
                  f"| {np.median(size_e)*1000:10.1f} / {max(size_e)*1000:6.1f}      | {np.median(ray_r)*1000:+.1f}")
    os._exit(0)   # skip EGL teardown noise


if __name__ == "__main__":
    main()
