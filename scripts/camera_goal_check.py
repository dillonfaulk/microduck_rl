"""Goal ranging vs ground truth (rung 6 Stage 4, Workstream A acceptance): before goal perception
feeds a mission, measure it. Duck standing (VelStand, head neutral) in the field scene; the goal
placed at each range at 5 bearings x 3 facings (15 placements). Reports found rate, centre error for
the fit and for base-ray->plane only, facing error, and the KICK-HEADING error that matters: the
direction from a ball 0.8 m in front of the goal to the measured vs true goal centre.

  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run python scripts/camera_goal_check.py <velstand.onnx>
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_goal import TARGET_GATE_M, GoalCamera, assign_target  # noqa: E402
from kick_battery import SCENES, KickSim  # noqa: E402
from kick_mission import place_goal, setup_field, setup_goal  # noqa: E402

DISTS = (0.6, 0.9, 1.2, 1.6, 2.0)
HFOV_FRAC = (-0.6, -0.3, 0.0, 0.3, 0.6)     # bearing as a fraction of the half horizontal FOV
FACINGS_DEG = (-35.0, 0.0, 35.0)            # goal facing vs the duck->goal line (GOAL_APPROACH-like)
BALL_BEFORE_GOAL_M = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("velstand")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dists", default=None, help="comma list of ranges, m (default 0.6,0.9,1.2,1.6,2.0)")
    args = ap.parse_args()
    sim = KickSim(args.velstand, SCENES["field"])
    setup_goal(sim)
    setup_field(sim)
    from mjlab_microduck.sim import board_profiles
    cam = GoalCamera(sim.model, args.profile or board_profiles.PRIMARY)
    half_h = math.atan(cam.K["cx"] / cam.K["fx"])
    lines = [f"goal ranging, profile {cam.profile.name}, {cam.w}x{cam.h}, fx {cam.K['fx']:.1f}, half hFOV "
             f"{math.degrees(half_h):.1f} deg; 15 placements per range (5 bearings x 3 facings)",
             "| dist | found | fit centre err mm med/max | base-only centre err mm med/max | facing err deg med/max "
             "| KICK-HEADING err deg med/max (ball 0.8 m before goal) | spacing m med |"]
    rng = np.random.default_rng(0)
    for dist in (DISTS if args.dists is None else [float(v) for v in args.dists.split(",")]):
        rows = []
        for frac in HFOV_FRAC:
            for face in FACINGS_DEG:
                yaw0 = rng.uniform(-math.pi, math.pi)
                sim.reset(0.0, 0.0, yaw0)
                sim.data.qpos[sim.bqa:sim.bqa + 3] = [9.0, 9.0, 0.035]     # ball out of the way
                for _ in range(40):
                    sim.step()
                (x, y, yaw), _, _ = sim.pose()
                bear = yaw + frac * half_h
                centre = (x + dist * math.cos(bear), y + dist * math.sin(bear))
                into = bear + math.radians(face)
                place_goal(sim, centre, into)
                mujoco.mj_forward(sim.model, sim.data)
                t = assign_target(centre)
                c0, s0 = math.cos(yaw), math.sin(yaw)
                near = (c0 * (t[0] - x) + s0 * (t[1] - y), -s0 * (t[0] - x) + c0 * (t[1] - y))
                m = cam.measure(sim.data, near_rel=near, gate=TARGET_GATE_M)
                if m is None:
                    rows.append(None)
                    continue
                c, s = math.cos(yaw), math.sin(yaw)

                def world(rel):
                    return np.array([x + c * rel[0] - s * rel[1], y + s * rel[0] + c * rel[1]])

                est_c, base_c = world(m["centre"]), world(m["base_centre"])
                true_c = np.array(centre)
                ball = true_c - BALL_BEFORE_GOAL_M * np.array([math.cos(into), math.sin(into)])
                kh_true = math.atan2(*(true_c - ball)[::-1])
                kh_est = math.atan2(*(est_c - ball)[::-1])
                face_err = (yaw + m["into"] - into + math.pi) % (2 * math.pi) - math.pi
                rows.append({"fit": np.linalg.norm(est_c - true_c), "base": np.linalg.norm(base_c - true_c),
                             "face": abs(math.degrees(face_err)),
                             "kh": abs(math.degrees((kh_est - kh_true + math.pi) % (2 * math.pi) - math.pi)),
                             "sp": m["spacing_m"]})
        got = [r for r in rows if r is not None]

        def mm(k, scale=1000.0):
            v = [r[k] * scale for r in got]
            return f"{np.median(v):.1f} / {max(v):.1f}" if v else "-"
        lines.append(f"| {dist:.1f} | {len(got)}/{len(rows)} | {mm('fit')} | {mm('base')} | {mm('face', 1.0)} | "
                     f"{mm('kh', 1.0)} | {np.median([r['sp'] for r in got]) if got else float('nan'):.3f} |")
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
