"""Corner-post ranging vs ground truth (4 m field): duck standing (VelStand, head neutral), a corner post
at each range and 3 bearings across the view. Reports found, range error, bearing error, and position error
of the post. Also counts false detections looking away from every post.

  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run python scripts/camera_corner_check.py <velstand.onnx> [--out FILE]
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_landmarks import CornerCamera, corner_map  # noqa: E402
from kick_battery import SCENES, KickSim  # noqa: E402
from kick_mission import setup_field, setup_goal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("velstand")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sim = KickSim(args.velstand, SCENES["field4"])
    setup_goal(sim)
    setup_field(sim)
    cam = CornerCamera(sim.model)
    corner = np.array(corner_map(sim.model)[0])            # one post; the duck is placed relative to it
    half_h = math.atan(cam.K["cx"] / cam.K["fx"])
    lines = ["corner-post ranging, face-board-v1, duck standing head-neutral; post 75 mm x 0.40 m blue",
             "| range m | found | range err mm med/max | bearing err deg med/max | position err mm med/max |"]
    for dist in (0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        rows = []
        for frac in (-0.6, 0.0, 0.6):
            # stand inside the field, on the diagonal-ish towards the field centre
            ang_in = math.atan2(-corner[1], -corner[0]) + 0.25 * frac
            x, y = corner + dist * np.array([math.cos(ang_in), math.sin(ang_in)])
            yaw = math.atan2(corner[1] - y, corner[0] - x) - frac * half_h
            sim.reset(x, y, yaw)
            sim.data.qpos[sim.bqa:sim.bqa + 3] = [9.0, 9.0, 0.035]
            for _ in range(40):
                sim.step()
            (x, y, yaw), _, _ = sim.pose()
            c, s = math.cos(yaw), math.sin(yaw)
            d = corner - (x, y)
            true = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
            ms = cam.measure(sim.data)
            if not ms:
                rows.append(None)
                continue
            m = min(ms, key=lambda q: np.linalg.norm(np.array(q["rel"]) - true))
            rel = np.array(m["rel"])
            rows.append({"rng": abs(np.linalg.norm(rel) - np.linalg.norm(true)) * 1000,
                         "brg": abs(math.degrees(math.atan2(rel[1], rel[0]) - math.atan2(true[1], true[0]))),
                         "pos": np.linalg.norm(rel - true) * 1000})
        got = [r for r in rows if r]

        def q(k):
            v = [r[k] for r in got]
            return f"{np.median(v):.1f} / {max(v):.1f}" if v else "-"
        lines.append(f"| {dist:.1f} | {len(got)}/{len(rows)} | {q('rng')} | {q('brg')} | {q('pos')} |")
        print(lines[-1], flush=True)
    false = 0
    for yaw in np.linspace(-math.pi, math.pi, 12, endpoint=False):
        sim.reset(0.0, 0.0, yaw)
        for _ in range(40):
            sim.step()
        for m in cam.measure(sim.data):
            (x, y, yw), _, _ = sim.pose()
            c, s = math.cos(yw), math.sin(yw)
            w = np.array([x + c * m["rel"][0] - s * m["rel"][1], y + s * m["rel"][0] + c * m["rel"][1]])
            if min(np.linalg.norm(w - np.array(p)) for p in corner_map(sim.model)) > 0.3:
                false += 1
    lines.append(f"false detections (not within 0.3 m of a corner post), 12 headings from the centre: {false}")
    print(lines[-1])
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
