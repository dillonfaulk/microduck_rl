"""Rung 6 Stage 4, Workstream B: what the real mapd contributes, near the boards vs in the open.

The duck walks two laps of a 1 m square (go-to-point between corners), stopping at every corner for
autod's survey (stand still, head yaw 0 / +0.6 / -0.6 / 0; mapd keys a keyframe after 0.4 s still).
It drives on the MAPD estimate (odometry + stride_v1 + mapd's offset). A second, odometry-only
provider with the SAME IMU-error seed rides along passively on the identical motion, so the
difference between the two at each stop is exactly mapd's contribution, nothing else.

Regimes (same square, same coordinates):
  near-boards: field scene, square at (1.7..2.7, 1.7..2.7): the +-3 m boards 0.3-1.3 m away.
  open field:  no boards (scene_allcollisions): the ToF has nothing to match.

  uv run python scripts/mapd_regimes.py <velstand.onnx> [--trials 10] [--out FILE]
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goto_point import SCENE, DuckSim  # noqa: E402
from goto_supervisor import GotoConfig, GotoSupervisor  # noqa: E402
from pose_provider import ImuYawError, OdometryProvider, make_provider, pose_error  # noqa: E402

FIELD = "src/mjlab_microduck/robot/microduck/scene_field.xml"
SQUARE = [(1.7, 1.7), (2.7, 1.7), (2.7, 2.7), (1.7, 2.7)]
LAPS = 2
SURVEY = (0.0, 0.6, -0.6, 0.0)
SURVEY_STEP_S = 0.6
ARENA = (-2.9, 2.9, -2.9, 2.9)


def trial(sim, seed):
    sim.reset(SQUARE[0][0], SQUARE[0][1], np.random.default_rng(seed).uniform(-math.pi, math.pi))
    md = make_provider("mapd", sim, seed=seed, calib="stride_v1") if trial.md is None else trial.md
    trial.md = md
    md.reset(sim, seed=seed)
    od = OdometryProvider(sim, imu=ImuYawError(seed), calib="stride_v1")   # passive
    od.reset(sim, seed=seed)                    # the SAME IMU-error seed as the mapd provider
    stops = []

    def step(twist):
        sim.set_twist(twist)
        sim.step()
        md.update(sim)
        od.update(sim)

    def survey():
        for yaw in SURVEY:
            sim.policy.head_offset[:] = [0.0, 0.0, yaw, 0.0]
            for _ in range(int(SURVEY_STEP_S / 0.02)):
                step((0.0, 0.0, 0.0))
        sim.policy.head_offset[:] = 0.0
        true = sim.pose()[0]
        e_m = pose_error(md.get(sim.t).pose, true)
        e_o = pose_error(od.get(sim.t).pose, true)
        stops.append({"mapd_mm": e_m[0] * 1000, "odom_mm": e_o[0] * 1000,
                      "mapd_yaw": abs(math.degrees(e_m[1])), "odom_yaw": abs(math.degrees(e_o[1]))})

    survey()
    route = (SQUARE[1:] + SQUARE[:1]) * LAPS
    falls = 0
    for wp in route:
        sup = GotoSupervisor([wp], ARENA, GotoConfig(arrival_radius=0.08))
        t0 = sim.t
        while sup.state != "DONE" and sim.t - t0 < 25.0:
            e = md.get(sim.t)
            sup.update(e.t, e.pose, e.tilt_deg, e.trunk_z)
            step(sup.command(sim.t))
        falls += sup.falls
        survey()
    s = md.summary()
    return stops, s, falls


trial.md = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("velstand")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lines = ["mapd regimes: 2 laps of a 1 m square, survey stop at every corner (9 stops), driven on the mapd "
             "estimate; odometry-only (same IMU seed) rides along passively. Pose error at each stop, mm / deg.",
             "| regime | stops | odom-only xy mm med/p90/max | WITH mapd xy mm med/p90/max | odom |yaw| deg med/max | "
             "mapd |yaw| deg med/max | last stop odom -> mapd mm (med) | corrections/trial med | keyframes/trial med | lost at end |"]
    for regime, scene in (("near-boards", FIELD), ("open field", SCENE)):
        sim = DuckSim(args.velstand, scene=scene)
        trial.md = None
        all_stops, last, corr, kfs, lost = [], [], [], [], 0
        for k in range(args.trials):
            stops, s, falls = trial(sim, k)
            all_stops += stops
            last.append((stops[-1]["odom_mm"], stops[-1]["mapd_mm"]))
            corr.append(s["mapd_corrections"])
            kfs.append(s["mapd_keyframes"] or 0)
            lost += bool(s["mapd_lost"])
            print(f"  {regime} trial {k}: last stop odom {stops[-1]['odom_mm']:.0f} mm -> mapd {stops[-1]['mapd_mm']:.0f} mm, "
                  f"{s['mapd_corrections']} corrections, {s['mapd_keyframes']} keyframes, lost {s['mapd_lost']}, falls {falls}",
                  flush=True)
        trial.md.shim.close()

        def q(key):
            v = [st[key] for st in all_stops]
            return f"{np.median(v):.1f} / {np.percentile(v, 90):.1f} / {max(v):.1f}"

        def q2(key):
            v = [st[key] for st in all_stops]
            return f"{np.median(v):.2f} / {max(v):.2f}"
        lines.append(f"| {regime} | {len(all_stops)} | {q('odom_mm')} | {q('mapd_mm')} | {q2('odom_yaw')} | {q2('mapd_yaw')} | "
                     f"{np.median([a for a, _ in last]):.0f} -> {np.median([b for _, b in last]):.0f} | "
                     f"{np.median(corr):.0f} | {np.median(kfs):.0f} | {lost}/{args.trials} |")
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
