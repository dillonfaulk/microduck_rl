"""Measure the real duck's odometry heading drift from Pollen's ground truth (rung 6 Stage 2, 2.3).

    uv run python scripts/odom_dataset_check.py ~/microduck_refs/slam_gt [--out DIR]

This is the source for the IMU yaw-error magnitudes the sim's `ImuYawError` uses. For every
walking stretch of `SLAM_JJ_Salon_with_GT` s2/s3, it compares odometry's heading change and
position change with the truth over the same interval. A stretch is policy == "walk", not picked
up, not fallen, and has truth at both ends.

The heading error is then split into a part that grows with time (gyro bias, deg/min) and a part
that grows with the angle turned (gyro scale error, % of turn). The split is a least-squares fit on
non-overlapping 20 s windows inside the stretches.

The dataset is read as data only; nothing from its `code/` directory is imported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tof_dataset_check import quat_to_R, se2_compose, se2_inv, wrap


def load(root: Path, name: str):
    sess = next((root / "sessions" / name / "session").iterdir())
    t, xy, yaw, ok = [], [], [], []
    for line in open(sess / "state.jsonl"):
        p = json.loads(line)["params"]
        if not p.get("t_ns") or not p.get("odom"):
            continue
        s = p.get("safety") or {}
        t.append(p["t_ns"] * 1e-9)
        xy.append(p["odom"]["position"][:2])
        yaw.append(p["odom"]["yaw"])
        ok.append(p.get("policy") == "walk" and not s.get("picked_up") and not s.get("fallen"))
    t, xy, yaw, ok = map(np.array, (t, xy, yaw, ok))
    gt = np.loadtxt(root / "sessions" / name / "ground_truth" / "ground_truth.csv", delimiter=",", skiprows=1)
    Rg = np.array([quat_to_R(q) for q in gt[:, 4:8]])
    truth = np.column_stack([gt[:, 1], gt[:, 2], np.arctan2(Rg[:, 1, 0], Rg[:, 0, 0])])
    return t, xy, np.unwrap(yaw), ok, gt[:, 0] * 1e-9, truth


def stretches(t, ok, min_s=5.0):
    out, start = [], None
    for i, o in enumerate(ok):
        if o and start is None:
            start = i
        if (not o or i == len(ok) - 1) and start is not None:
            end = i if o else i - 1
            if t[end] - t[start] >= min_s:
                out.append((start, end))
            start = None
    return out


def analyse(root, name, window_s=20.0):
    t, xy, yaw, ok, tg, truth = load(root, name)
    odo = np.column_stack([xy, yaw])
    k_of = lambda tt: np.clip(np.searchsorted(t, tt), 0, len(t) - 1)   # noqa: E731
    turned = np.concatenate([[0], np.cumsum(np.abs(np.diff(yaw)))])
    path = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    per_stretch, windows = [], []
    for a, b in stretches(t, ok):
        g = np.where((tg >= t[a]) & (tg <= t[b]))[0]
        if len(g) < 2:
            continue

        def err(g0, g1):
            i, j = k_of(tg[g0]), k_of(tg[g1])
            mo = se2_compose(se2_inv(odo[i]), odo[j])
            mt = se2_compose(se2_inv(truth[g0]), truth[g1])
            e = se2_compose(se2_inv(mt), mo)
            return (float(np.degrees(wrap(e[2]))), float(np.hypot(e[0], e[1])), float(tg[g1] - tg[g0]),
                    float(np.degrees(turned[j] - turned[i])), float(path[j] - path[i]),
                    float(np.degrees(odo[j, 2] - odo[i, 2])))
        yaw_e, xy_e, dur, turn, walked, net = err(g[0], g[-1])
        per_stretch.append({"t0_s": round(float(t[a] - t[0]), 1), "dur_s": round(dur, 1), "walked_m": round(walked, 2),
                            "turned_deg": round(turn, 0), "yaw_err_deg": round(yaw_e, 2), "xy_err_cm": round(xy_e * 100, 1),
                            "xy_err_per_m": round(xy_e / walked, 3) if walked > 0.3 else None})
        g0 = g[0]
        while True:
            g1s = np.where(tg >= tg[g0] + window_s)[0]
            if not len(g1s) or g1s[0] > g[-1]:
                break
            g1 = g1s[0]
            ye, xe, d, tu, wk, nt = err(g0, g1)
            windows.append((ye, d, tu, nt, xe, wk))
            g0 = g1
    return per_stretch, np.array(windows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report, W = {}, []
    for name in ("s2", "s3"):
        ps, w = analyse(args.root, name)
        report[name] = {"stretches": ps}
        print(f"== {name}: {len(ps)} walking stretches, {len(w)} 20 s windows")
        for s in ps:
            print(f"   t0 {s['t0_s']:6.1f} s  {s['dur_s']:5.1f} s  walked {s['walked_m']:5.2f} m  turned {s['turned_deg']:5.0f} deg  "
                  f"yaw err {s['yaw_err_deg']:+6.2f} deg  xy err {s['xy_err_cm']:5.1f} cm  per m {s['xy_err_per_m']}")
        W.append(w)
    W = np.concatenate(W)
    ye, dur, turn, net, xe, wk = W.T
    # signed fit: yaw_err = bias * dur + scale * net_turn (signed net turn), per window
    A = np.column_stack([dur / 60.0, net])
    (bias_dpm, scale), *_ = np.linalg.lstsq(A, ye, rcond=None)
    resid = ye - A @ np.array([bias_dpm, scale])
    rate = np.abs(ye) / (dur / 60.0)
    walking = wk > 0.3
    fit = {"windows": int(len(W)), "window_s": 20.0,
           "abs_yaw_err_deg_per_min": {"median": round(float(np.median(rate)), 2), "p90": round(float(np.percentile(rate, 90)), 2)},
           "fit_bias_deg_per_min": round(float(bias_dpm), 2), "fit_scale_frac_of_net_turn": round(float(scale), 4),
           "fit_residual_std_deg_per_window": round(float(np.std(resid)), 2),
           "abs_yaw_err_per_rad_turned": {"median": round(float(np.median(np.abs(ye) / np.maximum(np.radians(turn), 0.2))), 2)},
           "xy_err_per_m_walked": {"median": round(float(np.median(xe[walking] / wk[walking])), 3),
                                   "p90": round(float(np.percentile(xe[walking] / wk[walking], 90)), 3)}}
    report["fit"] = fit
    print("\n== heading error over 20 s windows of walking (both sessions)")
    print(f"   |yaw err| rate: median {fit['abs_yaw_err_deg_per_min']['median']} deg/min, p90 {fit['abs_yaw_err_deg_per_min']['p90']} deg/min")
    print(f"   signed fit: bias {fit['fit_bias_deg_per_min']:+.2f} deg/min + scale {fit['fit_scale_frac_of_net_turn']:+.4f} x net turn"
          f"  (residual std {fit['fit_residual_std_deg_per_window']} deg per window)")
    print(f"   |yaw err| per rad turned: median {fit['abs_yaw_err_per_rad_turned']['median']} deg/rad")
    print(f"   xy err per m walked (20 s windows, >0.3 m): median {fit['xy_err_per_m_walked']['median']:.1%}, p90 {fit['xy_err_per_m_walked']['p90']:.1%}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "odom_dataset_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
