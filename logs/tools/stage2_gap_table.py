"""Rung 6 Stage 2 gap table: GoalKick on ground truth vs estimated pose, from kick_mission stdout.

    python logs/tools/stage2_gap_table.py logs/evals/vision/stage2_2026-11-14

Reads goalkick/{right,left}_goal{0.5,0.8,1.1}_field_{gt,odom}.txt (per-run `run k:` lines) and
prints one row per (foot, distance, pose source), plus the published goal-scene baselines.
"""
import math
import re
import statistics as st
import sys
from pathlib import Path

PUBLISHED = {("right", "0.5"): 50, ("right", "0.8"): 43, ("right", "1.1"): 38,
             ("left", "0.5"): 50, ("left", "0.8"): 49, ("left", "1.1"): 44}
NUM = r"([-+]?\d+(?:\.\d+)?|None|nan)"


def runs(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("  run "):
            continue
        r = {}
        for key in ("goal_outcome",):
            m = re.search(rf"{key}=(\w+)", line)
            r[key] = m.group(1) if m else None
        for key in ("goal_lateral_mm", "pose_err_mm", "pose_err_yaw_deg", "true_align_ex_mm", "true_align_ey_mm",
                    "falls", "ball_disturbed", "peak_ball_fwd"):
            m = re.search(rf"\b{key}={NUM}", line)
            r[key] = float(m.group(1)) if m and m.group(1) not in ("None", "nan") else None
        m = re.search(r"aborted=(\w+)", line)
        r["aborted"] = m.group(1) if m else None
        r["board"] = "duck_board_contact=True" in line
        r["post_in"] = "goal_went_in=True" in line and r["goal_outcome"] == "post"
        out.append(r)
    return out


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else float("nan")


def p90(v):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(math.ceil(0.9 * len(v))) - 1)] if v else float("nan")


def main():
    root = Path(sys.argv[1])
    hdr = (f"{'foot':5s} {'D':>4s} {'pose':4s} | {'goals':>5s} {'post(in)':>8s} {'miss':>4s} {'short':>5s} {'abort':>5s} "
           f"{'bump':>4s} {'fall':>4s} {'board':>5s} | {'|lat| med':>9s} | {'pose err mm med/p90':>19s} {'|yaw| med':>9s} | "
           f"{'TRUE |ex| med':>13s} {'TRUE |ey| med/p90':>17s} | published goals")
    print(hdr)
    print("-" * len(hdr))
    for foot in ("right", "left"):
        for d in ("0.5", "0.8", "1.1"):
            for pose in ("gt", "odom"):
                f = root / "goalkick" / f"{foot}_goal{d}_field_{pose}.txt"
                if not f.exists():
                    continue
                rs = runs(f)
                oc = lambda o: sum(r["goal_outcome"] == o for r in rs)  # noqa: E731
                lat = [abs(r["goal_lateral_mm"]) for r in rs if r["goal_lateral_mm"] is not None]
                print(f"{foot:5s} {d:>4s} {pose:4s} | {oc('goal'):5d} {oc('post'):4d}({sum(r['post_in'] for r in rs)})  "
                      f"{oc('miss'):4d} {oc('no_cross'):5d} {sum(r['aborted'] is not None for r in rs):5d} "
                      f"{sum((r['ball_disturbed'] or 0) > 0 for r in rs):4d} {int(sum(r['falls'] or 0 for r in rs)):4d} "
                      f"{sum(r['board'] for r in rs):5d} | {med(lat):9.0f} | "
                      + (f"{med([r['pose_err_mm'] for r in rs]):9.1f} / {p90([r['pose_err_mm'] for r in rs]):6.1f}  "
                         f"{med([abs(r['pose_err_yaw_deg']) for r in rs if r['pose_err_yaw_deg'] is not None]):9.2f} | "
                         f"{med([abs(r['true_align_ex_mm']) for r in rs if r['true_align_ex_mm'] is not None]):13.1f} "
                         f"{med([abs(r['true_align_ey_mm']) for r in rs if r['true_align_ey_mm'] is not None]):8.1f} / "
                         f"{p90([abs(r['true_align_ey_mm']) for r in rs if r['true_align_ey_mm'] is not None]):6.1f} | "
                         if pose == "odom" else f"{'-':>19s} {'-':>9s} | {'-':>13s} {'-':>17s} | ")
                      + f"{PUBLISHED[(foot, d)]} (n={len(rs)})")


if __name__ == "__main__":
    main()
