"""How far a kicked ball rolls under the play BallDrag (from the battery traces): ball position at the kick
swap-in vs 10 s later (kicks whose ball was not respawned in between)."""
import csv
import glob
import json
import math
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "logs/evals/play/continuous_2026-11-16/logs"
for bat in sorted(glob.glob(f"{root}/B_*")):
    rolls = []
    for runs in glob.glob(f"{bat}/runs.jsonl"):
        for line in open(runs):
            r = json.loads(line)
            tr = list(csv.DictReader(open(f"{bat}/run{r['run']}_trace.csv")))
            ts = np.array([float(x["t"]) for x in tr])
            resp = [e["t"] for e in r["events"]]
            for k in r["kick_list"]:
                t0 = k["t_swap"]
                if any(t0 < t < t0 + 10.5 for t in resp) or t0 + 10 > ts[-1]:
                    continue
                i0, i1 = int(np.searchsorted(ts, t0)), int(np.searchsorted(ts, t0 + 10))
                a, b = tr[i0], tr[i1]
                rolls.append(math.hypot(float(b["ball_x"]) - float(a["ball_x"]), float(b["ball_y"]) - float(a["ball_y"])))
    if rolls:
        print(f"{bat.split('/')[-1]:28s} kicks {len(rolls):3d}: roll in 10 s median {np.median(rolls):.2f} m, "
              f"p10 {np.percentile(rolls, 10):.2f}, p90 {np.percentile(rolls, 90):.2f}, max {max(rolls):.2f}")
