"""False-bump audit: every BALL_DISTURBED in a battery, classified against ground truth (ball displacement
within the next 1.0 s from the 10 Hz trace): REAL if the ball truly moved > 20 mm, else FALSE. Also reports
the duck-ball range at the call, and what the false call cost (time waiting)."""
import csv, glob, json, math, sys
import numpy as np
root = sys.argv[1]
for bat in sorted(glob.glob(f"{root}/*")):
    real, false, rng_f, rng_r = 0, 0, [], []
    for f in glob.glob(f"{bat}/run*_events.jsonl"):
        tr = list(csv.DictReader(open(f.replace("_events.jsonl", "_trace.csv"))))
        ts = np.array([float(r["t"]) for r in tr])
        for l in open(f):
            e = json.loads(l)
            if e["event"] != "BALL_DISTURBED":
                continue
            i0 = int(np.searchsorted(ts, e["t"] - 0.15)); i1 = int(np.searchsorted(ts, e["t"] + 1.0))
            i0, i1 = min(i0, len(tr) - 1), min(i1, len(tr) - 1)
            a, b = tr[i0], tr[i1]
            mv = math.hypot(float(b["ball_x"]) - float(a["ball_x"]), float(b["ball_y"]) - float(a["ball_y"]))
            r = math.hypot(float(a["ball_x"]) - float(a["x"]), float(a["ball_y"]) - float(a["y"]))
            if mv > 0.02:
                real += 1; rng_r.append(r)
            else:
                false += 1; rng_f.append(r)
    n_runs = len(glob.glob(f"{bat}/run*_events.jsonl"))
    if n_runs:
        print(f"{bat.split('/')[-1]:28s} runs {n_runs:2d}: BALL_DISTURBED {real + false:3d} -> real {real:3d}, FALSE {false:3d}"
              + (f"; duck-ball range at a false call median {np.median(rng_f):.2f} m (p10 {np.percentile(rng_f, 10):.2f}, p90 {np.percentile(rng_f, 90):.2f})" if rng_f else "")
              + (f"; at a real one median {np.median(rng_r):.2f} m" if rng_r else ""))
