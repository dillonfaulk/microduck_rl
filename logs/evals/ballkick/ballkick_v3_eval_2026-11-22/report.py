"""BallKick v3 sweep report: per tag, uncal missions (kicked, drift median/std, peak speed, >1.6x) and the lateral
sensitivity: kick_dir_err_deg regressed on TRUE lateral placement (align_ey_mm, gt pose) over the lateral sweep,
with R^2 and a bootstrap CI; also the multi-regression with ex / eh (the closing-work form)."""
import glob, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = lambda k, s: (lambda m: float(m.group(1)) if m else None)(re.search(rf"{k}=(?:np\.float64\()?([-+0-9.e]+)", s))


def runs(path):
    out = []
    for line in open(path):
        if not line.startswith("  run ") or "kick_dir_err_deg" not in line:
            continue
        out.append({k: F(k, line) for k in ("align_ex_mm", "align_ey_mm", "align_eh_deg", "kick_dir_err_deg", "peak_ball_fwd",
                                             "ball_travel")})
    n = sum(1 for line in open(path) if line.startswith("  run "))
    return out, n


def main(tags):
    print(f"{'tag':<16}{'kicked':>8}{'drift':>8}{'std':>6}{'peak':>7}{'>0.40':>7} | lateral: slope deg/mm [95% CI]   R^2   n   "
          f"(+ex/eh: slope, R^2)  | true lat std mm")
    for tag in tags:
        U = [r for s in (0, 1) for r in runs(os.path.join(HERE, f"{tag}_uncal_s{s}.txt"))[0]] if os.path.exists(
            os.path.join(HERE, f"{tag}_uncal_s0.txt")) else []
        nU = sum(runs(os.path.join(HERE, f"{tag}_uncal_s{s}.txt"))[1] for s in (0, 1)) if U else 0
        L = [r for p in sorted(glob.glob(os.path.join(HERE, f"{tag}_lat*.txt"))) for r in runs(p)[0]]
        L = [r for r in L if r["kick_dir_err_deg"] is not None and r["align_ey_mm"] is not None]
        line = f"{tag:<16}"
        if U:
            d = np.array([r["kick_dir_err_deg"] for r in U if r["kick_dir_err_deg"] is not None])
            pk = np.array([r["peak_ball_fwd"] for r in U if r["peak_ball_fwd"] is not None])
            line += f"{len(U):>4}/{nU:<3}{np.median(d):>+8.1f}{d.std():>6.1f}{np.median(pk):>7.3f}{int((pk > 0.40).sum()):>7}"
        else:
            line += f"{'':>36}"
        if len(L) >= 10:
            x = np.array([r["align_ey_mm"] for r in L]); y = np.array([r["kick_dir_err_deg"] for r in L])
            sl, ic = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
            rng = np.random.default_rng(0); bs = []
            for _ in range(2000):
                i = rng.integers(0, len(x), len(x)); bs.append(np.polyfit(x[i], y[i], 1)[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            X = np.column_stack([x, [r["align_ex_mm"] for r in L], [r["align_eh_deg"] for r in L], np.ones(len(x))])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2m = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            line += f" | {sl:+6.2f} [{lo:+.2f},{hi:+.2f}]  {r2:4.2f} {len(x):3d}   ({b[0]:+.2f}, {r2m:.2f})  | {x.std():.1f}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
