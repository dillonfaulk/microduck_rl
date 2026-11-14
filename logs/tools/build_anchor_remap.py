"""Build the walk-anchor command-remap table from warp_cmd_response.py output.

Monotone inverse (running max) of the anchor's measured fed -> achieved response, per axis
and sign, capped at FED_CAP. Prints a dict literal for the velstand cfg.
Run: python logs/tools/build_anchor_remap.py logs/evals/velstand/anchor_cmd_response_4750.txt
"""
import re
import sys

import numpy as np

FED_CAP = {"x_pos": 0.5, "x_neg": 0.4, "y": 0.4, "wz": 1.3}  # beyond: drift (x back) / few samples
rows = []
for line in open(sys.argv[1]):
    m = re.match(r"\(([-+.\d]+), ([-+.\d]+), ([-+.\d]+)\)\s+\(([-+.\dna]+), ([-+.\dna]+), ([-+.\dna]+)\)", line)
    if m:
        rows.append(tuple(float(v) for v in m.groups()))
R = np.array(rows)


def inverse(fed, ach, targets, cap):
    keep = fed <= cap + 1e-9
    fed, ach = np.r_[0.0, fed[keep]], np.r_[0.0, np.maximum.accumulate(ach[keep])]
    ach = ach + np.arange(ach.size) * 1e-6  # strictly increasing for interp
    return tuple(round(float(np.interp(t, ach, fed)), 3) for t in targets)


def axis(col, sign, other_zero, cap, targets):
    sel = np.all(R[:, other_zero] == 0, axis=1) & (R[:, col] * sign > 0)
    order = np.argsort(np.abs(R[sel, col]))
    return inverse(np.abs(R[sel, col])[order], (R[sel, 3 + col] * sign)[order], targets, cap)


XT = (0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25)
YT = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
if "--wide" in sys.argv:
    # walk v2 reaches ~0.36 m/s / strafe ~0.13: targets past the last knot get the last fed
    # (distill.remap_anchor_twist), so the knots must span the commanded range.
    XT = XT + (0.3, 0.35, 0.4)
    YT = YT + (0.115, 0.13, 0.15)
WT = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
table = {
    "x": {"targets": XT, "pos": axis(0, 1, [1, 2], FED_CAP["x_pos"], XT), "neg": axis(0, -1, [1, 2], FED_CAP["x_neg"], XT)},
    "y": {"targets": YT, "pos": axis(1, 1, [0, 2], FED_CAP["y"], YT), "neg": axis(1, -1, [0, 2], FED_CAP["y"], YT)},
}
# yaw rows keyed by the anchor's ACHIEVED |vx| at each fed vx (the x remap feeds it that vx)
lin, pos, neg = [0.0], [axis(2, 1, [0, 1], FED_CAP["wz"], WT)], [axis(2, -1, [0, 1], FED_CAP["wz"], WT)]
for vx in (0.2, 0.3, 0.4):
    sel = (R[:, 0] == vx) & (R[:, 1] == 0) & (R[:, 2] != 0)
    S = R[sel]
    lin.append(round(float(np.abs(S[:, 3]).mean()), 3))
    for sign, out in ((1, pos), (-1, neg)):
        s = S[S[:, 2] * sign > 0]
        o = np.argsort(np.abs(s[:, 2]))
        out.append(inverse(np.abs(s[o, 2]), s[o, 5] * sign, WT, FED_CAP["wz"]))
table.update(lin=tuple(lin), targets=WT, pos=tuple(pos), neg=tuple(neg))
import pprint
pprint.pprint(table, width=110, sort_dicts=False)
