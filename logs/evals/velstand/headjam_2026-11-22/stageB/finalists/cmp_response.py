import re, sys
def load(f):
    rows = {}
    for l in open(f):
        m = re.match(r"\(([-+\d., ]+)\)\s+\(([-+\d., ]+)\)", l.strip())
        if m:
            rows[tuple(float(x) for x in m.group(1).split(","))] = [float(x) for x in m.group(2).split(",")]
    return rows
ref = load(sys.argv[1])
for f in sys.argv[2:]:
    new = load(f); bad = []; worst = 0
    for c, r in ref.items():
        if c not in new: continue
        for k in range(3):
            if abs(c[k]) < 1e-9: continue
            if abs(r[k]) < 0.03 and abs(new[c][k]) < 0.03: continue    # both dead (dead-zone rows)
            d = (new[c][k] - r[k]) / max(abs(r[k]), 0.03)
            worst = max(worst, abs(d))
            if abs(d) > 0.10: bad.append(f"{c} axis{k}: {r[k]:+.3f} -> {new[c][k]:+.3f} ({100*d:+.0f}%)")
    print(f"{f}: rows {len(new)}/{len(ref)}, outside +-10%: {len(bad)}, worst {100*worst:.0f}%")
    for b in bad: print("   ", b)
