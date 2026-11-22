"""Near-post aim scoping (2026-11-22): counterfactual aim rules on the logged kicks (kicks.jsonl from analyze.py).
Each kick keeps its own deviation from where it meant to go (dev = actual - intended); the intended line is
swapped for a candidate rule's line. Geometry only: ignores changed post physics, the approach the new
heading needs, and whether the ball is reachable from the new stance."""
import json, math, numpy as np
ks = [json.loads(l) for l in open("kicks.jsonl")]
CLEAN, MOUTH = 0.070, 0.105
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def cross_y(b, ang, gx):
    dx = gx - b[0]
    if math.cos(ang) * dx <= 1e-6: return None
    return b[1] + math.tan(ang) * dx
def rule_dir(k, rule):
    gx = k["goal"][0]; s = np.sign(gx); b = k["b0"]
    if rule == "deep0.3": tgt = (gx + s * 0.3, 0.0)
    elif rule == "centre": tgt = (gx, 0.0)
    elif rule == "bisect":   # bisect the clean window (lines through y = +-70 mm on the goal line)
        a1 = math.atan2(CLEAN - b[1], gx - b[0]); a2 = math.atan2(-CLEAN - b[1], gx - b[0])
        return math.atan2(math.sin(a1) + math.sin(a2), math.cos(a1) + math.cos(a2))
    return math.atan2(tgt[1] - b[1], tgt[0] - b[0])
def window_deg(k):
    gx = k["goal"][0]; b = k["b0"]
    if np.sign(gx) * (gx - b[0]) <= 0.0: return 0.0
    a1 = math.atan2(CLEAN - b[1], gx - b[0]); a2 = math.atan2(-CLEAN - b[1], gx - b[0])
    return abs(math.degrees(wrap(a1 - a2))) / 2
nr = [k for k in ks if (k["post_m"] < 0.5 or k["duck_post_m"] < 0.5) and k["map"] and not k["behind"]]
for k in nr:
    gx = k["goal"][0]; b = k["b0"]
    kt = math.atan2(k["goal"][1] - b[1], gx - b[0])
    k["actual"] = kt + math.radians(k["total"])
    k["dev"] = wrap(k["actual"] - rule_dir(k, "deep0.3"))
print(f"near-post map-aimed kicks with a shot: {len(nr)}")
for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 181), (0, 181)):
    sel = [k for k in nr if lo <= k["oblique_deg"] < hi]
    if not sel: continue
    line = f"  oblique {lo:>3}-{hi:<3} n {len(sel):3d} | clean half-window median {np.median([window_deg(k) for k in sel]):4.1f} deg |"
    for rule in ("deep0.3", "centre", "bisect"):
        ys_i = [cross_y(k["b0"], rule_dir(k, rule), k["goal"][0]) for k in sel]
        ys_a = [cross_y(k["b0"], rule_dir(k, rule) + k["dev"], k["goal"][0]) for k in sel]
        ci = np.mean([y is not None and abs(y) < CLEAN for y in ys_i])
        oa = np.mean([y is not None and abs(y) < MOUTH for y in ys_a])
        ca = np.mean([y is not None and abs(y) < CLEAN for y in ys_a])
        line += f" {rule}: intended clean {100*ci:3.0f}%, w/ actual dev on-target {100*oa:3.0f}% clean {100*ca:3.0f}% |"
    print(line)
devs = np.degrees([k["dev"] for k in nr])
print(f"  deviation from intended, near-post map kicks: median {np.median(devs):+.1f}, MAD*1.48 {1.48*np.median(np.abs(devs-np.median(devs))):.1f},"
      f" |dev|>30 deg {100*np.mean(np.abs(devs)>30):.0f}% (misfires)")
for foot in ("right", "left"):
    d = np.degrees([k["dev"] for k in nr if k["foot"] == foot])
    print(f"  {foot}: n {len(d)} dev MAD*1.48 {1.48*np.median(np.abs(d-np.median(d))):.1f} deg, misfire {100*np.mean(np.abs(d)>30):.0f}%")
mis = [k for k in nr if abs(math.degrees(k["dev"])) > 30]
print(f"  misfires: n {len(mis)}, duck<0.25 m from a post {100*np.mean([k['duck_post_m']<0.25 for k in mis]):.0f}%, "
      f"post contact {100*np.mean([k['post_hit'] for k in mis]):.0f}%, oblique median {np.median([k['oblique_deg'] for k in mis]):.0f}")
beh = [k for k in ks if k["behind"]]
print(f"\nball on/over the goal line: {len(beh)} kicks; |y| from the goal axis median {1000*np.median([abs(k['b0'][1]) for k in beh]):.0f} mm, "
      f"over the line by median {1000*np.median([abs(k['b0'][0])-2 for k in beh]):.0f} mm; camera-aimed {100*np.mean([not k['map'] for k in beh]):.0f}%")
# how many goals the near-post zone could plausibly add: kicks per run
print(f"near-post map kicks per 240 s run: {len(nr)/120:.2f}; behind-line kicks per run: {len(beh)/120:.2f}")
