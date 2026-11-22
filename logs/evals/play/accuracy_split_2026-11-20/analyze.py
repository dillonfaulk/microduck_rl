"""Accuracy split (brief: microduck-accuracy-split-measurement-brief.md, 2026-11-20).

Measurement only: reads the closing-work FINAL battery on the 4 m field (logs/evals/play/field4_2026-11-18/final,
60 s fix + pre-kick look ON, --estimated-only). No sim is run.

One record per kick (every KICK_SWAP_IN at a goal, retrieval kicks excluded). MISSION_DONE is NOT the source:
a goal whistle ends the mission before MISSION_DONE, so it would drop the kicks that scored.

Per kick, all from ground truth in the 10 Hz trace (scoring/analysis side only):
  total_err   true ball travel direction - true bearing ball -> goal centre            (what scores)
  plan_err    (kick heading - est yaw) - (true bearing - true yaw)  = the aim line as the duck executes it
              (same definition as kick_mission's aim_err_deg; validated against it below)
  exec_err    total_err - plan_err  = where the ball went relative to the duck's body vs where it meant to
              (alignment + the kick policy's scatter)
  yaw_err     est yaw - true yaw at swap-in (the pose heading error)
  lat_mm      true ball lateral in the duck's true trunk frame at swap-in, de-meaned per foot
              (proxy for true_align_ey_mm; correlation with it reported below)
  on_target   the true ball line crosses the goal line inside the mouth (|y| < 105 mm)
Sign: + = left (CCW).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/microduck_rl/logs/evals/play/field4_2026-11-18/final/logs")
ROWS = ["B_right_odom", "B_left_odom", "B_right_mapd", "B_left_mapd"]       # the shipping configuration
CONTRAST = ["A_right_odom_no_timer_prekick", "A_right_odom_no_corners"]   # labelled contrast only
POSTS = [(gx, sy * 0.115) for gx in (2.0, -2.0) for sy in (1, -1)]
MOUTH = 0.105


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_trace(p):
    with open(p) as f:
        r = csv.DictReader(f)
        rows = [{k: (float(v) if k not in ("sup", "mission", "localized") else v) for k, v in d.items()} for d in r]
    return rows, np.array([d["t"] for d in rows])


def at(rows, ts, t):
    i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
    if i > 0 and abs(ts[i - 1] - t) < abs(ts[i] - t):
        i -= 1
    return rows[i]


def kicks_of(row):
    out = []
    for ev_p in sorted(glob.glob(f"{ROOT}/{row}/run*_events.jsonl")):
        run = int(os.path.basename(ev_p)[3:].split("_")[0])
        rows, ts = load_trace(ev_p.replace("_events.jsonl", "_trace.csv"))
        evs = [json.loads(l) for l in open(ev_p)]
        foot = next((e["foot"] for e in evs if e["event"] == "MISSION_START"), None)
        last_fix, last_yawfix, goal, kh, ball_out, map_mode, aim_md = None, None, None, None, False, False, None
        pending = None
        for e in evs:
            k, t = e["event"], e["t"]
            if k == "ROUTINE_FIX":
                last_fix = t
                if e.get("yaw_fixed"):
                    last_yawfix = t
            elif k == "GOAL_PLACED":
                goal = e["centre"]
            elif k == "MISSION_START":
                kh, map_mode = e["kick_heading_deg"], False
            elif k == "PRE_KICK_LOOK_START":
                map_mode = True            # the pre-kick look runs only before map-aimed kicks (and is ON here)
            elif k == "HEADING_FROM_GOAL":
                kh = e["kick_heading_deg"]
            # out state: the out call, then WHISTLE out_window OPENS the retrieval window (a kick in it is a
            # retrieval kick toward the centre spot); retrieved / out_respawn / any other whistle closes it.
            # (2026-11-22 fix: the first version closed it on out_window and counted retrieval kicks as shots.)
            elif k == "out" or (k == "WHISTLE" and e.get("why") == "out_window"):
                ball_out = True
            elif k in ("retrieved", "out_respawn") or k == "WHISTLE":
                ball_out = False
            elif k == "MISSION_DONE" and pending is not None:
                pending["md"] = e
                pending = None
            elif k == "KICK_SWAP_IN":
                if ball_out or goal is None or kh is None:
                    pending = None
                    continue
                pending = {"row": row, "run": run, "foot": foot, "t": t, "goal": goal, "kh": math.radians(kh),
                           "map": map_mode, "t_fix": None if last_fix is None else t - last_fix,
                           "t_yawfix": None if last_yawfix is None else t - last_yawfix,
                           "rows": (rows, ts), "md": None}
                out.append(pending)
    return out


def measure(k):
    rows, ts = k.pop("rows")
    s = at(rows, ts, k["t"])
    b0 = np.array([s["ball_x"], s["ball_y"]])
    # true travel: first sample >= 0.10 m from b0 (the launch, before most deflections), else the farthest in 3 s.
    # post_hit: the ball centre comes within contact distance (+10 mm) of any goal or corner post in those 3 s.
    d, best, k["post_hit"] = None, None, False
    for t in np.arange(k["t"] + 0.1, k["t"] + 3.0, 0.1):
        r = at(rows, ts, t)
        p = np.array([r["ball_x"], r["ball_y"]])
        if min(math.hypot(px - p[0], py - p[1]) for px, py in POSTS) < 0.035 + 0.010 + 0.010 or \
                min(math.hypot(cx - p[0], cy - p[1]) for cx in (2, -2) for cy in (2, -2)) < 0.035 + 0.0375 + 0.010:
            k["post_hit"] = True
        v = p - b0
        n = float(np.linalg.norm(v))
        if best is None or n > best[0]:
            best = (n, v)
        if d is None and n >= 0.10:
            d = v
    if d is None:
        d = best[1]
    k["travel_m"] = round(float(np.linalg.norm(d)), 3)
    g = np.array(k["goal"])
    kt = math.atan2(g[1] - b0[1], g[0] - b0[0])
    k["total"] = math.degrees(wrap(math.atan2(d[1], d[0]) - kt))
    k["plan"] = math.degrees(wrap((k["kh"] - s["est_yaw"]) - (kt - s["yaw"])))
    k["exec"] = k["total"] - k["plan"]
    k["yaw_err"] = math.degrees(wrap(s["est_yaw"] - s["yaw"]))
    # true lateral of the ball in the true trunk frame (left +)
    c, sn = math.cos(s["yaw"]), math.sin(s["yaw"])
    rel = b0 - np.array([s["x"], s["y"]])
    k["lat_raw_mm"] = 1000 * (-sn * rel[0] + c * rel[1])
    # on target: the true line from b0 to the goal line x = gx
    dirn = d / max(np.linalg.norm(d), 1e-9)
    if abs(dirn[0]) > 1e-6 and (g[0] - b0[0]) / dirn[0] > 0:
        y_at = b0[1] + dirn[1] * (g[0] - b0[0]) / dirn[0]
        k["on_target"] = abs(y_at - g[1]) < MOUTH
    else:
        k["on_target"] = False
    k["b0"] = [float(b0[0]), float(b0[1])]
    k["behind"] = bool(b0[0] * np.sign(g[0]) > abs(g[0]) - 0.035)    # on/over the goal line: no direct shot
    k["ball_goal_m"] = float(np.linalg.norm(g - b0))
    k["half_tol_deg"] = math.degrees(math.atan2(MOUTH, max(k["ball_goal_m"], 1e-3)))
    pd = [math.hypot(px - b0[0], py - b0[1]) for px, py in POSTS if px * g[0] > 0]
    k["post_m"] = min(pd)
    k["oblique_deg"] = abs(math.degrees(wrap(math.atan2(b0[1] - g[1], (b0[0] - g[0]) * -np.sign(g[0]))) ))
    k["duck_post_m"] = min(math.hypot(px - s["x"], py - s["y"]) for px, py in POSTS)
    return k


def stats(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "   -"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{len(v):4d}  {med:+6.2f}  {q3 - q1:5.2f}  {np.mean(np.abs(v)):5.2f}  {np.percentile(np.abs(v), 90):5.2f}"


HDR = "   n   median   IQR   mean|e|  p90|e|"


def table(title, ks, key, bins, labels, fields=("total", "plan", "exec")):
    print(f"\n### {title}")
    for f in fields:
        print(f"  [{f}_err, deg]{'':<14}{HDR}   on-target")
        for (lo, hi), lab in zip(bins, labels):
            sel = [k for k in ks if k.get(key) is not None and lo <= k[key] < hi]
            ot = f"{100 * np.mean([k['on_target'] for k in sel]):5.1f}%" if sel else ""
            print(f"    {lab:<24}{stats([k[f] for k in sel])}   {ot if f == 'total' else ''}")


def trend(ks, key, f="total"):
    x = np.array([k[key] for k in ks if k.get(key) is not None], float)
    y = np.array([abs(k[f]) for k in ks if k.get(key) is not None], float)
    if len(x) < 10:
        return "n/a"
    sl, ic = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    # bootstrap the slope
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"|{f}| slope {sl:+.4f}/unit (95% CI {lo:+.4f}..{hi:+.4f}), r {r:+.2f}, n {len(x)}"


def main():
    allk = {}
    for row in ROWS + CONTRAST:
        allk[row] = [measure(k) for k in kicks_of(row)]
    ks = [k for r in ROWS for k in allk[r]]
    weak = [k for k in ks if k["travel_m"] < 0.10]
    ks = [k for k in ks if k["travel_m"] >= 0.10]
    for f in ("right", "left"):
        m = np.mean([k["lat_raw_mm"] for k in ks if k["foot"] == f])
        for k in ks:
            if k["foot"] == f:
                k["lat_mm"] = k["lat_raw_mm"] - m
    print(f"kicks at a goal (shipping rows {', '.join(ROWS)}): {len(ks)} kept, {len(weak)} dropped (ball moved < 0.10 m)")

    # --- validation against the mission's own logged numbers (kicks that reached MISSION_DONE)
    v = [k for k in ks if k["md"] and "aim_err_deg" in k["md"]]
    dp = np.array([k["plan"] - k["md"]["aim_err_deg"] for k in v])
    print(f"validation: plan_err vs logged aim_err_deg on {len(v)} kicks: median diff {np.median(dp):+.2f} deg, "
          f"p95 |diff| {np.percentile(np.abs(dp), 95):.2f}")
    vm = [k for k in v if "aim_mode" in k["md"]]
    agree = np.mean([(k["md"]["aim_mode"] == "map") == k["map"] for k in vm])
    print(f"validation: map/camera classification agrees with logged aim_mode on {100 * agree:.1f}% of {len(vm)}")
    for f in ("right", "left"):
        vl = [k for k in v if k["foot"] == f and "true_align_ey_mm" in k["md"]]
        if vl:
            r = np.corrcoef([k["lat_mm"] for k in vl], [k["md"]["true_align_ey_mm"] for k in vl])[0, 1]
            print(f"validation: lat_mm vs logged true_align_ey_mm ({f}): r {r:+.2f}, n {len(vl)}")
    nm = sum(1 for k in ks if k["md"] is None)
    print(f"kicks with no MISSION_DONE (mission cut by a whistle, mostly goals): {nm}  "
          f"(on target {100 * np.mean([k['on_target'] for k in ks if k['md'] is None]):.0f}%)")

    print("\nComponents: total = plan + exec. plan = the aim line the duck executes (pose heading for map aim, "
          "perception for camera aim); exec = ball vs the duck's body (alignment + kick).")
    print(f"  overall{'':<17}{HDR}")
    for f in ("total", "plan", "exec", "yaw_err"):
        print(f"    {f:<22}{stats([k[f] for k in ks])}")

    # 1. elapsed time / time since fix
    tb = [(0, 60), (60, 120), (120, 180), (180, 241)]
    table("1a. elapsed time in run at kick (s)", ks, "t", tb, ["0-60", "60-120", "120-180", "180-240"],
          ("total", "plan", "exec", "yaw_err"))
    fb = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 1e9)]
    fl = ["0-10", "10-20", "20-40", "40-60", ">60"]
    table("1b. time since last landed landmark fix (s)", ks, "t_fix", fb, fl, ("total", "plan", "yaw_err"))
    nofix = [k for k in ks if k["t_fix"] is None]
    print(f"    (no landed fix yet: {len(nofix)} kicks; |total| mean "
          f"{np.mean([abs(k['total']) for k in nofix]) if nofix else float('nan'):.2f})")
    table("1c. time since last yaw-correcting fix (s)", ks, "t_yawfix", fb, fl, ("total", "plan", "yaw_err"))
    for key in ("t", "t_fix"):
        print(f"  trend vs {key}: {trend(ks, key, 'total')};  {trend(ks, key, 'plan')}")
    for side, sel in (("camera", [k for k in ks if not k["map"]]), ("map", [k for k in ks if k["map"]])):
        print(f"  trend vs t_fix, {side}-aimed only: {trend(sel, 't_fix', 'plan')}")

    # contrast: no 60 s fix / no pre-kick look (labelled, not the shipping config)
    ck = [measure(k) if "total" not in k else k for k in allk["A_right_odom_no_timer_prekick"]]
    ck = [k for k in ck if k["travel_m"] >= 0.10]
    table("1d. CONTRAST (not shipping): right odom, NO 60 s fix, NO pre-kick look - elapsed time", ck, "t", tb,
          ["0-60", "60-120", "120-180", "180-240"], ("total", "plan", "yaw_err"))
    print(f"  trend vs t (contrast): {trend(ck, 't', 'total')};  {trend(ck, 't', 'yaw_err')}")

    # 2. post proximity
    pb = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 9)]
    pl = ["ball <0.25 m", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">2.0"]
    table("2a. ball distance to the nearest post of the attacked goal", ks, "post_m", pb, pl)
    ob = [(0, 15), (15, 30), (30, 50), (50, 70), (70, 181)]
    table("2b. ball angle off the attacked goal's axis (deg; >50 = map-aimed edge-on)", ks, "oblique_deg", ob,
          ["0-15", "15-30", "30-50", "50-70", ">70"])
    near = [k for k in ks if k["post_m"] < 0.5]
    print("  near-post (<0.5 m) kicks, how far off is 'off': |total| / goal half-angle from the ball:")
    for lo, hi in ((0, 0.25), (0.25, 0.5)):
        sel = [k for k in near if lo <= k["post_m"] < hi]
        if sel:
            print(f"    {lo}-{hi} m: n {len(sel)}, half-angle median {np.median([k['half_tol_deg'] for k in sel]):.1f} deg,"
                  f" on-target {100 * np.mean([k['on_target'] for k in sel]):.0f}%, map-aimed "
                  f"{100 * np.mean([k['map'] for k in sel]):.0f}%, oblique median "
                  f"{np.median([k['oblique_deg'] for k in sel]):.0f} deg")
    table("2c. duck (trunk) distance to the nearest goal post", ks, "duck_post_m", pb,
          ["duck <0.25 m", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">2.0"])

    # 3. foot x aim mode
    print("\n### 3. foot x aim mode")
    for f in ("total", "plan", "exec"):
        print(f"  [{f}_err, deg]{'':<14}{HDR}   on-target")
        for foot in ("right", "left"):
            for mm, lab in ((False, "camera"), (True, "map")):
                sel = [k for k in ks if k["foot"] == foot and k["map"] == mm]
                ot = f"{100 * np.mean([k['on_target'] for k in sel]):5.1f}%" if sel else ""
                print(f"    {foot + ' ' + lab:<24}{stats([k[f] for k in sel])}   {ot if f == 'total' else ''}")

    # 4. true lateral placement
    lb = [(-1e9, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 1e9)]
    ll = ["< -4 mm", "-4..-2", "-2..0", "0..2", "2..4", "> 4 mm"]
    for foot in ("right", "left"):
        sel = [k for k in ks if k["foot"] == foot]
        table(f"4. true lateral placement at contact, {foot} foot (de-meaned, mm)", sel, "lat_mm", lb, ll, ("exec", "total"))
        x = np.array([k["lat_mm"] for k in sel])
        y = np.array([k["exec"] for k in sel])
        sl, ic = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        print(f"  {foot}: exec_err = {sl:+.2f} deg/mm x lat + {ic:+.2f}  (R^2 {r2:.2f}, lateral std {x.std():.1f} mm, n {len(x)})")

    # ---------------- pass 2: stratified ----------------
    for k in ks:
        k["near"] = k["post_m"] < 0.5 or k["duck_post_m"] < 0.5
    opn = [k for k in ks if not k["near"] and not k["post_hit"]]
    nr = [k for k in ks if k["near"]]
    print("\n================ PASS 2: stratified ================")
    print(f"near-post (ball or duck < 0.5 m from a goal post): {len(nr)} kicks; open field (neither, no post contact): "
          f"{len(opn)}; open-field kicks that still touched a post: {sum(1 for k in ks if not k['near'] and k['post_hit'])}")
    print("\n### near-post share by elapsed time (the confound check for the time-decay read)")
    for (lo, hi), lab in zip(tb, ["0-60", "60-120", "120-180", "180-240"]):
        sel = [k for k in ks if lo <= k["t"] < hi]
        print(f"    {lab:<10} n {len(sel):4d}  near-post {100 * np.mean([k['near'] for k in sel]):5.1f}%  "
              f"map-aimed {100 * np.mean([k['map'] for k in sel]):5.1f}%  first-kick-of-run-ball "
              f"{100 * np.mean([k['post_m'] > 2.0 for k in sel]):5.1f}% (ball >2 m from posts)")
    table("P2-1a. OPEN FIELD only: elapsed time", opn, "t", tb, ["0-60", "60-120", "120-180", "180-240"],
          ("total", "plan", "exec", "yaw_err"))
    table("P2-1b. OPEN FIELD only: time since last landed fix", opn, "t_fix", fb, fl, ("total", "plan", "yaw_err"))
    for key in ("t", "t_fix"):
        print(f"  open-field trend vs {key}: {trend(opn, key, 'total')}")
        print(f"  open-field trend vs {key}: {trend(opn, key, 'plan')}")
        print(f"  open-field trend vs {key}: {trend(opn, key, 'yaw_err')}")
    for side, sel in (("camera", [k for k in opn if not k["map"]]), ("map", [k for k in opn if k["map"]])):
        print(f"  open-field trend vs t_fix, {side}-aimed: {trend(sel, 't_fix', 'plan')}")
    nearT = [k for k in ks if k["near"]]
    for (lo, hi), lab in zip(tb, ["0-60", "60-120", "120-180", "180-240"]):
        sel = [k for k in nearT if lo <= k["t"] < hi]
        if sel:
            print(f"    near-post by time {lab:<8} n {len(sel):3d} on-target {100 * np.mean([k['on_target'] for k in sel]):5.1f}%")

    print("\n### P2-2. near-post outcomes (ball or duck < 0.5 m from a goal post)")
    cats = defaultdict(list)
    for k in nr:
        if k["behind"]:
            c = "ball on/over the goal line (no direct shot)"
        elif k["post_hit"] and k["on_target"]:
            c = "on target, touched a post"
        elif k["post_hit"]:
            c = "hit a post (wide line)"
        elif k["on_target"]:
            c = "on target, clean"
        else:
            c = "wide, no post contact"
        cats[c].append(k)
    for c, sel in sorted(cats.items(), key=lambda x: -len(x[1])):
        print(f"    {c:<46} {len(sel):4d} ({100 * len(sel) / len(nr):4.1f}%)  map-aimed {100 * np.mean([k['map'] for k in sel]):3.0f}%"
              f"  oblique median {np.median([k['oblique_deg'] for k in sel]):4.0f} deg  |plan| median {np.median([abs(k['plan']) for k in sel]):5.1f}"
              f"  |exec| median {np.median([abs(k['exec']) for k in sel]):5.1f}")
    for lab, sel in (("near-post", nr), ("open field", opn)):
        for mm in (False, True):
            s2 = [k for k in sel if k["map"] == mm and not k["behind"]]
            if s2:
                print(f"    {lab:<11} {'map' if mm else 'camera':<7} n {len(s2):4d} on-target "
                      f"{100 * np.mean([k['on_target'] for k in s2]):5.1f}%  |plan| median {np.median([abs(k['plan']) for k in s2]):5.2f}"
                      f"  |exec| median {np.median([abs(k['exec']) for k in s2]):5.2f}  yaw_err |median| {np.median([abs(k['yaw_err']) for k in s2]):4.2f}")
    # near-post map-aim: how wide the mouth is from there, vs the errors
    print("  near-post, not behind the line: oblique bins")
    for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 181)):
        s2 = [k for k in nr if not k["behind"] and lo <= k["oblique_deg"] < hi]
        if s2:
            print(f"    oblique {lo:>3}-{hi:<3} n {len(s2):3d} on-target {100 * np.mean([k['on_target'] for k in s2]):5.1f}%"
                  f"  post hit {100 * np.mean([k['post_hit'] for k in s2]):4.0f}%  map {100 * np.mean([k['map'] for k in s2]):3.0f}%"
                  f"  |total| median {np.median([abs(k['total']) for k in s2]):5.1f}  mouth half-angle median "
                  f"{np.median([k['half_tol_deg'] for k in s2]):4.1f} (head-on; edge-on it is smaller)")

    print("\n### P2-3. OPEN FIELD foot x aim mode")
    for f in ("total", "plan", "exec"):
        print(f"  [{f}_err, deg]{'':<14}{HDR}   on-target")
        for foot in ("right", "left"):
            for mm, lab in ((False, "camera"), (True, "map")):
                sel = [k for k in opn if k["foot"] == foot and k["map"] == mm]
                ot = f"{100 * np.mean([k['on_target'] for k in sel]):5.1f}%" if sel else ""
                print(f"    {foot + ' ' + lab:<24}{stats([k[f] for k in sel])}   {ot if f == 'total' else ''}")

    print("\n### P2-4. OPEN FIELD lateral placement vs exec_err (robust: |exec| < 30 deg)")
    for foot in ("right", "left"):
        sel = [k for k in opn if k["foot"] == foot and abs(k["exec"]) < 30]
        table(f"P2-4 {foot} foot", sel, "lat_mm", lb, ll, ("exec",))
        x = np.array([k["lat_mm"] for k in sel]); y = np.array([k["exec"] for k in sel])
        sl, ic = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
        res = y - (sl * x + ic)
        print(f"  {foot}: exec = {sl:+.2f} deg/mm x lat {ic:+.2f}  R^2 {r2:.2f}  lateral std {x.std():.1f} mm  "
              f"exec std {y.std():.2f} -> {res.std():.2f} deg if lateral were zero  (n {len(x)}, dropped |exec|>=30: "
              f"{sum(1 for k in opn if k['foot'] == foot and abs(k['exec']) >= 30)})")
        tol = np.median([k["half_tol_deg"] for k in sel])
        print(f"  {foot}: open-field goal half-angle median {tol:.1f} deg (ball {np.median([k['ball_goal_m'] for k in sel]):.2f} m out)")

    print("\n### P2-5. what the open-field misses are made of (on-target fails, open field, not behind)")
    for foot in ("right", "left"):
        sel = [k for k in opn if k["foot"] == foot]
        P = np.array([k["plan"] for k in sel]); E = np.array([k["exec"] for k in sel]); T = np.array([k["total"] for k in sel])
        Y = np.array([k["yaw_err"] for k in sel])
        print(f"  {foot}: n {len(sel)}  std total {T.std():.2f}  plan {P.std():.2f}  exec {E.std():.2f}  yaw_err {Y.std():.2f} | "
              f"robust (MAD*1.48): total {1.48 * np.median(np.abs(T - np.median(T))):.2f}  plan {1.48 * np.median(np.abs(P - np.median(P))):.2f}"
              f"  exec {1.48 * np.median(np.abs(E - np.median(E))):.2f}  yaw {1.48 * np.median(np.abs(Y - np.median(Y))):.2f}")

    print("\n### P2-6. near-post map aim geometry: where the INTENDED line (true ball -> the 0.3 m-deep aim point) crosses"
          " the goal line, i.e. with perfect execution")
    for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 181)):
        s2 = [k for k in nr if not k["behind"] and k["map"] and lo <= k["oblique_deg"] < hi]
        ys = []
        for k in s2:
            gx = k["goal"][0]; sg = np.sign(gx); bx, by = k["b0"]
            ax = gx + sg * 0.3
            ys.append(abs(by + (0 - by) * (gx - bx) / (ax - bx)))
        ys = np.array(ys)
        if len(ys):
            print(f"    oblique {lo:>3}-{hi:<3} n {len(ys):3d}  |y at line| median {1000 * np.median(ys):4.0f} mm  "
                  f"clean (<70 mm) {100 * np.mean(ys < 0.070):4.0f}%  inside posts (<105 mm) {100 * np.mean(ys < 0.105):4.0f}%  "
                  f"actual on-target {100 * np.mean([k['on_target'] for k in s2]):4.0f}%")
    beh = [k for k in nr if k["behind"]]
    print(f"  balls on/over the goal line: {len(beh)} kicks ({100 * len(beh) / len(ks):.1f}% of all kicks), "
          f"camera-aimed {100 * np.mean([not k['map'] for k in beh]):.0f}%")

    # variance shares: how much of total spread each component carries, per foot
    print("\n### variance of total_err carried by each component (var(total) = var(plan) + var(exec) + 2cov)")
    for foot in ("right", "left"):
        sel = [k for k in ks if k["foot"] == foot]
        T = np.array([k["total"] for k in sel]); P = np.array([k["plan"] for k in sel]); E = np.array([k["exec"] for k in sel])
        print(f"  {foot}: std total {T.std():.2f}, plan {P.std():.2f}, exec {E.std():.2f}, cov {np.cov(P, E)[0, 1]:+.2f}")

    with open(os.path.join(os.path.dirname(__file__), "kicks.jsonl"), "w") as f:
        for k in ks:
            f.write(json.dumps({x: (bool(v) if isinstance(v, (np.bool_,)) else v) for x, v in k.items()
                                if x not in ("md",)}, default=float) + "\n")


if __name__ == "__main__":
    main()
