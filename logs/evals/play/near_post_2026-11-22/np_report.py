"""Near-post aim fix (NP1-NP3) report: goals, near-post shot accuracy, and what the setup taps did.
Reuses the accuracy-split kick parser (accuracy_split_2026-11-20/analyze.py): ground truth for scoring only."""
import json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "accuracy_split_2026-11-20"))
os.environ["SPLIT_ROOT"] = os.path.join(HERE, "logs")
import analyze as A                                        # noqa: E402

A.ROOT = os.path.join(HERE, "logs")
FINAL = os.path.join(HERE, "..", "field4_2026-11-18", "final")


def summary_numbers(txt):
    s = open(txt).read()
    g = re.search(r"goals \(attacked end\): total (\d+).*?runs with >= 1 goal (\d+)/(\d+)", s)
    o = re.search(r"goals in the OTHER end: (\d+)", s)
    st = re.search(r"near-post setup taps: (\d+) \(([^)]*)\).*?setup-cap restarts (\d+)", s)
    return {"goals": int(g.group(1)), "runs_goal": f"{g.group(2)}/{g.group(3)}", "own": int(o.group(1)),
            "taps": int(st.group(1)) if st else 0, "tap_why": st.group(2) if st else "", "setup_cap": int(st.group(3)) if st else 0}


def row_report(name, root):
    A.ROOT = root
    ks = [A.measure(k) for k in A.kicks_of(name)]
    shots = [k for k in ks if not k["setup"] and k["travel_m"] >= 0.10]
    taps = [k for k in ks if k["setup"]]
    for k in shots:
        k["near"] = k["post_m"] < 0.5 or k["duck_post_m"] < 0.5
    near = [k for k in shots if k["near"]]
    opn = [k for k in shots if not k["near"] and not k["post_hit"]]
    out = {"shots": len(shots), "near_shots": len(near),
           "near_on": 100 * np.mean([k["on_target"] for k in near]) if near else float("nan"),
           "open_on": 100 * np.mean([k["on_target"] for k in opn]) if opn else float("nan"),
           "near_misfire": 100 * np.mean([abs(k["exec"]) > 30 for k in near]) if near else float("nan"),
           "near_post_hit": 100 * np.mean([k["post_hit"] for k in near]) if near else float("nan"),
           "taps": len(taps)}
    bins = {}
    for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 181)):
        sel = [k for k in near if not k["behind"] and lo <= k["oblique_deg"] < hi]
        bins[f"{lo}-{hi}"] = (len(sel), 100 * np.mean([k["on_target"] for k in sel]) if sel else float("nan"))
    out["bins"] = bins
    # taps: where the ball came to rest (trace, 4 s after the swap-in): shootable = in front of the line by
    # > 0.12 m and <= 50 deg off the attacked goal's axis
    good = []
    for k in taps:
        rows, ts = A.load_trace(os.path.join(root, name, f"run{k['run']}_trace.csv"))
        r = A.at(rows, ts, k["t"] + 4.0)
        g = np.array(k["goal"]); sg = np.sign(g[0])
        along = (r["ball_x"] - g[0]) * sg
        obl = math.degrees(math.atan2(abs(r["ball_y"] - g[1]), -along)) if along < 0 else 180.0
        good.append(along < -0.12 and obl <= 50.0)
    out["tap_shootable"] = 100 * np.mean(good) if good else float("nan")
    return out


def main():
    rows = [("T_right_off", "near-post spawn, right, NP OFF"), ("T_right_np1", "near-post spawn, right, NP1 only"),
            ("T_right_np12", "near-post spawn, right, NP1+NP2"), ("T_right_on", "near-post spawn, right, NP1-3 ON"),
            ("T_right_on70", "right, NP1-3, NP2 at 70 deg"), ("T_right_stance", "right, NP1-3, NP2 stance gate only"),
            ("T_left_off", "near-post spawn, left, NP OFF"), ("T_left_np1", "left, NP1 only"),
            ("T_left_on", "near-post spawn, left, NP1-3 ON"), ("T_left_on70", "left, NP1-3, NP2 at 70 deg"),
            ("T_left_stance", "left, NP1-3, NP2 stance gate only")]
    ship = ["B_right_odom", "B_left_odom", "B_right_mapd", "B_left_mapd"]
    print("== targeted near-post batteries (30 x 240 s, every ball spawned near the attacked goal's posts) ==")
    print(f"{'row':<15}{'goals':>6}{'runs':>7}{'own':>5}{'taps':>6}{'cap':>5} | {'shots':>5}{'near':>5}{'near on%':>9}"
          f"{'misfire%':>9}{'post%':>6}{'open on%':>9}{'tap ok%':>8} | near on% by off-axis 0-30 / 30-50 / 50-70 / 70+")
    for name, lab in rows:
        p = os.path.join(HERE, name + ".txt")
        if not os.path.exists(p):
            continue
        s, r = summary_numbers(p), row_report(name, os.path.join(HERE, "logs"))
        b = " / ".join(f"{v[1]:3.0f}% (n{v[0]})" for v in r["bins"].values())
        print(f"{name:<15}{s['goals']:>6}{s['runs_goal']:>7}{s['own']:>5}{s['taps']:>6}{s['setup_cap']:>5} | {r['shots']:>5}"
              f"{r['near_shots']:>5}{r['near_on']:>9.1f}{r['near_misfire']:>9.1f}{r['near_post_hit']:>6.0f}{r['open_on']:>9.1f}"
              f"{r['tap_shootable']:>8.0f} | {b}   ({lab})")
    print("\n== shipping rows (30 x 240 s, normal spawns): NP ON vs the closing-work FINAL (NP did not exist = OFF) ==")
    for name in ship:
        for tag, root, txt in (("FINAL (off)", os.path.join(FINAL, "logs"), os.path.join(FINAL, name + ".txt")),
                               ("NP1-3 @50", os.path.join(HERE, "logs"), os.path.join(HERE, name + ".txt")),
                               ("NP1 only", os.path.join(HERE, "logs"), os.path.join(HERE, name + "_np1.txt"))):
            if tag == "NP1 only":
                name_ = name + "_np1"
                if not os.path.exists(txt):
                    continue
                s_, r_ = summary_numbers(txt), row_report(name_, root)
                print(f"  {name:<13} {tag:<12} goals {s_['goals']:>3}  runs {s_['runs_goal']:>5}  own {s_['own']}  taps {s_['taps']:>3}"
                      f"  near shots {r_['near_shots']:>3} on {r_['near_on']:5.1f}%  misfire {r_['near_misfire']:4.1f}%"
                      f"  open on {r_['open_on']:5.1f}%")
                continue
            if not os.path.exists(txt):
                continue
            s, r = summary_numbers(txt), row_report(name, root)
            print(f"  {name:<13} {tag:<12} goals {s['goals']:>3}  runs {s['runs_goal']:>5}  own {s['own']}  taps {s['taps']:>3}"
                  f"  near shots {r['near_shots']:>3} on {r['near_on']:5.1f}%  misfire {r['near_misfire']:4.1f}%"
                  f"  open on {r['open_on']:5.1f}%  tap->shootable {r['tap_shootable']:4.0f}%")


if __name__ == "__main__":
    main()
