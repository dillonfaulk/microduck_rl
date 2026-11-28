"""OmniHop shared regression (2026-11-27): the rows a swap-in hop policy can affect, CPU MuJoCo + BAM.

Run from the microduck_rl repo root:
  uv run python scripts/hop_regression.py <omnihop.onnx> [--episodes 20] [--workers 12] [--out DIR]

Rows (every row: VelStand 4250 prestand 1.5 s -> hop ONNX swap-in, immediate request = runtime default,
drift model ON, 5 directions as in hop_battery):
  swapback  hop 2.5 s, then swap BACK to VelStand (the runtime contract): stand 1 s, walk vx 0.2 m/s 3 s,
            in-place turn wz 0.8 rad/s 2 s, stop 1 s. Reports falls, leg deviation at the swap-back,
            achieved walk vx / turn wz vs a no-hop VelStand control.
  push      a 0.3 m/s horizontal shove (random direction) at a random time in [0, 0.7] s after the swap (crouch / push-off / flight; the hop is over by ~0.7 s).
            Fall (tilt > 45 deg or z < 0.06) -> the supervisor swaps VelStand in -> got up within 8 s?
  friction  BAM's velocity-independent friction budget x1.5 (the sim-to-real stiction row): feet at the
            longest flight's touchdown, falls.
Peak trunk spin is reported for every row against the 10.7 rad/s bar.
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hop_battery as hb  # noqa: E402

DIRS = {"in_place": (0.0, 0.0), "fwd": (0.03, 0.0), "back": (-0.03, 0.0), "left": (0.0, 0.02), "right": (0.0, -0.02)}
FALL_TILT, FALL_Z, UP_TILT, UP_Z = 45.0, 0.06, 25.0, 0.09
_G = {}


def _sim(onnx, friction):
    key = (onnx, friction)
    if _G.get("key") != key:
        hs = hb.HopSim(onnx, hb.DEFAULT_VELSTAND)
        if friction != 1.0:
            bm = hs.sim.bam.model
            orig = bm.compute_frictions

            def scaled(*a, **k):
                fl, damp = orig(*a, **k)
                return fl * friction, damp
            bm.compute_frictions = scaled
        _G.update(key=key, hs=hs)
    return _G["hs"]


def _start(hs, rng):
    sim = hs.sim
    sim.reset(0.0, 0.0, rng.uniform(-math.pi, math.pi))
    hs.gyro_bias[:] = 0.0
    sim.policy.ort_session = hs.prestand
    sim.set_twist((0.0, 0.0, 0.0))
    while sim.t < hb.PRESTAND_S - 1e-9:
        sim.step()
    k = float(np.clip(rng.normal(-0.019, 0.006), -0.05, 0.02))
    hs.gyro_bias[:] = rng.uniform(-0.03, 0.03, 3)
    return k


def _hop(hs, target, k, secs, push=None, rng=None):
    """Run the hop policy for secs; returns dict (fell_t, spin_peak, td_feet, leg_dev at end)."""
    sim = hs.sim
    (_, _, yaw0), _, _ = sim.pose()
    x0, y0 = hs.feet_mid()
    c, s = math.cos(yaw0), math.sin(yaw0)
    goal = (x0 + c * target[0] - s * target[1], y0 + s * target[0] + c * target[1])
    sim.policy.ort_session = hs.hop_session
    sim.t = 0.0
    air = best = 0
    td, fell_t, spin = None, None, 0.0
    while sim.t < secs - 1e-9:
        (_, _, yaw), _, _ = sim.pose()
        fx, fy = hs.feet_mid()
        tx, ty = hb.to_frame(goal[0] - x0, goal[1] - y0, yaw)
        dx, dy = hb.to_frame(fx - x0, fy - y0, yaw)
        hs.set_cmd((tx - (1 + k) * dx, ty - (1 + k) * dy), sim.t < hb.PULSE_S)
        if push is not None and push[0] <= sim.t < push[0] + 0.02 - 1e-9:
            sim.push(push[1] * math.cos(push[2]), push[1] * math.sin(push[2]))
        sim.step()
        ld, rd, _ = hs.contacts()
        if not (ld or rd):
            air += 1
        else:
            if air > 0 and air >= best:
                best = air
                fx, fy = hs.feet_mid()
                td = [round(1000 * v, 1) for v in hb.to_frame(fx - x0, fy - y0, yaw0)]
            air = 0
        spin = max(spin, float(np.linalg.norm(sim.data.qvel[sim.va + 3:sim.va + 6])))
        _, tilt, z = sim.pose()
        if fell_t is None and (tilt > FALL_TILT or z < FALL_Z):
            fell_t = round(sim.t, 2)
            break
    q = np.array([sim.data.qpos[j] for j in sim.policy.joint_qpos_indices]) - sim.policy.default_pose
    return {"fell_t": fell_t, "spin_peak": round(spin, 2), "td_feet_mm": td,
            "leg_dev_deg": round(math.degrees(float(np.abs(q[hb.LEG_JOINTS]).max())), 1)}


def _velstand(hs, schedule):
    """Swap VelStand back in and run [(secs, twist)]; returns falls, achieved per segment, spin."""
    sim = hs.sim
    sim.policy.ort_session = hs.prestand
    out, fell, spin = [], None, 0.0
    t_all = 0.0
    for secs, twist in schedule:
        sim.set_twist(twist)
        (x0, y0, yaw0), _, _ = sim.pose()
        yaws, n = [yaw0], int(round(secs / 0.02))
        for i in range(n):
            sim.step()
            t_all += 0.02
            (_, _, yaw), tilt, z = sim.pose()
            yaws.append(yaw)
            spin = max(spin, float(np.linalg.norm(sim.data.qvel[sim.va + 3:sim.va + 6])))
            if fell is None and (tilt > FALL_TILT or z < FALL_Z):
                fell = round(t_all, 2)
        (x1, y1, _), _, _ = sim.pose()
        bx, _ = hb.to_frame(x1 - x0, y1 - y0, yaw0)
        out.append({"twist": twist, "vx": round(bx / secs, 3), "wz": round(float(np.unwrap(yaws)[-1] - yaws[0]) / secs, 3)})
    return {"segments": out, "fell_t": fell, "spin_peak": round(spin, 2)}


SWAPBACK_SCHEDULE = [(1.0, (0.0, 0.0, 0.0)), (3.0, (0.2, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.8)), (1.0, (0.0, 0.0, 0.0))]


def run(job):
    onnx, row, d, ep = job
    rng = np.random.default_rng(10_000 + 97 * ep + (list(DIRS).index(d) if d in DIRS else 7))
    hs = _sim(onnx, 1.5 if row == "friction" else 1.0)
    k = _start(hs, rng)
    if row == "control":                                     # VelStand only, no hop (the swap-back baseline)
        hs.sim.t = 0.0
        vs = _velstand(hs, SWAPBACK_SCHEDULE)
        return {"row": row, "dir": "none", "ep": ep, "hop": None, "velstand": vs}
    if row == "swapback":
        h = _hop(hs, DIRS[d], k, 2.5)
        vs = _velstand(hs, SWAPBACK_SCHEDULE) if h["fell_t"] is None else None
        return {"row": row, "dir": d, "ep": ep, "hop": h, "velstand": vs}
    if row == "push":
        push = (rng.uniform(0.0, 0.7), 0.3, rng.uniform(0, 2 * math.pi))   # during crouch / push / flight
        h = _hop(hs, DIRS[d], k, 2.5, push=push)
        got_up = None
        if h["fell_t"] is not None:                          # supervisor: fall -> VelStand
            hs.sim.policy.ort_session = hs.prestand
            hs.sim.set_twist((0.0, 0.0, 0.0))
            up_run, t = 0, 0.0
            while t < 8.0:
                hs.sim.step()
                t += 0.02
                _, tilt, z = hs.sim.pose()
                up_run = up_run + 1 if (tilt < UP_TILT and z > UP_Z) else 0
                if up_run >= 50:
                    got_up = round(t, 2)
                    break
        return {"row": row, "dir": d, "ep": ep, "hop": h, "push": [round(push[0], 2), push[1]], "got_up_t": got_up}
    if row == "friction":
        h = _hop(hs, DIRS[d], k, 2.5)
        return {"row": row, "dir": d, "ep": ep, "hop": h}
    raise ValueError(row)


def summarize(rows):
    L = []
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    ctl = [r for r in rows if r["row"] == "control"]
    if ctl:
        segs = np.array([[s["vx"], s["wz"]] for r in ctl for s in r["velstand"]["segments"]]).reshape(len(ctl), -1, 2)
        L.append(f"== control (VelStand only, no hop), n={len(ctl)}: falls {sum(r['velstand']['fell_t'] is not None for r in ctl)}, "
                 f"walk vx {med(segs[:, 1, 0]):.3f} (cmd 0.2), turn wz {med(segs[:, 2, 1]):.3f} (cmd 0.8), "
                 f"spin peak med {med([r['velstand']['spin_peak'] for r in ctl]):.1f}")
    for row, title in (("swapback", "swap back to VelStand after the hop"), ("push", "0.3 m/s push during the hop"),
                       ("friction", "BAM friction x1.5")):
        rs = [r for r in rows if r["row"] == row]
        if not rs:
            continue
        L.append(f"== {row}: {title}")
        for d in DIRS:
            x = [r for r in rs if r["dir"] == d]
            hops = [r["hop"] for r in x]
            hop_falls = sum(h["fell_t"] is not None for h in hops)
            spin = [h["spin_peak"] for h in hops]
            td = [h["td_feet_mm"] for h in hops if h["td_feet_mm"]]
            tds = str([round(float(v), 1) for v in np.median(np.array(td), axis=0)]) if td else "-"
            line = (f"  {d:9s} n={len(x):<3d} hop falls {hop_falls:<3d} feet@touchdown {tds:>14s} "
                    f"spin pk med/max {med(spin):4.1f}/{max(spin):4.1f} (>10.7: {sum(s > 10.7 for s in spin)})")
            if row == "swapback":
                vs = [r["velstand"] for r in x if r["velstand"]]
                line += (f"  leg dev @swap med {med([h['leg_dev_deg'] for h in hops]):.1f} deg | VelStand falls "
                         f"{sum(v['fell_t'] is not None for v in vs)}, walk vx {med([v['segments'][1]['vx'] for v in vs]):.3f}, "
                         f"turn wz {med([v['segments'][2]['wz'] for v in vs]):.3f}, VelStand spin max "
                         f"{max(v['spin_peak'] for v in vs) if vs else float('nan'):.1f}")
            if row == "push":
                fl = [r for r in x if r["hop"]["fell_t"] is not None]
                line += f"  | fell -> VelStand got up {sum(r['got_up_t'] is not None for r in fl)}/{len(fl)}"
            L.append(line)
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    onnx = os.path.abspath(a.onnx)
    jobs = [(onnx, "control", "none", ep) for ep in range(a.episodes)]
    jobs += [(onnx, row, d, ep) for row in ("swapback", "push", "friction") for d in DIRS for ep in range(a.episodes)]
    with Pool(a.workers) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    lines = [f"OmniHop shared regression: {a.onnx}", f"  {a.episodes} episodes per row/direction; VelStand 4250 prestand "
             f"-> swap-in, immediate request, drift ON; feet-midpoint yardstick", ""] + summarize(rows)
    print("\n".join(lines))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "REGRESSION.txt"), "w").write("\n".join(lines) + "\n")
        json.dump(rows, open(os.path.join(a.out, "regression.json"), "w"), indent=1)
