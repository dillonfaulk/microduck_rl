"""HeadJamRecover head-under battery (2026-11-22). CPU MuJoCo + BAM, field scene, 15 s per state, zero command.

States: #0 = the exact dumped 42-deg stall (with its velocities and action buffer); #1..N-1 = the training spawn
family (mdp.sample_head_under_np, fixed seed 20261122, zero velocity, dropped from 0.13-0.14 m).
"Up" = tilt < 25 deg and trunk z > 0.09 m, held 1 s (warp_standup_eval's rule).
Neck audit per state: the most negative RAW neck target the policy commands, and the share of steps the clip engages.

usage: python battery.py <policy.onnx> [--clamp] [--n 200] [--secs 15] [--workers 16] [--out file.json]
"""
import argparse, json, math, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
D = "logs/evals/velstand/headjam_2026-11-22"
NECK_CLIP = (-math.pi / 2 + 0.05, math.radians(60) - 0.05)   # == standup cfg HEADJAM_NECK_CLIP
SEED = 20261122


def _states(n, lo, hi):
    from mjlab_microduck.tasks.mdp import sample_head_under_np
    return sample_head_under_np(np.random.default_rng(SEED), n - 1, lo, hi)


def run_one(args):
    onnx, clamp, idx, secs = args
    import mujoco
    from kick_battery import SCENES, KickSim
    from kick_mission import setup_goal, setup_field
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        sim = KickSim(onnx, SCENES["field"]); setup_goal(sim); setup_field(sim)
    m, d, p = sim.model, sim.data, sim.policy
    qi = list(p.joint_qpos_indices)
    jids = [list(m.jnt_qposadr).index(q) for q in qi]
    lo, hi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
    ineck = [m.joint(j).name for j in jids].index("neck_pitch")
    seen = {"min_raw": math.inf, "n": 0, "clipped": 0}
    orig = p.set_position_targets

    def spt(t):
        t = np.asarray(t, float).copy()
        seen["n"] += 1
        if seen["n"] > 5:                 # the exact stall's first 0.1 s is the dumped round-3 buffer
            seen["min_raw"] = min(seen["min_raw"], float(t[ineck]))
            seen["clipped"] += int(t[ineck] < NECK_CLIP[0] or t[ineck] > NECK_CLIP[1])
        if clamp:
            t[ineck] = min(max(t[ineck], NECK_CLIP[0]), NECK_CLIP[1])
        orig(t)
    p.set_position_targets = spt
    sim.reset(0, 0, 0)
    if idx < 0:          # stall neighbourhood: the exact stall + joint noise +-0.1 rad (probe.py's set), seed -idx
        S = np.load(f"{D}/stall_state_6m_look_run5_t60.npz")
        d.qpos[:] = S["qpos"]; d.qvel[:] = S["qvel"]
        r = np.random.default_rng(-idx - 1)
        for q in qi:
            d.qpos[q] += r.uniform(-0.1, 0.1)
        p.last_action[:] = S["last_action"]
        if p.action_buffer is not None:
            for a, b in zip(p.action_buffer, S["buf"]):
                a[:] = b
    elif idx == 0:
        S = np.load(f"{D}/stall_state_6m_look_run5_t60.npz")
        d.qpos[:] = S["qpos"]; d.qvel[:] = S["qvel"]
        p.last_action[:] = S["last_action"]
        if p.action_buffer is not None:
            for a, b in zip(p.action_buffer, S["buf"]):
                a[:] = b
    else:
        q, z, j = (x[idx - 1] for x in _states(200, lo, hi))
        bx, by = d.qpos[0], d.qpos[1]
        d.qpos[2] = z; d.qpos[3:7] = q
        for k, a in enumerate(qi):
            d.qpos[a] = j[k]
        d.qvel[:] = 0.0
        d.qpos[0], d.qpos[1] = bx, by
    p.head_offset[:] = 0.0
    sim.set_twist((0.0, 0.0, 0.0))
    mujoco.mj_forward(m, d)
    held, up_t, tilts = 0, None, []
    for k in range(int(secs / 0.02)):
        sim.step()
        tilt = sim.pose()[1]
        tilts.append(tilt)
        ok = tilt < 25.0 and d.qpos[2] > 0.09
        held = held + 1 if ok else 0
        if up_t is None and held >= 50:
            up_t = (k + 1) * 0.02 - 1.0
    return {"idx": idx, "up_s": up_t, "final_tilt": tilts[-1], "min_tilt": min(tilts),
            "min_raw_neck": seen["min_raw"], "clip_rate": seen["clipped"] / max(1, seen["n"] - 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx"); ap.add_argument("--clamp", action="store_true")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--secs", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--out")
    ap.add_argument("--nbhd", type=int, default=50, help="stall-neighbourhood states (harder, reported separately)")
    a = ap.parse_args()
    jobs = [(a.onnx, a.clamp, i, a.secs) for i in range(a.n)] + [(a.onnx, a.clamp, -k, a.secs) for k in range(1, a.nbhd + 1)]
    with Pool(a.workers) as pool:
        allres = pool.map(run_one, jobs, chunksize=1)
    res, nb = [r for r in allres if r["idx"] >= 0], [r for r in allres if r["idx"] < 0]
    up = [r for r in res if r["up_s"] is not None]
    t = [r["up_s"] for r in up]
    stuck = [r for r in res if r["up_s"] is None]
    summ = {"onnx": a.onnx, "clamp": a.clamp, "n": a.n, "up": len(up), "up_pct": 100 * len(up) / a.n,
            "exact_stall_up_s": res[0]["up_s"], "t_med": float(np.median(t)) if t else None,
            "t_p90": float(np.quantile(t, 0.9)) if t else None,
            "raw_neck_min": min(r["min_raw_neck"] for r in res),
            "clip_rate_mean": float(np.mean([r["clip_rate"] for r in res])),
            "stuck_idx": [r["idx"] for r in stuck], "stuck_final_tilt": [round(r["final_tilt"], 1) for r in stuck]}
    print(f"{os.path.basename(a.onnx)} clamp={a.clamp}: up {len(up)}/{a.n} ({summ['up_pct']:.1f}%)  "
          f"exact stall {'up at %.1f s' % res[0]['up_s'] if res[0]['up_s'] is not None else 'STUCK'}  "
          f"t med {summ['t_med']}  p90 {summ['t_p90']}  raw neck min {summ['raw_neck_min']:+.2f}  "
          f"clip engaged {100 * summ['clip_rate_mean']:.1f}% of steps")
    if nb:
        nup = [r["up_s"] for r in nb if r["up_s"] is not None]
        summ["nbhd_n"], summ["nbhd_up"] = len(nb), len(nup)
        summ["nbhd_t_med"] = float(np.median(nup)) if nup else None
        summ["nbhd_raw_neck_min"] = min(r["min_raw_neck"] for r in nb)
        print(f"   stall neighbourhood: up {len(nup)}/{len(nb)}  t med {summ['nbhd_t_med']}  raw neck min {summ['nbhd_raw_neck_min']:+.2f}")
    if a.out:
        json.dump({"summary": summ, "states": res, "nbhd": nb}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
