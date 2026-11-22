"""HeadJamRecover run-plan probe (2026-11-22): from the dumped 42-deg stall state (6 m LOOK run 5 at 60 s, closing
work), 20 s at zero command, for: (a) VelStand round 3, (b) the stand expert round 3 cloned on fallen frames
(spincap model_11000), (c)/(d) the same two with every joint target CLAMPED to the joint's range at run time.
Reports tilt at 5/10/20 s, min tilt, whether it got up (tilt < 25 deg held 1 s), and the most negative neck_pitch
target commanded. Also a perturbed set: 20 nearby states (joint noise +-0.1 rad, the stall's trunk pose)."""
import math, sys
import numpy as np, mujoco
sys.path.insert(0, "scripts")
from kick_battery import SCENES, KickSim
from kick_mission import setup_goal, setup_field

D = "logs/evals/velstand/headjam_2026-11-22"
S = np.load(f"{D}/stall_state_6m_look_run5_t60.npz")
POL = {"round3": "logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx",
       "stand_expert": f"{D}/stand_expert_spincap_model_11000.onnx"}


def run(name, clamp, noise_seed=None, secs=20.0):
    sim = KickSim(POL[name], SCENES["field"]); setup_goal(sim); setup_field(sim)
    m, d, p = sim.model, sim.data, sim.policy
    jids = [m.jnt_qposadr.tolist().index(q) for q in p.joint_qpos_indices]
    lo, hi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
    names = [m.joint(j).name for j in jids]
    ineck = names.index("neck_pitch")
    seen = {"min_neck": math.inf, "calls": 0}
    orig = p.set_position_targets

    def spt(t):
        t = np.asarray(t, float)
        seen["calls"] += 1
        if seen["calls"] > 5:                 # skip the first 0.1 s: the dumped round-3 action buffer
            seen["min_neck"] = min(seen["min_neck"], float(t[ineck]))
        orig(np.clip(t, lo, hi) if clamp else t)
    p.set_position_targets = spt
    sim.reset(0, 0, 0)
    d.qpos[:] = S["qpos"]; d.qvel[:] = S["qvel"]
    if noise_seed is not None:
        r = np.random.default_rng(noise_seed)
        for q in p.joint_qpos_indices:
            d.qpos[q] += r.uniform(-0.1, 0.1)
    p.last_action[:] = S["last_action"]
    if p.action_buffer is not None:
        for a, b in zip(p.action_buffer, S["buf"]):
            a[:] = b
    p.head_offset[:] = 0.0
    sim.set_twist((0.0, 0.0, 0.0))
    mujoco.mj_forward(m, d)
    tl, up_t = [], None
    n = int(secs / 0.02)
    for k in range(n):
        sim.step()
        tl.append(sim.pose()[1])
        if up_t is None and len(tl) >= 50 and max(tl[-50:]) < 25.0:
            up_t = (k + 1) * 0.02
    return {"t5": tl[249], "t10": tl[499], "t20": tl[-1], "min": min(tl), "up_s": up_t, "min_neck_target": seen["min_neck"],
            "neck_range": (float(lo[ineck]), float(hi[ineck]))}


if __name__ == "__main__":
    print("exact stall state, 20 s, zero command, head neutral")
    for name in POL:
        for clamp in (False, True):
            r = run(name, clamp)
            print(f"  {name:<13} clamp={str(clamp):<5} tilt 5/10/20 s {r['t5']:5.1f} / {r['t10']:5.1f} / {r['t20']:5.1f}  min {r['min']:5.1f}"
                  f"  got up {'at %.1f s' % r['up_s'] if r['up_s'] else 'NO':<10} neck target min {r['min_neck_target']:+.2f} rad"
                  f" (range {r['neck_range'][0]:+.2f}..{r['neck_range'][1]:+.2f})")
    print("20 perturbed stall states (joint noise +-0.1 rad), 15 s each: got up / 20")
    for name in POL:
        for clamp in (False, True):
            res = [run(name, clamp, noise_seed=s, secs=15.0) for s in range(20)]
            up = [r["up_s"] for r in res if r["up_s"]]
            print(f"  {name:<13} clamp={str(clamp):<5} got up {len(up):2d}/20"
                  f"  (median {np.median(up) if up else float('nan'):.1f} s)  neck target min {min(r['min_neck_target'] for r in res):+.2f}")
