"""OmniHop feasibility probe (2026-11-23): can the duck's servos produce a two-foot hop at all?

CPU MuJoCo, allcollisions scene, BAM xl330 actuator (same model as every policy's runtime), training solver.
Open-loop, no policy: hold a crouch target 0.3 s (the duck needs active balance: with no policy it tips over
from HOME in ~1.3 s, so LANDING can't be judged open-loop -- only take-off), then step the targets to an extension pose (optionally a
second "tuck" pose after T_TUCK s to pull the feet up). Leg offsets are symmetric (right = mirrored left)
around HOME on hip_pitch / knee / ankle; head held at HOME.
Measures: flight time = longest stretch with NO robot-floor contact; apex clearance estimate g*t^2/8; trunk z
rise; trunk tilt at lift-off (a hop counts only if it leaves the ground below 30 deg -- not a tumble). Random
search then cross-entropy refinement on that flight time. This is a LOWER bound on what a trained policy could reach.
"""
import json, math, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, "scripts")
SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
OUT = "logs/evals/hop/feasibility_2026-11-23"
LEG_L = (2, 3, 4)            # indices in the 14-servo order: left hip_pitch, knee, ankle
LEG_R = (11, 12, 13)         # mirrored (sign -1)
_SIM = {}


def sim():
    if "s" not in _SIM:
        import contextlib, io, mujoco
        with contextlib.redirect_stdout(io.StringIO()):
            import infer_policy as ip
            ip.MICRODUCK_XML = SCENE
            bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
            m, d, bam, _ = ip.load_mujoco_with_bam(SCENE, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.opt.iterations, m.opt.ls_iterations = 10, 20
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        names = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch",
                 "head_pitch", "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                 "right_knee", "right_ankle"]
        jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in names])
        hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in names])
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        home = np.array([0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0,
                         0, 0.0873, 0.4579, 0.0049, -0.453])
        _SIM["s"] = (mujoco, m, d, bam, m.jnt_qposadr[fj], jq, lo, hi, floor, home)
    return _SIM["s"]


def pose(home, off, lo, hi, head=(0.0, 0.0)):
    q = home.copy()
    q[5] += head[0]
    q[6] += head[1]
    for k in range(3):
        q[LEG_L[k]] += off[k]
        q[LEG_R[k]] -= off[k]
    return np.clip(q, lo + 0.02, hi - 0.02)


def trial(params):
    """params: crouch(3), extend(3), tuck(3), t_tuck. Returns metrics."""
    mujoco, m, d, bam, qa, jq, lo, hi, floor, home = sim()
    p = np.asarray(params, float)
    hc = tuple(p[10:12]) if len(p) >= 14 else (0.0, 0.0)
    he = tuple(p[12:14]) if len(p) >= 14 else (0.0, 0.0)
    crouch, ext, tuck, t_tuck = pose(home, p[0:3], lo, hi, hc), pose(home, p[3:6], lo, hi, he), pose(home, p[6:9], lo, hi, he), p[9]
    mujoco.mj_resetData(m, d)
    d.qpos[qa:qa + 3] = [0, 0, 0.125]
    d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    for i, q in enumerate(jq):
        d.qpos[q] = home[i]
    bam.reset(d.qpos)

    def run(target, n, rec=None):
        bam.q_target[:] = target
        for _ in range(n):
            bam.update() if hasattr(bam, "update") else None
            mujoco.mj_step(m, d)
            if rec is not None:
                rec(d)

    run(home, 100)
    run(crouch, 60)                               # 0.3 s crouch
    q = d.qpos[qa + 3:qa + 7]
    if math.degrees(math.acos(max(-1, min(1, 1 - 2 * (q[1] ** 2 + q[2] ** 2))))) > 20:
        return {"ok": False, "why": "fell in crouch"}
    z0 = float(d.qpos[qa + 2])
    st = {"air": 0, "best_air": 0, "zmax": z0, "t": 0, "lift_tilt": 0.0, "best_lift_tilt": 0.0,
          "vz_peak": 0.0, "vz_lift": 0.0, "best_vz_lift": 0.0, "com0": 0.0, "com_rise": 0.0, "clear": 0.0, "best_com_rise": 0.0, "best_clear": 0.0}
    robot_geoms = [g for g in range(m.ngeom) if g != floor and m.geom_bodyid[g] > 0
                   and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def rec(d):
        st["t"] += 1
        touching = any(d.contact[c].geom1 == floor or d.contact[c].geom2 == floor for c in range(d.ncon))
        st["air"] = 0 if touching else st["air"] + 1
        com = float(d.subtree_com[1][2])
        mujoco.mj_subtreeVel(m, d)
        vz = float(d.subtree_linvel[1][2])
        if touching:
            st["vz_peak"] = max(st["vz_peak"], vz)
        if st["air"] == 1:
            st["vz_lift"] = vz
            w, x, y, z = d.qpos[qa + 3:qa + 7]
            st["lift_tilt"] = math.degrees(math.acos(max(-1, min(1, 1 - 2 * (x * x + y * y)))))
            st["com0"], st["com_rise"], st["clear"] = com, 0.0, 0.0
        if st["air"] >= 1:
            st["com_rise"] = max(st["com_rise"], com - st["com0"])
            gap = min(mujoco.mj_geomDistance(m, d, floor, g, 0.1, None) for g in robot_geoms)
            st["clear"] = max(st["clear"], gap)
        if st["air"] > st["best_air"]:
            st["best_air"], st["best_lift_tilt"] = st["air"], st["lift_tilt"]
            st["best_com_rise"], st["best_clear"], st["best_vz_lift"] = st["com_rise"], st["clear"], st["vz_lift"]
        st["zmax"] = max(st["zmax"], float(d.qpos[qa + 2]))

    n_tuck = int(round(t_tuck / 0.005))
    run(ext, n_tuck, rec)
    run(tuck, 60, rec)
    run(home, 100, rec)                           # 0.5 s more
    q = d.qpos[qa + 3:qa + 7]
    tilt = math.degrees(math.acos(max(-1, min(1, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
    ft = st["best_air"] * 0.005
    return {"ok": True, "flight_s": ft, "apex_clear_mm": 1000 * 9.81 * ft * ft / 8, "trunk_rise_mm": 1000 * (st["zmax"] - z0),
            "lift_tilt": st["best_lift_tilt"], "upright": st["best_lift_tilt"] < 30,
            "com_rise_mm": 1000 * st["best_com_rise"], "clear_mm": 1000 * st["best_clear"],
            "com_vz_peak": st["vz_peak"], "com_vz_lift": st["best_vz_lift"], "params": p.tolist()}


def is_hop(r):
    """Clean lift-off (< 30 deg tilt) with >= 20 ms off the floor. Tuck hops (feet snapped up while the CoM
    already falls) count; launched hops (CoM rises after lift-off) are reported separately."""
    return bool(r.get("ok") and r["upright"] and r["flight_s"] >= 0.02)


def score(r):
    return r["clear_mm"] if is_hop(r) else -1.0


if __name__ == "__main__":
    rng = np.random.default_rng(20261123)
    LO = np.array([-1.2] * 9 + [0.03] + [-1.2] * 4)
    HI = np.array([1.2] * 9 + [0.25] + [1.2] * 4)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    allres = []
    with Pool(workers) as pool:
        X = rng.uniform(LO, HI, size=(2500, 14))
        res = pool.map(trial, X.tolist(), chunksize=10)
        allres += res
        mu, sd = None, None
        for it in range(10):                      # CEM on the measured floor clearance of a real hop
            ok = sorted([r for r in allres if is_hop(r)], key=score, reverse=True)[:40]
            E = np.array([r["params"] for r in ok])
            mu, sd = E.mean(0), E.std(0) + 0.02
            X = np.clip(rng.normal(mu, sd, size=(300, 14)), LO, HI)
            allres += pool.map(trial, X.tolist(), chunksize=10)
            b = max(allres, key=score)
            print(f"cem {it}: best hop clearance {b['clear_mm']:.1f} mm, flight {1000 * b['flight_s']:.0f} ms, "
                  f"CoM rise {b['com_rise_mm']:.1f} mm, lift-off tilt {b['lift_tilt']:.0f} deg", flush=True)
    ok = [r for r in allres if r.get("ok")]
    hops = [r for r in ok if is_hop(r)]
    launched = [r for r in hops if r["com_rise_mm"] > 2.0]
    lines = [f"trials {len(allres)}, crouch held {len(ok)}, hops (clean lift-off < 30 deg, >= 20 ms off the floor) {len(hops)}",
             f"  launched hops (CoM still rising after lift-off by > 2 mm): {len(launched)}"
             + (f", best CoM rise {max(r['com_rise_mm'] for r in launched):.1f} mm" if launched else ""),
             f"  peak upward CoM speed while feet push, any trial: {max(r['com_vz_peak'] for r in ok):.2f} m/s "
             f"(ballistic CoM rise at that speed {1000 * max(r['com_vz_peak'] for r in ok) ** 2 / (2 * 9.81):.1f} mm)",
             f"  CoM vertical speed at lift-off, best-clearance hops: "
             + " ".join(f"{r['com_vz_lift']:+.2f}" for r in sorted(hops, key=score, reverse=True)[:10])]
    for thr in (5, 10, 15, 20, 30):
        lines.append(f"  hops with floor clearance >= {thr} mm: {sum(r['clear_mm'] >= thr for r in hops)}")
    best = max(hops, key=score)
    lines.append(f"best hop: floor clearance {best['clear_mm']:.1f} mm, flight {1000 * best['flight_s']:.0f} ms, CoM rise "
                 f"{best['com_rise_mm']:.1f} mm, lift-off tilt {best['lift_tilt']:.0f} deg; params {np.round(best['params'], 3).tolist()}")
    top = sorted(hops, key=score, reverse=True)[:20]
    lines.append("top-20 hop clearances (mm): " + " ".join(f"{r['clear_mm']:.0f}" for r in top))
    lines.append("top-20 flight (ms):          " + " ".join(f"{1000 * r['flight_s']:.0f}" for r in top))
    json.dump({"best_hop": best, "top20": top}, open(f"{OUT}/best.json", "w"), indent=1)
    open(f"{OUT}/PROBE.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
