"""Odometry drift, round 3 vs HeadJamRecover 4250 (2026-11-22). CPU MuJoCo + BAM, open scene, no supervisor.

Rebuilds rung-6 Stage 2's isolation protocol (SUMMARY: 3.1 m straight walk ends -1.74..-1.98 % short along-track
under round 3 -> stride_v1 = +1.9 %), plus soccer-like command mixes:
  straight  vx 0.30 for 3.1 m-ish (11 s), headings 0/120/240 deg
  pivot     wz +0.8 and -0.8, 10 s
  arc       vx 0.20 wz +-0.30, 12 s
  strafe    vy +-0.10, 9 s
  mix       60 s of commands resampled every 2 s (vx 0..0.35, vy +-0.05, wz +-1.0; 25 % pivots), 8 seeds
The IMU heading is PERFECT here (ImuYawError disabled), so every error is the leg odometry's. The duck stands 1 s,
then the odometry is anchored at the true trunk pose. Reports raw ('none') and stride_v1 errors; for straight
walks the along-/across-track split and the implied stride scale.
"""
import json, math, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, "scripts")
POL = {"round3": "logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx",
       "hj4250": "logs/rsl_rl/velstand/2026-11-22_19-47-54_headjamB/2026-11-22_19-47-54_headjamB_model_4250.onnx"}
OUT = "logs/evals/velstand/headjam_2026-11-22/stageB/odom_drift"


def segments(kind, arg):
    if kind == "straight": return [((0.30, 0.0, 0.0), 11.0)]
    if kind == "pivot":    return [((0.0, 0.0, arg), 10.0)]
    if kind == "arc":      return [((0.20, 0.0, arg), 12.0)]
    if kind == "strafe":   return [((0.0, arg, 0.0), 9.0)]
    if kind == "mix":
        r = np.random.default_rng(1000 + int(arg)); seg = []
        for _ in range(30):
            if r.random() < 0.25: seg.append(((0.0, 0.0, float(r.uniform(-1, 1))), 2.0))
            else: seg.append(((float(r.uniform(0, 0.35)), float(r.uniform(-0.05, 0.05)), float(r.uniform(-1, 1))), 2.0))
        return seg
    raise ValueError(kind)


def run(job):
    pol, kind, arg = job
    import os, contextlib, io
    os.environ["MICRODUCK_NECK_CLIP"] = "1" if pol == "hj4250" else "0"
    with contextlib.redirect_stdout(io.StringIO()):
        import goto_point as gp
        from pose_provider import OdometryProvider, ImuYawError
        gp.ip.NECK_CLIP_ENABLED = pol == "hj4250"    # pool workers are reused: set it per job
        sim = gp.DuckSim(POL[pol])
    yaw0 = math.radians(arg) if kind == "straight" else 0.0
    sim.reset(0.0, 0.0, yaw0)
    for _ in range(50):
        sim.step()
    prov = {c: OdometryProvider(sim, imu=ImuYawError(enabled=False), calib=c) for c in ("none", "stride_v1")}
    truth0 = prov["none"].frames.trunk_world(sim.data)[0][:2].copy()
    for twist, secs in segments(kind, arg):
        sim.set_twist(twist)
        for _ in range(int(round(secs / 0.02))):
            sim.step()
            for p in prov.values():
                p.update(sim)
    sim.set_twist((0.0, 0.0, 0.0))
    for _ in range(50):
        sim.step()
        for p in prov.values():
            p.update(sim)
    truth = prov["none"].frames.trunk_world(sim.data)[0][:2].copy()
    disp = truth - truth0
    res = {"pol": pol, "kind": kind, "arg": arg, "true_dist_m": float(np.linalg.norm(disp)),
           "fell": bool(sim.pose()[1] > 45)}
    for c, p in prov.items():
        e = np.array(p.get(sim.t).pose[:2]) - truth
        res[f"err_{c}_mm"] = float(1000 * np.linalg.norm(e))
        if kind == "straight" and np.linalg.norm(disp) > 0.5:
            u = disp / np.linalg.norm(disp); n = np.array([-u[1], u[0]])
            res[f"along_{c}_mm"] = float(1000 * e @ u); res[f"across_{c}_mm"] = float(1000 * e @ n)
            res[f"along_{c}_pct"] = float(100 * (e @ u) / np.linalg.norm(disp))
    return res


if __name__ == "__main__":
    jobs = []
    for pol in POL:
        jobs += [(pol, "straight", h) for h in (0, 120, 240)]
        jobs += [(pol, "pivot", w) for w in (0.8, -0.8)] + [(pol, "arc", w) for w in (0.3, -0.3)]
        jobs += [(pol, "strafe", v) for v in (0.1, -0.1)] + [(pol, "mix", s) for s in range(8)]
    with Pool(16) as pool:
        res = pool.map(run, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/drift.json", "w"), indent=1)
    lines = []
    for kind in ("straight", "pivot", "arc", "strafe", "mix"):
        lines.append(f"== {kind}")
        for pol in POL:
            rs = [r for r in res if r["pol"] == pol and r["kind"] == kind]
            if kind == "straight":
                s = "  ".join(f"h{r['arg']}: {r['true_dist_m']:.2f} m along {r['along_none_pct']:+.2f}% ({r['along_none_mm']:+.0f} mm) "
                              f"across {r['across_none_mm']:+.0f} | v1 {r['err_stride_v1_mm']:.0f} mm" for r in rs)
                lines.append(f"  {pol:7s} {s}")
                lines.append(f"  {pol:7s} implied stride scale {1 - np.mean([r['along_none_pct'] for r in rs]) / 100:.4f} (stride_v1 = 1.019)")
            else:
                s = "  ".join(f"{r['arg']:+g}: {r['true_dist_m']:.2f} m raw {r['err_none_mm']:.0f} / v1 {r['err_stride_v1_mm']:.0f} mm"
                              + (" FELL" if r["fell"] else "") for r in rs)
                lines.append(f"  {pol:7s} {s}")
                if kind == "mix":
                    lines.append(f"  {pol:7s} mix 60 s: v1 error median {np.median([r['err_stride_v1_mm'] for r in rs]):.0f} mm, "
                                 f"max {max(r['err_stride_v1_mm'] for r in rs):.0f}; raw median {np.median([r['err_none_mm'] for r in rs]):.0f}; "
                                 f"distance median {np.median([r['true_dist_m'] for r in rs]):.2f} m")
    open(f"{OUT}/DRIFT.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
