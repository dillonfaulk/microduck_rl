"""FineTurn Step 0 (2026-11-23): low-end command response, CPU MuJoCo + BAM (1-step delay, training solver).

Pure-channel commands, each from two starts:
  still   stand 2 s at zero, then the command for 8 s
  rolling walk 3 s at a carrier (vx 0.20 for linear rows, in-place wz 0.8 same sign for yaw rows), then the
          command for 8 s -- separates a start-up threshold from a can't-sustain one
Achieved = mean over the last 6 s of the command window (body-frame velocity from trunk displacement, yaw rate
from heading change). Policies:
  hj4250   shipped VelStand (neck clip on)
  round3   previous VelStand
  anchor   walk v2 (velocity walkv2_chunk2 model_7998), fed the command VelStand's BC feeds it on upright frames
           (WALKV2_ANCHOR_TWIST_REMAP, distill.remap_anchor_twist rules for pure-channel commands)
  anchor_raw  walk v2 fed the raw command
"""
import json, math, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, "scripts")
POL = {"hj4250": "logs/rsl_rl/velstand/2026-11-22_19-47-54_headjamB/2026-11-22_19-47-54_headjamB_model_4250.onnx",
       "round3": "logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx",
       "anchor": "logs/rsl_rl/velocity/2026-11-01_17-21-17_walkv2_chunk2/2026-11-01_17-21-17_walkv2_chunk2.onnx",
       "anchor_raw": "logs/rsl_rl/velocity/2026-11-01_17-21-17_walkv2_chunk2/2026-11-01_17-21-17_walkv2_chunk2.onnx"}
OUT = "logs/evals/velstand/fineturn_2026-11-23"
# Overrides (FineTurn checkpoint evals, 2026-11-24): SWEEP_POLS='{"name": "path.onnx", ...}' replaces the policy
# set, SWEEP_FRICTION scales BAM's velocity-independent friction budget (frictionloss; = training's friction DR
# hook), SWEEP_OUT sets the output dir. Unset = the Step 0 sweep exactly.
import os as _os
if _os.environ.get("SWEEP_POLS"):
    POL = json.loads(_os.environ["SWEEP_POLS"])
FRICTION = float(_os.environ.get("SWEEP_FRICTION", "1.0"))
OUT = _os.environ.get("SWEEP_OUT", OUT)
WZ = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
VX = (0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2)
VY = (0.02, 0.05, 0.075, 0.1, 0.15)


def anchor_fed(twist):
    """Pure-channel version of distill.remap_anchor_twist with WALKV2_ANCHOR_TWIST_REMAP (0 -> 0 knot, linear)."""
    from mjlab_microduck.tasks.microduck_velstand_env_cfg import WALKV2_ANCHOR_TWIST_REMAP as T
    vx, vy, wz = twist
    out = [vx, vy, wz]
    for i, key in ((0, "x"), (1, "y")):
        v = twist[i]
        if v != 0:
            side = "pos" if v > 0 else "neg"
            out[i] = math.copysign(float(np.interp(abs(v), [0, *T[key]["targets"]], [0, *T[key][side]])), v)
    if wz != 0 and vx == 0 and vy == 0:   # yaw row keyed by target |v_xy| = 0 -> first row
        side = "pos" if wz > 0 else "neg"
        out[2] = math.copysign(float(np.interp(abs(wz), [0, *T["targets"]], [0, *T[side][0]])), wz)
    return tuple(out)


def run(job):
    pol, ch, val, start = job
    import os, contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        import goto_point as gp
        gp.ip.NECK_CLIP_ENABLED = False          # hj4250 gets it from the file name (auto clip)
        sim = gp.DuckSim(POL[pol])
    if FRICTION != 1.0:
        bm = sim.bam.model
        orig = bm.compute_frictions
        def scaled(*a, **k):
            fl, damp = orig(*a, **k)
            return fl * FRICTION, damp
        bm.compute_frictions = scaled
    twist = {"wz": (0.0, 0.0, val), "vx": (val, 0.0, 0.0), "vy": (0.0, val, 0.0)}[ch]
    fed = anchor_fed(twist) if pol == "anchor" else twist
    sim.reset(0.0, 0.0, 0.0)
    for _ in range(100):
        sim.step()
    if start == "rolling":
        carrier = (0.0, 0.0, math.copysign(0.8, val)) if ch == "wz" else (0.20, 0.0, 0.0)
        if pol == "anchor":
            carrier = anchor_fed(carrier)
        sim.set_twist(carrier)
        for _ in range(150):
            sim.step()
    sim.set_twist(fed)
    for _ in range(100):                        # 2 s transition, not scored
        sim.step()
    xs, ys, yaws = [], [], []
    tilt_max = 0.0
    for _ in range(300):                        # 6 s scored
        sim.step()
        q = sim.data.qpos[sim.qa:sim.qa + 7]
        xs.append(q[0]); ys.append(q[1])
        w, x, y, z = q[3:7]
        yaws.append(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        tilt_max = max(tilt_max, tilt)
    yaw = np.unwrap(np.array(yaws))
    dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
    ym = float(np.mean(yaw))                    # body frame at the mean heading of the window
    bx = math.cos(ym) * dx + math.sin(ym) * dy
    by = -math.sin(ym) * dx + math.cos(ym) * dy
    T = 299 * 0.02
    return {"pol": pol, "ch": ch, "cmd": val, "start": start, "fed": fed,
            "vx": bx / T, "vy": by / T, "wz": float(yaw[-1] - yaw[0]) / T, "tilt_max": tilt_max,
            "fell": tilt_max > 45}


if __name__ == "__main__":
    jobs = []
    for pol in POL:
        for start in ("still", "rolling"):
            jobs += [(pol, "wz", s * v, start) for v in WZ for s in (1, -1)]
            jobs += [(pol, "vx", s * v, start) for v in VX for s in (1, -1)]
            jobs += [(pol, "vy", s * v, start) for v in VY for s in (1, -1)]
    with Pool(16) as pool:
        res = pool.map(run, jobs, chunksize=2)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/sweep.json", "w"), indent=1)
    os.makedirs(OUT, exist_ok=True)
    lines = [f"friction x{FRICTION}"] if FRICTION != 1.0 else []
    lines += ["achieved / commanded on the commanded channel (pure-channel commands, 6 s window); * = fell",
             "anchor = walk v2 fed VelStand's remapped BC command (the 'fed' value in sweep.json)", ""]
    for ch, vals in (("wz", WZ), ("vx", VX), ("vy", VY)):
        for start in ("still", "rolling"):
            lines.append(f"== {ch} from {start}" + ("   (rad/s)" if ch == "wz" else "   (m/s)"))
            lines.append("  cmd      " + "".join(f"{s * v:>8.3f}" for v in vals for s in (1, -1)))
            for pol in POL:
                row = []
                for v in vals:
                    for s in (1, -1):
                        r = next(r for r in res if r["pol"] == pol and r["ch"] == ch and r["cmd"] == s * v and r["start"] == start)
                        row.append(f"{r[ch]:>7.3f}" + ("*" if r["fell"] else " "))
                lines.append(f"  {pol:9s}" + "".join(row))
            lines.append("")
    open(f"{OUT}/SWEEP.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
