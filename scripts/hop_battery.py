"""Headless battery for an OmniHop ONNX (CPU MuJoCo, BAM, 1-step delay, training solver).

Run from the microduck_rl repo root:
  uv run python scripts/hop_battery.py <omnihop.onnx> [--episodes 20] [--drift on|off|both]
      [--prestand <velstand.onnx>] [--dirs in_place,fwd,back,left,right] [--fwd 0.03 --side 0.02]
      [--workers 8] [--out DIR]

Each episode: standing start at a random heading, VelStand (default: shipped 4250) holds the
stand for 1.5 s, then the hop ONNX is swapped in (the runtime contract) and gets
  twist slots = estimated remaining offset to the hop target, trunk yaw frame [dx, dy, 0]
  body slot 0 = hop-request pulse, 1.0 for 0.6 s starting --pulse-delay after the swap
               (default 0.25 s = the runtime settle; 0 = request immediately at the swap)
for 4 s. Drift ON = the training drift model (mdp.py OmniHop section): per-episode odometry
scale bias N(-1.9 %, 0.6 %), a touchdown jump of the estimate up to 2 cm, a persistent gyro
bias up to 0.03 rad/s per axis (the IMU tilt offset has no CPU counterpart here: the CPU
harness reads the real gravity sensor, as on the robot).

What counts as ONE HOP (a measurement definition, not a pass floor): both feet off the
floor >= 60 ms, floor clearance >= 10 mm (min robot-geom gap, not the foot-site rise), touchdown on the feet only, upright
(< 25 deg) at touchdown and still standing (< 25 deg, no fall) 1 s later.
Per direction: success, clearance, flight time, landing falls, true and estimated landing
error vs the target, hops per swap, peak trunk spin (vs the 10.7 rad/s bar), leg
deviation from HOME at the end (swap-back readiness).
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VELSTAND = os.path.join(REPO, "logs/rsl_rl/velstand/2026-11-22_19-47-54_headjamB/2026-11-22_19-47-54_headjamB_model_4250.onnx")
EPISODE_S, PULSE_S, PRESTAND_S = 4.0, 0.6, 1.5
SETTLE_S = 0.25   # RUNTIME DEFAULT (Dillon 2026-11-24): the hop policy holds the stand this long after the
                  # swap before the request pulse. --pulse-delay 0 tests an immediate request.
MIN_AIR_TICKS, MIN_CLEAR, LAND_TILT, HOLD_S = 3, 0.010, 25.0, 1.0
SPIN_BAR = 10.7
LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_frame(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


class HopSim:
    def __init__(self, onnx, velstand):
        import contextlib, io
        import mujoco
        import onnxruntime as ort
        with contextlib.redirect_stdout(io.StringIO()):
            import goto_point as gp
            self.sim = gp.DuckSim(onnx)
        self.mj = mujoco
        self.hop_session = self.sim.policy.ort_session
        self.prestand = ort.InferenceSession(velstand)
        m = self.sim.model
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS}
        ankles = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("ankle_left", "ankle_right")}
        self.ankle_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in ankles}
        self.sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")]
        self.robot_geoms = [g for g in range(m.ngeom) if g != self.floor and m.geom_bodyid[g] > 0
                            and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.gyro_bias = np.zeros(3, dtype=np.float32)
        p = self.sim.policy
        raw = p.get_base_ang_vel
        p.get_base_ang_vel = lambda: raw() + self.gyro_bias

    def floor_gap(self):
        """True clearance: min distance floor <-> any robot collision geom (the probe's metric).
        The foot-site rise the training reward uses overstates it during a tuck (probe best hop:
        site rise 55 mm vs 38 mm geom gap)."""
        m, d = self.sim.model, self.sim.data
        return min(self.mj.mj_geomDistance(m, d, self.floor, g, 0.2, None) for g in self.robot_geoms)

    def contacts(self):
        """(left foot down, right foot down, any non-ankle robot geom on the floor)."""
        d, down, bad = self.sim.data, [False, False], False
        lf, rf = sorted(self.foot_geoms)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if self.floor not in (g1, g2):
                continue
            g = g2 if g1 == self.floor else g1
            if g == lf:
                down[0] = True
            elif g == rf:
                down[1] = True
            elif g not in self.ankle_geoms and self.sim.model.geom_bodyid[g] > 0:
                bad = True
        return down[0], down[1], bad

    def set_cmd(self, off, pulse):
        self.sim.set_twist((off[0], off[1], 0.0))
        self.sim.policy.command[7] = 1.0 if pulse else 0.0


def run_episode(job):
    onnx, velstand, direction, target, drift, seed, delay = job
    global _HS
    if "_HS" not in globals() or _HS is None or _HS[0] != (onnx, velstand):
        _HS = ((onnx, velstand), HopSim(onnx, velstand))
    hs = _HS[1]
    sim = hs.sim
    rng = np.random.default_rng(seed)
    yaw0 = rng.uniform(-math.pi, math.pi)
    sim.reset(0.0, 0.0, yaw0)
    hs.gyro_bias[:] = 0.0
    sim.policy.ort_session = hs.prestand
    sim.set_twist((0.0, 0.0, 0.0))
    while sim.t < PRESTAND_S - 1e-9:
        sim.step()
    (x0, y0, yaw0), tilt0, _ = sim.pose()
    k = float(np.clip(rng.normal(-0.019, 0.006), -0.05, 0.02)) if drift else 0.0
    ja, jr = rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.02)
    jump = np.array([jr * math.cos(ja), jr * math.sin(ja)]) if drift else np.zeros(2)
    hs.gyro_bias[:] = rng.uniform(-0.03, 0.03, 3) if drift else 0.0
    c, s = math.cos(yaw0), math.sin(yaw0)
    goal = (x0 + c * target[0] - s * target[1], y0 + s * target[0] + c * target[1])

    def est_offset(landed):
        (x, y, yaw), _, _ = sim.pose()
        tx, ty = to_frame(goal[0] - x0, goal[1] - y0, yaw)
        ddx, ddy = to_frame(x - x0, y - y0, yaw)
        j = jump if landed else np.zeros(2)
        return (tx - (1 + k) * ddx - j[0], ty - (1 + k) * ddy - j[1])

    sim.policy.ort_session = hs.hop_session
    sim.t = 0.0
    base = None
    air, clear, lift_tilt, site_rise = 0, 0.0, 0.0, 0.0
    hops, landed, first = 0, False, None
    fell_t, bad_t, spin_peak = None, None, 0.0
    while sim.t < EPISODE_S - 1e-9:
        hs.set_cmd(est_offset(landed), delay <= sim.t < delay + PULSE_S)
        sim.step()
        ld, rd, bad = hs.contacts()
        fz = np.array([sim.data.site_xpos[i][2] for i in hs.sites])
        if base is None:
            base = fz.copy()
        for i, dn in enumerate((ld, rd)):
            if dn:
                base[i] = 0.9 * base[i] + 0.1 * fz[i]
        _, tilt, z = sim.pose()
        w = float(np.linalg.norm(sim.data.qvel[sim.va + 3:sim.va + 6]))   # free-joint angular velocity
        spin_peak = max(spin_peak, w)
        if bad and bad_t is None:
            bad_t = sim.t
        if fell_t is None and (tilt > 60 or z < 0.06):
            fell_t = sim.t
        airborne = not (ld or rd)
        if airborne:
            if air == 0:
                lift_tilt, clear, site_rise = tilt, 0.0, 0.0
            clear = max(clear, float(hs.floor_gap()))
            site_rise = max(site_rise, float((fz - base).min()))
            air += 1
        elif air > 0:
            if air >= MIN_AIR_TICKS and clear >= MIN_CLEAR and lift_tilt < 30:
                hops += 1
                if first is None:
                    (x, y, _), _, _ = sim.pose()
                    eo = est_offset(False)
                    first = {"t": round(sim.t, 3), "flight_ms": air * 20, "clear_mm": round(clear * 1000, 1), "site_rise_mm": round(site_rise * 1000, 1),
                             "land_tilt": round(tilt, 1), "true_err_mm": round(1000 * math.hypot(goal[0] - x, goal[1] - y), 1),
                             "est_err_mm": round(1000 * math.hypot(*eo), 1), "bad_contact": bad,
                             "disp_mm": [round(1000 * v, 1) for v in to_frame(x - x0, y - y0, yaw0)]}
                    landed = True
            air = 0
        if first is not None and "hold_tilt" not in first and sim.t >= first["t"] + HOLD_S:
            first["hold_tilt"] = round(tilt, 1)
    q = np.array([sim.data.qpos[j] for j in sim.policy.joint_qpos_indices]) - sim.policy.default_pose
    ok = (first is not None and first["land_tilt"] < LAND_TILT and not first["bad_contact"]
          and first.get("hold_tilt", 99) < LAND_TILT and fell_t is None and bad_t is None)
    return {"dir": direction, "drift": drift, "target_mm": [1000 * target[0], 1000 * target[1]], "seed": seed,
            "success": bool(ok), "hops": hops, "first": first, "fell_t": fell_t, "nonfoot_t": bad_t,
            "landing_fall": bool(first is not None and fell_t is not None and fell_t >= first["t"] - 0.1),
            "spin_peak": round(spin_peak, 2), "leg_dev_deg_end": round(math.degrees(float(np.abs(q[LEG_JOINTS]).max())), 1),
            "start_tilt": round(tilt0, 1)}


def summarize(rows):
    lines = []
    for drift in sorted({r["drift"] for r in rows}, reverse=True):
        lines.append(f"== drift {'ON' if drift else 'OFF'}")
        lines.append(f"  {'dir':9s} {'target mm':>12s} {'success':>8s} {'clear mm':>9s} {'flight ms':>9s} "
                     f"{'true err':>8s} {'est err':>7s} {'hops/swap':>9s} {'land falls':>10s} {'falls':>5s} {'nonfoot':>7s} {'spin pk':>7s} {'>10.7':>5s} {'leg dev':>7s}")
        for d in DIRS_ORDER:
            rs = [r for r in rows if r["drift"] == drift and r["dir"] == d]
            if not rs:
                continue
            f = [r["first"] for r in rs if r["first"]]
            med = lambda v: f"{np.median(v):.1f}" if v else "-"
            lines.append(
                f"  {d:9s} {str([round(v) for v in rs[0]['target_mm']]):>12s} {sum(r['success'] for r in rs):>3d}/{len(rs):<4d} "
                f"{med([x['clear_mm'] for x in f]):>9s} {med([x['flight_ms'] for x in f]):>9s} "
                f"{med([x['true_err_mm'] for x in f]):>8s} {med([x['est_err_mm'] for x in f]):>7s} "
                f"{np.mean([r['hops'] for r in rs]):>9.2f} {sum(r['landing_fall'] for r in rs):>10d} "
                f"{sum(r['fell_t'] is not None for r in rs):>5d} {sum(r['nonfoot_t'] is not None for r in rs):>7d} "
                f"{np.median([r['spin_peak'] for r in rs]):>7.1f} {sum(r['spin_peak'] > SPIN_BAR for r in rs):>5d} "
                f"{np.median([r['leg_dev_deg_end'] for r in rs]):>7.1f}")
    return lines + direction_check(rows)


SAME_HOP_MM = 5.0        # every direction's median landing within this of each other = one hop for all
PAIR_SEP_MIN = 0.5       # opposite directions must separate by >= half the commanded span


def direction_check(rows):
    """Conditioning guard (IsaacLab PR #9173: a collapsed goal latent gave one fixed motion for
    every command). Per direction: median achieved landing displacement (start yaw frame, first
    qualifying hop) vs commanded, gain along the commanded axis, cross-axis drift. Flags (warnings,
    not pass floors): SAME HOP REGARDLESS OF DIRECTION when every direction's median landing sits
    within SAME_HOP_MM of every other; DIRECTION NOT FOLLOWED when an opposite pair (fwd/back,
    left/right) separates by < PAIR_SEP_MIN x the commanded span."""
    out = []
    for drift in sorted({r["drift"] for r in rows}, reverse=True):
        med = {}
        out.append(f"== direction check, drift {'ON' if drift else 'OFF'}")
        out.append(f"  {'dir':9s} {'cmd mm':>12s} {'achieved mm':>14s} {'n':>3s} {'gain':>6s} {'cross mm':>8s}")
        for d in DIRS_ORDER:
            rs = [r for r in rows if r["drift"] == drift and r["dir"] == d]
            ds = [r["first"]["disp_mm"] for r in rs if r["first"]]
            if not rs:
                continue
            t = np.array(rs[0]["target_mm"], float)
            if not ds:
                out.append(f"  {d:9s} {str([round(v) for v in t]):>12s} {'no hops':>14s} {0:>3d}")
                continue
            a = np.median(np.array(ds, float), axis=0)
            med[d] = (t, a)
            n = np.linalg.norm(t)
            gain = f"{a @ t / n ** 2:6.2f}" if n > 0 else "     -"
            cross = abs(a[0] * t[1] - a[1] * t[0]) / n if n > 0 else np.linalg.norm(a)
            out.append(f"  {d:9s} {str([round(v) for v in t]):>12s} {str([round(float(v), 1) for v in a]):>14s} {len(ds):>3d} {gain} {cross:>8.1f}")
        flags = []
        if len(med) >= 3:
            pts = [a for _, a in med.values()]
            spread = max(np.linalg.norm(p - q) for p in pts for q in pts)
            if spread < SAME_HOP_MM:
                flags.append(f"SAME HOP REGARDLESS OF DIRECTION (median landings within {spread:.1f} mm)")
        for p, q, ax in (("fwd", "back", 0), ("left", "right", 1)):
            if p in med and q in med:
                span = abs(med[p][0][ax] - med[q][0][ax])
                sep = (med[p][1][ax] - med[q][1][ax]) / span if span > 0 else 0.0
                out.append(f"  {p}/{q} separation: {sep:.2f} of the commanded span")
                if sep < PAIR_SEP_MIN:
                    flags.append(f"DIRECTION NOT FOLLOWED: {p}/{q} separation {sep:.2f} < {PAIR_SEP_MIN}")
        out += [f"  FLAG: {f}" for f in flags] or ["  flags: none"]
    return out


DIRS_ORDER = ("in_place", "fwd", "back", "left", "right")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--drift", choices=("on", "off", "both"), default="both")
    ap.add_argument("--prestand", default=DEFAULT_VELSTAND)
    ap.add_argument("--dirs", default=",".join(DIRS_ORDER))
    ap.add_argument("--fwd", type=float, default=0.03)
    ap.add_argument("--side", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pulse-delay", type=float, default=SETTLE_S, help="s after the swap before the request pulse")
    a = ap.parse_args()
    tgt = {"in_place": (0.0, 0.0), "fwd": (a.fwd, 0.0), "back": (-a.fwd, 0.0), "left": (0.0, a.side), "right": (0.0, -a.side)}
    drifts = {"on": [True], "off": [False], "both": [True, False]}[a.drift]
    jobs = [(os.path.abspath(a.onnx), os.path.abspath(a.prestand), d, tgt[d], dr, 1000 * i + j, a.pulse_delay)
            for dr in drifts for j, d in enumerate(a.dirs.split(",")) for i in range(a.episodes)]
    with Pool(a.workers) as pool:
        rows = pool.map(run_episode, jobs, chunksize=1)
    lines = [f"OmniHop battery: {a.onnx}", f"  {a.episodes} episodes per row; VelStand prestand {PRESTAND_S}s -> swap-in; request pulse {a.pulse_delay:.2f}s after the swap; "
             f"hop = >= {MIN_AIR_TICKS * 20} ms off, >= {MIN_CLEAR * 1000:.0f} mm clearance, feet-only touchdown "
             f"< {LAND_TILT:.0f} deg, still up {HOLD_S:.0f} s later", ""] + summarize(rows)
    print("\n".join(lines))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "BATTERY.txt"), "w").write("\n".join(lines) + "\n")
        json.dump(rows, open(os.path.join(a.out, "battery.json"), "w"), indent=1)
