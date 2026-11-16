"""Drive a VelStand policy to (x, y) waypoints with the v0 go-to-point supervisor.

Run from the microduck_rl repo root.

Live (MuJoCo viewer, keys typed in the terminal: P push, SPACE e-stop, Q quit):
  uv run python scripts/goto_point.py <onnx> --viewer --waypoints "1.2,0;1.2,1;0,0"

Headless acceptance suite (writes logs/goto/<stamp>_<suite>/):
  uv run python scripts/goto_point.py <onnx> --suite all

Sim settings match logs/tools/headless_rehearsal.py's fixed harness: BAM actuator,
1-step action delay, training solver, all-collisions scene, projected gravity.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_policy as ip  # noqa: E402
from goto_supervisor import GotoConfig, GotoSupervisor  # noqa: E402
from pose_provider import ODOM_CALIBS, make_provider, pose_error  # noqa: E402

POSE_LOST_ABORT_S = 5.0   # pose absent this long: the route ends cleanly (aborted pose_lost)

SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
CONTROL_DT = 0.02  # 50 Hz, physics 5 ms x decimation 4
PUSH_MAX = 1.0     # infer_policy's P key push


class DuckSim:
    """One CPU MuJoCo duck driven by the policy at 50 Hz."""

    def __init__(self, onnx, scene=SCENE, delay=1, train_solver=True):
        ip.MICRODUCK_XML = scene
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(scene, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
        if train_solver:
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.model.opt.iterations, self.model.opt.ls_iterations = 10, 20
        self.policy = ip.PolicyInference(self.model, self.data, bam_ctrl=self.bam, walking_onnx_path=onnx,
                                         new_cmd_obs=True, use_projected_gravity=True,
                                         delay_min_lag=delay, delay_max_lag=delay)
        fj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qa, self.va = self.model.jnt_qposadr[fj], self.model.jnt_dofadr[fj]
        self.trunk_body = self.model.jnt_bodyid[fj]
        self.t = 0.0

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        d, qa = self.data, self.qa
        d.qpos[qa:qa + 3] = [x, y, 0.125]
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, q in enumerate(self.policy.joint_qpos_indices):
            d.qpos[q] = self.policy.default_pose[i]
        d.qvel[:] = 0
        d.time = 0.0
        p = self.policy  # per-episode policy state: last action obs + delay buffer
        p.last_action[:] = 0
        if p.action_buffer is not None:
            for a in p.action_buffer:
                a[:] = 0
        self.bam.reset(d.qpos)
        self.policy.set_position_targets(self.policy.default_pose)
        self.set_twist((0.0, 0.0, 0.0))
        mujoco.mj_forward(self.model, d)
        self.t = 0.0

    def set_twist(self, twist):
        # Twist slots only; head/body pose slots stay zero, as in infer_policy driving.
        # (set_vel_cmd would print every tick.)
        self.policy.vel_cmd = np.array(twist, dtype=np.float32)
        self.policy._update_command()

    def step(self):
        self.policy.apply_action(self.policy.infer())
        for _ in range(4):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
        self.t += CONTROL_DT

    def pose(self):
        """((x, y, yaw), tilt_deg, trunk_z) from ground truth."""
        q = self.data.qpos
        w, x, y, z = q[self.qa + 3:self.qa + 7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        return (float(q[self.qa]), float(q[self.qa + 1]), yaw), tilt, float(q[self.qa + 2])

    def achieved(self):
        """Body-frame (vx, wz)."""
        quat = self.data.qpos[self.qa + 3:self.qa + 7].astype(np.float32)
        vb = self.policy.quat_rotate_inverse(quat, self.data.qvel[self.va:self.va + 3].astype(np.float32))
        return float(vb[0]), float(self.data.qvel[self.va + 5])

    def push(self, vx, vy):
        """Overwrite the trunk's world xy velocity, like the training push event / P key."""
        self.data.qvel[self.va], self.data.qvel[self.va + 1] = vx, vy


class RunLog:
    """Events (jsonl) + full 50 Hz command trace (csv) for one run."""

    def __init__(self, path=None, echo=True):
        self.events, self.trace, self.path, self.echo = [], [], path, echo

    def event(self, e):
        self.events.append(e)
        if self.echo and e["event"] != "GO_TO_TRACE":
            print(f"  [{e['t']:6.2f}s] {e['event']:12s} " +
                  " ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "event")), flush=True)

    def tick(self, sim, sup, twist):
        (x, y, yaw), tilt, z = sim.pose()
        avx, awz = sim.achieved()
        L = sup.last
        self.trace.append({"t": round(sim.t, 3), "x": round(x, 4), "y": round(y, 4), "yaw": round(yaw, 4),
                           "tilt_deg": round(tilt, 1), "trunk_z": round(z, 4), "state": L["state"],
                           "reason": L["reason"], "wp": L["wp"],
                           "dist": None if L["dist"] is None else round(L["dist"], 4),
                           "heading_err": None if L["err"] is None else round(L["err"], 4),
                           "cmd_vx": twist[0], "cmd_vy": twist[1], "cmd_wz": round(twist[2], 4),
                           "ach_vx": round(avx, 4), "ach_wz": round(awz, 4)})

    def save(self, stem):
        with open(stem + "_events.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e, default=float) + "\n")
        if not self.trace:
            return
        with open(stem + "_trace.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.trace[0].keys()))
            w.writeheader()
            w.writerows(self.trace)


def leg_report(trace):
    """Commanded vs achieved per waypoint leg, split by arc/pivot and turn direction."""
    lines, legs = [], {}
    for r in trace:
        if r["reason"] not in ("arc", "pivot"):
            continue
        kind = r["reason"]
        if abs(r["cmd_wz"]) >= 0.05:
            kind += " left" if r["cmd_wz"] > 0 else " right"
        else:
            kind += " straight"
        legs.setdefault(r["wp"], {}).setdefault(kind, []).append(r)
    for wp, kinds in legs.items():
        for kind, rows in sorted(kinds.items()):
            c = np.array([(r["cmd_vx"], r["cmd_wz"], r["ach_vx"], r["ach_wz"]) for r in rows])
            lines.append(f"    wp {wp} {kind:15s} {len(rows) * CONTROL_DT:5.1f}s  cmd vx {c[:,0].mean():+.2f} "
                         f"wz {c[:,1].mean():+.2f}  ->  achieved vx {c[:,2].mean():+.2f} wz {c[:,3].mean():+.2f}")
    return lines


def run_route(sim, waypoints, arena, start, timeout_s, cfg=None, pushes=(), echo=True, pose_src=None, seed=0,
              dropouts=()):
    """Run one route headless. pushes = [(trigger(sim, sup) -> bool, (vx, vy))], each fired once.

    pose_src (pose_provider.py) is the supervisor's pose: None = simulator truth, exactly as before
    the seam. While it reports the pose absent the supervisor is not updated, so its 200 ms
    watchdog holds a zero twist; absent past POSE_LOST_ABORT_S ends the route (aborted pose_lost)."""
    sim.reset(*start)
    if pose_src is not None:
        pose_src.reset(sim, seed=seed)
        pose_src.dropouts = list(dropouts)
    log = RunLog(echo=echo)
    sup = GotoSupervisor(waypoints, arena, cfg, log=log.event)
    sup.aborted, lost_t = None, None
    pending = list(pushes)
    marker_history = [tuple(i for i, _ in sup.queue)]

    def feed():
        """Update the supervisor; False while the pose is absent (the caller then holds zero twist)."""
        nonlocal lost_t
        if pose_src is None:
            sup.update(sim.t, *sim.pose())
            return True
        e = pose_src.get(sim.t)
        if e.pose is None:
            if lost_t is None:
                lost_t = sim.t
                e = pose_src.get(sim.t)
                log.event({"t": round(sim.t, 3), "event": "POSE_LOST", "lost": e.lost})
            if sim.t - lost_t > POSE_LOST_ABORT_S and sup.aborted is None:
                sup.aborted = "pose_lost"
                log.event({"t": round(sim.t, 3), "event": "POSE_LOST_ABORT", "after_s": round(sim.t - lost_t, 2)})
            return False
        if lost_t is not None:
            log.event({"t": round(sim.t, 3), "event": "POSE_BACK", "after_s": round(sim.t - lost_t, 2)})
            lost_t = None
        sup.update(e.t, e.pose, e.tilt_deg, e.trunk_z)   # the estimate's own stamp feeds the watchdog
        return True

    def tick_extra():
        if pose_src is not None:
            e = pose_src.get(sim.t)
            log.trace[-1].update({"est_x": None if e.pose is None else round(e.pose[0], 4),
                                  "est_y": None if e.pose is None else round(e.pose[1], 4),
                                  "est_yaw": None if e.pose is None else round(e.pose[2], 4), "est_lost": e.lost})

    while sim.t < timeout_s and sup.state != "DONE" and sup.aborted is None:
        twist = sup.command(sim.t) if feed() else (0.0, 0.0, 0.0)
        sim.set_twist(twist)
        log.tick(sim, sup, twist)
        tick_extra()
        for p in list(pending):
            if p[0](sim, sup):
                sim.push(*p[1])
                log.event({"t": round(sim.t, 3), "event": "PUSH", "v": list(p[1])})
                pending.remove(p)
        sim.step()
        if pose_src is not None:
            pose_src.update(sim)
        q = tuple(i for i, _ in sup.queue)
        if q != marker_history[-1]:
            marker_history.append(q)
    for _ in range(50):  # 1 s of standing after the route, so a stop-time fall would show
        twist = sup.command(sim.t) if feed() and sup.aborted is None else (0.0, 0.0, 0.0)
        sim.set_twist(twist)
        log.tick(sim, sup, twist)
        tick_extra()
        sim.step()
        if pose_src is not None:
            pose_src.update(sim)
    return sup, log, marker_history


def arrival_errors(log, goal_by_wp):
    """Per GO_TO_ARRIVE: (wp, true distance to the goal, |estimate - truth|) from the 50 Hz trace."""
    out = []
    for e in log.events:
        if e["event"] != "GO_TO_ARRIVE":
            continue
        row = min(log.trace, key=lambda r: abs(r["t"] - e["t"]))
        gx, gy = goal_by_wp[e["wp"]]
        true_d = math.hypot(row["x"] - gx, row["y"] - gy)
        est_err = (math.hypot(row["est_x"] - row["x"], row["est_y"] - row["y"])
                   if row.get("est_x") is not None else 0.0)
        out.append((e["wp"], true_d, est_err))
    return out


# ---------------------------------------------------------------- acceptance suites

ARENA = (-2.5, 2.5, -2.5, 2.5)


def unrecovered(sup, log):
    """Falls with no matching RECOVERED (i.e. still down when the run ended)."""
    count = lambda name: sum(e["event"] == name for e in log.events)  # noqa: E731
    return count("FALL") - count("RECOVERED")


def suite_single(sim, out, trials=10, seed=0, pose_src=None):
    rng = np.random.default_rng(seed)
    print(f"\n== single point: {trials} trials, random start pose + heading, goal 0.6-2.0 m any bearing ==")
    ok = 0
    for k in range(trials):
        start = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-math.pi, math.pi))
        d, b = rng.uniform(0.6, 2.0), rng.uniform(-math.pi, math.pi)  # bearing relative to start heading
        goal = (float(np.clip(start[0] + d * math.cos(start[2] + b), -2.3, 2.3)),
                float(np.clip(start[1] + d * math.sin(start[2] + b), -2.3, 2.3)))
        dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        sup, log, _ = run_route(sim, [goal], ARENA, start, timeout_s=10 + 2 * dist / 0.2, echo=False,
                                pose_src=pose_src, seed=k)
        arrived = sup.arrived == [1]
        errs = arrival_errors(log, {1: goal})
        true_d, est_err = (errs[0][1], errs[0][2]) if errs else (float("nan"), float("nan"))
        final = log.trace[-1]
        final_true_d = math.hypot(final["x"] - goal[0], final["y"] - goal[1])
        SINGLE_ROWS.append({"trial": k, "arrived": arrived, "dist_m": round(dist, 2), "true_dist_at_arrival_m": round(true_d, 3),
                            "est_err_at_arrival_m": round(est_err, 3), "final_true_dist_m": round(final_true_d, 3),
                            **(pose_src.summary() if hasattr(pose_src, "summary") else {})})
        t_arr = next((e["t"] for e in log.events if e["event"] == "GO_TO_ARRIVE"), None)
        bad = unrecovered(sup, log)
        ok += arrived and bad == 0
        print(f"  trial {k}: start ({start[0]:+.2f},{start[1]:+.2f}, {math.degrees(start[2]):+4.0f}deg) "
              f"goal ({goal[0]:+.2f},{goal[1]:+.2f}) dist {dist:.2f} m, bearing {math.degrees(b):+4.0f}deg -> "
              f"{'ARRIVED %.1fs' % t_arr if arrived else 'NOT ARRIVED'}  falls {sup.falls}"
              f"{' UNRECOVERED' if bad else ''}"
              f" | TRUE dist to goal at arrival {true_d * 100:.1f} cm (radius 15), estimate off by {est_err * 100:.1f} cm,"
              f" final true dist {final_true_d * 100:.1f} cm", flush=True)
        log.save(f"{out}/single_{k}")
    print(f"  single point: {ok}/{trials} arrivals with no unrecovered falls")
    td = [r["true_dist_at_arrival_m"] for r in SINGLE_ROWS[-trials:]]
    ee = [r["est_err_at_arrival_m"] for r in SINGLE_ROWS[-trials:]]
    print(f"  true distance to goal at arrival: median {np.nanmedian(td) * 100:.1f} cm, max {np.nanmax(td) * 100:.1f} cm; "
          f"estimate error at arrival: median {np.nanmedian(ee) * 100:.1f} cm, max {np.nanmax(ee) * 100:.1f} cm; "
          f"truly inside the 15 cm radius {sum(d <= 0.15 for d in td)}/{trials}")
    return ok == trials


CHAIN = [(1.2, 0.0), (1.2, 1.0), (2.2, 1.3), (0.0, 0.0)]  # straight, left 90, right arc, behind (pivot)
SINGLE_ROWS = []


def suite_chain(sim, out, pose_src=None):
    print(f"\n== chain: {CHAIN} from (0,0) facing +x ==")
    sup, log, markers = run_route(sim, CHAIN, ARENA, (0, 0, 0), timeout_s=90, pose_src=pose_src)
    log.save(f"{out}/chain")
    print("  marker queue over time: " + " -> ".join(str(list(m)) for m in markers))
    print("\n".join(leg_report(log.trace)))
    ok = sup.arrived == [1, 2, 3, 4] and unrecovered(sup, log) == 0
    for wp, td, ee in arrival_errors(log, {i + 1: g for i, g in enumerate(CHAIN)}):
        print(f"  wp {wp}: TRUE dist to goal at arrival {td * 100:.1f} cm, estimate off by {ee * 100:.1f} cm")
    print(f"  chain: arrived {sup.arrived} in {sim.t:.1f}s, falls {sup.falls} -> {'PASS' if ok else 'FAIL'}")
    return ok


def suite_fall(sim, out, push_v=1.6, retries=4, pose_src=None):
    """Knock it over mid-leg; retry with a harder push if the first one doesn't topple it."""
    print(f"\n== mid-route fall: route (0,0) -> (2.0, 0) -> (2.0, 1.0), lateral push at x > 0.8 m ==")
    for k in range(retries):
        v = push_v + 0.4 * k
        trig = (lambda s, sup: s.pose()[0][0] > 0.8 and sup.current is not None and sup.current[0] == 1)
        sup, log, _ = run_route(sim, [(2.0, 0.0), (2.0, 1.0)], ARENA, (0, 0, 0), timeout_s=60,
                                pushes=[(trig, (0.0, v))], echo=False, pose_src=pose_src)
        if sup.falls:
            break
        print(f"  push {v:.1f} m/s did not topple it, retrying harder", flush=True)
    for e in log.events:
        if e["event"] not in ("GO_TO_TRACE",):
            print(f"  [{e['t']:6.2f}s] {e['event']:12s} " +
                  " ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "event")))
    log.save(f"{out}/fall")
    fall_t = next((e["t"] for e in log.events if e["event"] == "FALL"), None)
    rec_t = next((e["t"] for e in log.events if e["event"] == "RECOVERED"), None)
    zero_during = all(r["cmd_vx"] == 0 and r["cmd_wz"] == 0 for r in log.trace
                      if fall_t is not None and rec_t is not None and fall_t <= r["t"] < rec_t)
    ok = bool(sup.falls) and sup.arrived == [1, 2] and unrecovered(sup, log) == 0 and zero_during
    print(f"  fall: falls {sup.falls}, zero twist while down {zero_during}, "
          f"FALL->RECOVERED {None if fall_t is None or rec_t is None else round(rec_t - fall_t, 2)}s, "
          f"arrived {sup.arrived} -> {'PASS' if ok else 'FAIL'}")
    return ok


def suite_geofence(sim, out, pose_src=None):
    wps = [(1.0, 0.0), (4.0, 0.0), (1.0, 1.0)]
    print(f"\n== geofence: arena x/y in [-2.5, 2.5], waypoints {wps} (#2 is outside) ==")
    sup, log, _ = run_route(sim, wps, ARENA, (0, 0, 0), timeout_s=40, pose_src=pose_src)
    log.save(f"{out}/geofence")
    rej = [e for e in log.events if e["event"] == "GO_TO_REJECT"]
    first_start = next(i for i, e in enumerate(log.events) if e["event"] == "GO_TO_START")
    before_motion = all(log.events.index(e) < first_start for e in rej) and all(e["t"] == 0 for e in rej)
    ok = [e["wp"] for e in rej] == [2] and before_motion and sup.arrived == [1, 3]
    print(f"  geofence: rejected {[e['wp'] for e in rej]} before motion {before_motion}, arrived {sup.arrived}"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def suite_dropout(sim, out, pose_src):
    """Kill the pose source mid-route. 2 s outage: hold (zero twist), then resume and arrive.
    Permanent outage: hold, then end cleanly (aborted pose_lost) with the duck standing."""
    print("\n== dropout: route (0,0) -> (2.0, 0) -> (2.0, 1.0); pose link down at t = 3 s ==")
    results = []
    for label, dur in (("2 s outage", 2.0), ("permanent", 1e9)):
        sup, log, _ = run_route(sim, [(2.0, 0.0), (2.0, 1.0)], ARENA, (0, 0, 0), timeout_s=60, echo=False,
                                pose_src=pose_src, dropouts=[(3.0, 3.0 + dur)])
        lost = next((e["t"] for e in log.events if e["event"] == "POSE_LOST"), None)
        back = next((e["t"] for e in log.events if e["event"] == "POSE_BACK"), None)
        end = back if back is not None else log.trace[-1]["t"] + 1e-9
        held = [r for r in log.trace if lost is not None and lost <= r["t"] < end]
        zero = all(r["cmd_vx"] == 0 and r["cmd_vy"] == 0 and r["cmd_wz"] == 0 for r in held)
        # what the body did while held: the trunk displacement over the outage
        moved = (math.hypot(held[-1]["x"] - held[0]["x"], held[-1]["y"] - held[0]["y"]) if held else float("nan"))
        if dur < 1e8:
            ok = zero and back is not None and sup.arrived == [1, 2] and unrecovered(sup, log) == 0
            what = f"resumed after {back - lost:.2f} s, arrived {sup.arrived}"
        else:
            ok = zero and sup.aborted == "pose_lost" and unrecovered(sup, log) == 0
            what = f"ended {sup.aborted} at {log.trace[-1]['t']:.1f} s, arrived {sup.arrived}"
        print(f"  {label}: POSE_LOST at {lost} s, zero twist while held {zero} ({len(held)} ticks), trunk moved "
              f"{moved * 100:.1f} cm while held, {what}, falls {sup.falls} -> {'PASS' if ok else 'FAIL'}")
        log.save(f"{out}/dropout_{label.split()[0]}")
        results.append(ok)
    return all(results)


# ---------------------------------------------------------------- live viewer

def draw_scene(scn, sup, arena, sim):
    scn.ngeom = 0

    def add(gtype, size, pos, rgba, mat=np.eye(3)):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).flatten(), np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def line(a, b, rgba, width=0.006):
        g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, [1, 1, 1], [0, 0, 0], rgba)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))

    xmin, xmax, ymin, ymax = arena
    corners = [(xmin, ymin, 0.002), (xmax, ymin, 0.002), (xmax, ymax, 0.002), (xmin, ymax, 0.002)]
    for i in range(4):
        line(corners[i], corners[(i + 1) % 4], [1.0, 0.2, 0.2, 0.8], 0.008)
    r = sup.cfg.arrival_radius
    for n, (wid, (x, y)) in enumerate(sup.queue):  # arrived waypoints have left the queue -> despawned
        rgba = [0.1, 0.9, 0.2, 0.55] if n == 0 else [1.0, 0.6, 0.1, 0.45]
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [r, 0.003, 0], [x, y, 0.003], rgba)
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.006, 0.08, 0], [x, y, 0.08], rgba[:3] + [0.9])
    if sup.current is not None:
        (px, py, _), _, _ = sim.pose()
        line((px, py, 0.01), (*sup.current[1], 0.01), [0.1, 0.9, 0.2, 0.6], 0.003)


def run_viewer(sim, waypoints, arena, cfg):
    import mujoco.viewer
    sim.reset(0, 0, 0)
    log = RunLog()
    sup = GotoSupervisor(waypoints, arena, cfg, log=log.event)
    import random
    print("\nKeys (type in THIS terminal): P = random push, SPACE = e-stop on/off, Q = quit\n")
    with ip.TerminalInput() as term, \
            mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.trunk_body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.6, -35, 135
        quit_req = False
        while viewer.is_running() and not quit_req:
            t0 = time.time()
            for key in term.get_keys():
                if key == "q":
                    quit_req = True
                elif key == " ":
                    sup.estop = not sup.estop
                    log.event({"t": round(sim.t, 3), "event": "ESTOP_ON" if sup.estop else "ESTOP_OFF"})
                elif key == "p":
                    a = random.uniform(0, 2 * math.pi)
                    sim.push(PUSH_MAX * math.cos(a), PUSH_MAX * math.sin(a))
                    log.event({"t": round(sim.t, 3), "event": "PUSH", "angle_deg": round(math.degrees(a))})
            sup.update(sim.t, *sim.pose())
            twist = sup.command(sim.t)
            sim.set_twist(twist)
            log.tick(sim, sup, twist)
            L = sup.last
            if int(sim.t / CONTROL_DT) % 50 == 0 and L["dist"] is not None:
                avx, awz = sim.achieved()
                print(f"  wp {L['wp']} {L['reason']:6s} dist {L['dist']:.2f} m  err {math.degrees(L['err']):+5.0f}deg  "
                      f"cmd vx {twist[0]:.2f} wz {twist[2]:+.2f}  achieved vx {avx:+.2f} wz {awz:+.2f}", flush=True)
            sim.step()
            draw_scene(viewer.user_scn, sup, arena, sim)
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
    out = f"logs/goto/{time.strftime('%Y-%m-%d_%H-%M-%S')}_viewer"
    os.makedirs(out, exist_ok=True)
    log.save(f"{out}/route")
    print(f"\narrived {sup.arrived}, rejected {sup.rejected}, falls {sup.falls}; log: {out}/")
    print("\n".join(leg_report(log.trace)))


def parse_points(s):
    return [tuple(float(v) for v in p.split(",")) for p in s.split(";") if p.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--waypoints", default="1.2,0;1.2,1;2.2,1.3;0,0", help='"x1,y1;x2,y2;..." world frame, m')
    ap.add_argument("--arena", default="-2.5,2.5,-2.5,2.5", help="xmin,xmax,ymin,ymax geofence, m")
    ap.add_argument("--suite", choices=["single", "chain", "fall", "geofence", "dropout", "all"], default="all")
    ap.add_argument("--pose", choices=("gt", "odom", "mapd"), default="gt",
                    help="supervisor pose source (rung 6): gt = simulator truth, odom = contact odometry + IMU")
    ap.add_argument("--no-imu-error", action="store_true", help="odom: perfect IMU heading")
    ap.add_argument("--odom-calib", choices=tuple(ODOM_CALIBS), default="none",
                    help="odom: versioned odometry calibration (stride_v1 = +1.9 %% stride scale)")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--scene", default=SCENE)
    args = ap.parse_args()

    sim = DuckSim(args.onnx, scene=args.scene, delay=args.delay)
    if args.viewer:
        run_viewer(sim, parse_points(args.waypoints), tuple(float(v) for v in args.arena.split(",")), GotoConfig())
        return
    base, n = f"logs/goto/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.suite}_{args.pose}", 1
    out = base
    while os.path.exists(out):   # suites launched in the same second must not share a directory
        n += 1
        out = f"{base}_{n}"
    os.makedirs(out)
    pose_src = None if args.pose == "gt" else make_provider(args.pose, sim, imu_error=not args.no_imu_error, calib=args.odom_calib)
    print(f"onnx={args.onnx}\nlogs -> {out}/\npose source: {args.pose}"
          + (f" (IMU yaw error {'off' if args.no_imu_error else 'on'}, calib {args.odom_calib})" if pose_src is not None else ""))
    results = {}
    if args.suite in ("single", "all"):
        results["single"] = suite_single(sim, out, args.trials, pose_src=pose_src)
    if args.suite in ("chain", "all"):
        results["chain"] = suite_chain(sim, out, pose_src=pose_src)
    if args.suite in ("fall", "all"):
        results["fall"] = suite_fall(sim, out, pose_src=pose_src)
    if args.suite in ("geofence", "all"):
        results["geofence"] = suite_geofence(sim, out, pose_src=pose_src)
    if args.suite == "dropout" or (args.suite == "all" and pose_src is not None):
        results["dropout"] = suite_dropout(sim, out, pose_src or make_provider("gt", sim))
    if SINGLE_ROWS:
        with open(f"{out}/single_errors.json", "w") as f:
            json.dump(SINGLE_ROWS, f, indent=1)
    print("\n== summary: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    if hasattr(pose_src, "shim"):
        pose_src.shim.close()


if __name__ == "__main__":
    main()
