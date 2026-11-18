"""Continuous-play viewer (play.py --viewer). Same runs as the battery, real time, with keys.

Keys (type in THIS terminal): C = duck-cam / V = free view, P = push (1 m/s), F = knock it over (2 m/s
sideways, the fall protocol then runs), R = pick a STUCK duck up (after a get-up failure: viewer only, counted),
N = next run, Q = quit.
Overlays (free view): blue disk = where the supervisor thinks the duck is (RED while not localized after a
fall); yellow arrow = the intended kick line; cyan line = the perceived goal; green disk = stance target.
Launch WITHOUT the PRIME-offload variables (the viewer renders on GPU1; with them the window stays blank).
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np
import onnxruntime as ort

import infer_policy as ip
import kick_mission as km
import play as pl
from goto_point import CONTROL_DT
from pose_provider import make_provider


def run(args):
    import mujoco.viewer
    from camera_ball import CameraBall
    from camera_goal import CameraGoal
    sim, goals, lm = pl.setup_play(args)
    kick, precision = ort.InferenceSession(args.kick), ort.InferenceSession(args.precision)
    pose_src = make_provider(args.pose, sim, imu_error=not args.no_imu_error, calib=args.odom_calib)
    ball_cam, goal_cam = CameraBall(sim, pose_src, profile=args.profile), CameraGoal(sim, pose_src, profile=args.profile)
    drag = pl.BallDrag(sim, args.ball_decel)
    head_cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    out = km.unique_dir(f"logs/play/{time.strftime('%Y-%m-%d_%H-%M-%S')}_viewer")
    print(f"\nKeys (type in THIS terminal): C = duck-cam / V = free view, P = push, F = knock over (fall protocol), "
          f"R = pick up a stuck duck (viewer only), "
          f"N = next run, Q = quit\ntime cap {args.time_cap:.0f} s per run, attack {args.attack}, pose {args.pose}, "
          f"ball decel {args.ball_decel} m/s^2\nlogs -> {out}/\n", flush=True)
    st = {"quit": False, "next": False, "cam": False, "seen": (0, 0, 0, 0, 0, 0)}
    with ip.TerminalInput() as term, \
            mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.trunk_body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.6, -35, 135
        viewer.opt.geomgroup[5] = 1              # the 4 m field's transparent boundary wall (hidden from the duck's camera)

        def set_view(on):
            with viewer.lock():
                if on:
                    viewer.cam.type, viewer.cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, head_cam
                else:
                    viewer.cam.type, viewer.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, sim.trunk_body
            print(f"  view: {'DUCK CAM' if on else 'FREE'}", flush=True)

        def cb(sim, sup, rec):
            t0 = time.time()
            for key in term.get_keys():
                if key == "q":
                    st["quit"] = True
                elif key == "n":
                    st["next"] = True
                elif key == "c":
                    st["cam"] = not st["cam"]
                    set_view(st["cam"])
                elif key == "v" and st["cam"]:
                    st["cam"] = False
                    set_view(False)
                elif key == "p":
                    a = np.random.uniform(0, 2 * math.pi)
                    sim.push(math.cos(a), math.sin(a))
                    sup.log.event({"t": round(sim.t, 3), "event": "PUSH"})
                elif key == "r" and sup.state == "STUCK":
                    pl.stand_up_in_place(sim)          # VIEWER ONLY: the human pick-up (counted)
                    sup.picked_up()
                    print(f"  t {sim.t:6.1f} s  picked up and stood in place (viewer reset #{sup.viewer_resets}; "
                          f"the get-up failure stays counted)", flush=True)
                elif key == "f":
                    (_, _, yaw), _, _ = sim.pose()
                    a = yaw + math.pi / 2 * (1 if np.random.rand() < 0.5 else -1)
                    sim.push(2.0 * math.cos(a), 2.0 * math.sin(a))
                    sup.log.event({"t": round(sim.t, 3), "event": "KNOCK_OVER"})
            if sup.state == "STUCK" and not st.get("stuck_said"):
                st["stuck_said"] = True
                print(f"  t {sim.t:6.1f} s  GET-UP FAILURE (down > {pl.GETUP_FAIL_S:.0f} s; counted). Press R to pick it up.",
                      flush=True)
            elif sup.state != "STUCK":
                st["stuck_said"] = False
            seen = (rec["goals_for"], rec["goals_against"], rec["out"], len(sup.kicks), len(sup.relocs), len(sup.routine))
            if seen != st["seen"]:
                if seen[:3] != st["seen"][:3]:
                    print(f"  t {sim.t:6.1f} s  SCORE: goals {rec['goals_for']} (attacking {rec['attack']}), other end "
                          f"{rec['goals_against']}, out {rec['out']}", flush=True)
                if seen[3] != st["seen"][3]:
                    k = sup.kicks[-1]
                    print(f"  t {sim.t:6.1f} s  kick {seen[3]} ({k.get('aim_mode')} aim, aim err {k.get('aim_err_deg')} deg)",
                          flush=True)
                if seen[4] != st["seen"][4]:
                    x = sup.relocs[-1]
                    print(f"  t {sim.t:6.1f} s  RE-LOCALIZED after a fall: {x.get('source')}, pose error "
                          f"{x.get('err_before_mm')} -> {x.get('err_after_mm')} mm in {x.get('protocol_s')} s", flush=True)
                if seen[5] != st["seen"][5]:
                    x = sup.routine[-1]
                    print(f"  t {sim.t:6.1f} s  routine landmark fix (goal {x['landmark']}): {x['err_before_mm']} -> "
                          f"{x['err_after_mm']} mm", flush=True)
                st["seen"] = seen
            if st["cam"]:
                viewer.user_scn.ngeom = 0
            else:
                if sup.mission is not None and sup.state == "MISSION":
                    km.draw_mission(viewer.user_scn, sup.mission, sim)
                else:
                    viewer.user_scn.ngeom = 0
                e = sup.pose_src.get(sim.t)
                if e.pose is not None and viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    rgba = [0.2, 0.4, 1.0, 0.8] if sup.localized else [1.0, 0.15, 0.1, 0.9]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.035, 0.004, 0]),
                                        np.array([e.pose[0], e.pose[1], 0.006]), np.eye(3).flatten(), np.array(rgba, np.float32))
                    viewer.user_scn.ngeom += 1
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              f"run {rec['run']}  attack {rec['attack']}  t {sim.t:5.1f}",
                              f"goals {rec['goals_for']}  other {rec['goals_against']}  out {rec['out']}  kicks {len(sup.kicks)}"
                              f"  {sup.state}{'' if sup.localized else '  NOT LOCALIZED'}"))
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
            if st["next"]:
                st["next"] = False
                return False
            return viewer.is_running() and not st["quit"]

        for k in range(args.runs):
            rng_k = np.random.default_rng([args.seed, k])
            st["seen"] = (0, 0, 0, 0, 0, 0)
            r = pl.run_one(args, k, sim, goals, lm, kick, precision, pose_src, ball_cam, goal_cam, rng_k, drag, out, viewer=cb)
            print(pl.fmt_run(r), flush=True)
            if st["quit"] or not viewer.is_running():
                break
    if hasattr(pose_src, "shim"):
        pose_src.shim.close()
    print(f"logs: {out}/")
