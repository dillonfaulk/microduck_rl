"""OmniHop viewer (2026-11-27): watch the hop policy in the CPU MuJoCo viewer (BAM, 1-step delay, training solver).

Run from the microduck_rl repo root (WITHOUT the PRIME-offload prefix -- it blanks the MuJoCo viewer):
  uv run python scripts/hop_viewer.py <omnihop.onnx> [--velstand <velstand.onnx>] [--settle] [--auto N] [--drift]

Runtime contract, as the mission stack would run it: VelStand (default: shipped 4250) holds the stand; on a
hop request the hop ONNX swaps in with the target in the twist slots (stance-midpoint remaining offset, trunk
yaw frame) and the 0.6 s request pulse in body slot 0 -- immediately (runtime default) or after the 0.25 s
settle (--settle, the fallback) -- then after 2.5 s VelStand swaps back in.

Keys (type in THIS terminal): 0 = in place, 8 = forward, 2 = back, 4 = left, 6 = right (numpad layout),
A = auto-cycle all five on/off, S = immediate / settle request, P = random push (0.3 m/s), Q = quit.
--auto N runs N full cycles of the five directions and exits (recording-friendly). The green disc is the
hop target under the feet' start midpoint + offset; the grey disc is the start midpoint.
"""
import argparse
import math
import os
import random
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hop_battery as hb  # noqa: E402
import infer_policy as ip  # noqa: E402

DT = 0.02
KEYS = {"0": "in_place", "8": "fwd", "2": "back", "4": "left", "6": "right"}
TARGETS = {"in_place": (0.0, 0.0), "fwd": (0.03, 0.0), "back": (-0.03, 0.0), "left": (0.0, 0.02), "right": (0.0, -0.02)}
ORDER = ("in_place", "fwd", "back", "left", "right")
HOP_S, REST_S = 2.5, 1.5


def disc(scn, pos, rgba, r=0.012):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([r, r, 0.0008]),
                        np.array([pos[0], pos[1], 0.001]), np.eye(3).flatten(), np.array(rgba, np.float32))
    scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--velstand", default=hb.DEFAULT_VELSTAND)
    ap.add_argument("--settle", action="store_true", help="request after the 0.25 s settle (fallback path)")
    ap.add_argument("--auto", type=int, default=0, help="run N cycles of the five directions, then exit")
    ap.add_argument("--drift", action="store_true", help="apply the drift model (scale bias + gyro bias)")
    a = ap.parse_args()

    hs = hb.HopSim(os.path.abspath(a.onnx), os.path.abspath(a.velstand))
    sim = hs.sim
    sim.reset(0.0, 0.0, 0.0)
    sim.policy.ort_session = hs.prestand
    sim.set_twist((0.0, 0.0, 0.0))
    delay = hb.SETTLE_S if a.settle else hb.RUNTIME_PULSE_DELAY
    auto = False                      # A toggles continuous cycling; --auto N uses the queue below
    queue = [d for _ in range(a.auto) for d in ORDER] if auto else []
    state = {"mode": "rest", "t": 0.0, "dir": None, "goal": None, "start": None, "yaw0": 0.0, "k": 0.0, "done": 0}
    print(__doc__.split("Keys")[1].split("--auto")[0].strip())
    print(f"request: {'settle 0.25 s' if delay else 'immediate'}; drift {'ON' if a.drift else 'OFF'}")

    def start_hop(d):
        (_, _, yaw0), _, _ = sim.pose()
        x0, y0 = hs.feet_mid()
        t = TARGETS[d]
        c, s = math.cos(yaw0), math.sin(yaw0)
        state.update(mode="hop", t=0.0, dir=d, start=(x0, y0), yaw0=yaw0,
                     goal=(x0 + c * t[0] - s * t[1], y0 + s * t[0] + c * t[1]),
                     k=float(np.clip(np.random.normal(-0.019, 0.006), -0.05, 0.02)) if a.drift else 0.0)
        hs.gyro_bias[:] = np.random.uniform(-0.03, 0.03, 3) if a.drift else 0.0
        sim.policy.ort_session = hs.hop_session
        print(f"[{sim.t:6.2f}s] HOP {d:8s} target {[round(1000 * v) for v in t]} mm", flush=True)

    def end_hop():
        x, y = hs.feet_mid()
        x0, y0 = state["start"]
        dx, dy = hb.to_frame(x - x0, y - y0, state["yaw0"])
        print(f"[{sim.t:6.2f}s]   landed: feet moved [{1000 * dx:+.1f}, {1000 * dy:+.1f}] mm -> VelStand", flush=True)
        sim.policy.ort_session = hs.prestand
        sim.set_twist((0.0, 0.0, 0.0))
        hs.gyro_bias[:] = 0.0
        state.update(mode="rest", t=0.0, done=state["done"] + 1)

    with ip.TerminalInput() as term, \
            mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.trunk_body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.9, -25, 135
        while viewer.is_running():
            t0 = time.time()
            keys = term.get_keys()
            if "q" in keys:
                break
            for key in keys:
                if key == "a":
                    auto = not auto
                    print(f"auto-cycle {'ON' if auto else 'OFF'}")
                elif key == "s":
                    delay = hb.RUNTIME_PULSE_DELAY if delay else hb.SETTLE_S
                    print(f"request: {'settle 0.25 s' if delay else 'immediate'}")
                elif key == "p":
                    ang = random.uniform(0, 2 * math.pi)
                    sim.push(0.3 * math.cos(ang), 0.3 * math.sin(ang))
                    print(f"[{sim.t:6.2f}s] push 0.3 m/s at {math.degrees(ang):.0f} deg")
                elif key in KEYS and state["mode"] == "rest":
                    start_hop(KEYS[key])
            if state["mode"] == "rest" and state["t"] >= REST_S and (auto or queue):
                if queue:
                    start_hop(queue.pop(0))
                elif auto:
                    start_hop(ORDER[state["done"] % len(ORDER)])
            if state["mode"] == "hop":
                (_, _, yaw), _, _ = sim.pose()
                fx, fy = hs.feet_mid()
                x0, y0 = state["start"]
                tx, ty = hb.to_frame(state["goal"][0] - x0, state["goal"][1] - y0, yaw)
                dx, dy = hb.to_frame(fx - x0, fy - y0, yaw)
                k = state["k"]
                hs.set_cmd((tx - (1 + k) * dx, ty - (1 + k) * dy), delay <= state["t"] < delay + hb.PULSE_S)
            sim.step()
            state["t"] += DT
            if state["mode"] == "hop" and state["t"] >= HOP_S:
                end_hop()
            if a.auto and not queue and state["mode"] == "rest" and state["t"] >= REST_S and state["done"] >= 5 * a.auto:
                break
            viewer.user_scn.ngeom = 0
            if state["start"] is not None:
                disc(viewer.user_scn, state["start"], (0.6, 0.6, 0.6, 0.8))
                disc(viewer.user_scn, state["goal"], (0.1, 0.9, 0.2, 0.9))
            label = f"{state['mode'].upper()} {state['dir'] or ''}  ({'settle' if delay else 'immediate'})"
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, label, ""))
            viewer.sync()
            time.sleep(max(0.0, DT - (time.time() - t0)))
    print(f"done: {state['done']} hops")


if __name__ == "__main__":
    main()
