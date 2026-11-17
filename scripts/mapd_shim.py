"""Rung 6 Stage 4, Workstream B: the real, unmodified `mapd` in the sim's loop.

`mapd` is a client of two daemons: `robotd` (`robot.subscribe` -> `robot.state` at [map] state_hz,
and `robot.mode` for the "duck on its own" autonomy gate) and `tofd` (`tof.stream` -> `tof.frame`).
This module plays both, on Unix sockets in duck-ipc-proto's wire format (newline-delimited JSON-RPC
2.0), feeding mapd the twin's odometry, IMU, ToF pose and ToF frames. mapd's answer (`map.status`:
pose-or-absent + lost cause, no uncertainty) comes back on its own socket and reaches the pose
provider as autod uses it: an OFFSET map<-odometry, layered on the 50 Hz odometry, never a pose
replacement (MapdProvider below).

Lockstep, so a battery is deterministic: every input carries a `t_ns`, mapd's worker stamps
`map.status.state.t_ns` with the last input it processed, and the shim waits for that stamp after
every input before the sim steps on. That also serialises mapd's two feed threads (robot vs ToF),
which otherwise race. A ToF frame is stamped 1 ns after the state of the same tick so the two are
told apart. The clock is sim time (+1 s, so it is never zero), monotonic across a whole battery.

One mapd process per battery; `map.wipe` between runs (a fresh map is Tracking with identity
map<-odom: the map frame starts as the anchored odometry frame) and a 2 s clock gap so no history
bracket spans two runs.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time

import mujoco
import numpy as np

from pose_provider import OdometryProvider, apply_offset

MAPD_BIN = os.path.expanduser("~/microduck_runtime/target/release/mapd")
STATE_HZ = 25                 # [map] state_hz default; robotd decimates robot.subscribe to this
TOF_SENSOR = "VL53L8CX"       # face-board-v1 (GC2093 board): the Stage 2 profile's ToF
FALLEN_TILT_DEG = 60.0        # the supervisor's own fall rule, on the estimated tilt (never sim state)
BARRIER_TIMEOUT_S = 10.0
RUN_GAP_NS = 2_000_000_000


def _line(obj) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def notify(method: str, params: dict) -> bytes:
    return _line({"jsonrpc": "2.0", "method": method, "params": params})


def robot_state(t_ns: int, t_s: float, trunk_pos, odom_yaw: float, imu_quat, gyro, tof_pos, tof_quat,
                requested, fallen: bool, head=(0.0, 0.0, 0.0, 0.0)) -> dict:
    """A `robot.state` with every field the RobotState struct requires, plus what maploc reads
    (t_ns, odom, imu, frames, move.requested, policy, safety). Joints are not reported (empty)."""
    return {
        "t": t_s,
        "move": {"requested": [float(v) for v in requested], "applied": [float(v) for v in requested]},
        "head": [float(v) for v in head],
        "policy": "walk",
        "safety": {"fallen": bool(fallen), "limp": False, "gravity": [0.0, 0.0, -1.0], "gain": None,
                   "picked_up": False},
        "loop": {"hz": 50.0, "missed": 0},
        "joints": [], "targets": [],
        "odom": {"position": [float(v) for v in trunk_pos], "yaw": float(odom_yaw)},
        "autonomous": True,
        "t_ns": int(t_ns),
        "imu": {"gyro": [float(v) for v in gyro], "quat": [float(v) for v in imu_quat]},
        "frames": {"camera": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
                   "tof": {"pos": [float(v) for v in tof_pos], "quat": [float(v) for v in tof_quat]}},
    }


def tof_frame(t_ns: int, seq: int, distance_mm, status) -> dict:
    return {"seq": int(seq), "at_us": int(t_ns // 1000), "t_ns": int(t_ns), "rows": 8, "cols": 8,
            "distance_mm": [int(max(-32768, min(32767, d))) for d in distance_mm],
            "status": [int(s) for s in status]}


def mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


class _Server(threading.Thread):
    """One daemon's socket: answers calls, keeps subscribers to push notifications to."""

    def __init__(self, path: str, handle):
        super().__init__(daemon=True)
        self.path, self.handle = path, handle
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(8)
        self.subscribers: list[socket.socket] = []
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def run(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        f = conn.makefile("rb")
        try:
            for raw in f:
                req = json.loads(raw)
                self.calls.append(req.get("method", "?"))
                result, subscribe = self.handle(req.get("method"), req.get("params"))
                if result is None:
                    resp = {"jsonrpc": "2.0", "id": req.get("id"),
                            "error": {"code": -32601, "message": f"shim does not serve {req.get('method')}"}}
                else:
                    resp = {"jsonrpc": "2.0", "id": req.get("id"), "result": result}
                conn.sendall(_line(resp))
                if subscribe:
                    with self.lock:
                        self.subscribers.append(conn)
                    return          # the subscriber's connection now only receives
        except (OSError, ValueError):
            pass

    def push(self, data: bytes) -> int:
        with self.lock:
            alive = []
            for c in self.subscribers:
                try:
                    c.sendall(data)
                    alive.append(c)
                except OSError:
                    pass
            self.subscribers = alive
            return len(alive)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class MapdShim:
    """Run unmodified mapd against shim robotd/tofd sockets; feed it in lockstep."""

    def __init__(self, mapd_bin: str = MAPD_BIN, autonomous: bool = True, log_path: str | None = None):
        if not os.path.exists(mapd_bin):
            raise SystemExit(f"mapd not built: {mapd_bin} (cargo build --locked --release -p mapd in ~/microduck_runtime)")
        self.dir = tempfile.mkdtemp(prefix="mapd_shim_")
        self.autonomous = autonomous
        self.robot = _Server(os.path.join(self.dir, "robotd.sock"), self._robot_call)
        self.tof = _Server(os.path.join(self.dir, "tof.sock"), self._tof_call)
        self.robot.start()
        self.tof.start()
        cfg = os.path.join(self.dir, "robotd.toml")
        with open(cfg, "w") as f:      # mapping OFF in the file: it runs because the duck is on its own (the gate)
            f.write(f"[map]\nenabled = false\nstate_hz = {STATE_HZ}\ntof_latency_ms = 60.0\n")
        self.log = open(log_path or os.path.join(self.dir, "mapd.log"), "w")
        env = dict(os.environ, RUST_LOG=os.environ.get("MAPD_LOG", "info"))
        self.proc = subprocess.Popen(
            [mapd_bin, "--config", cfg, "--socket", os.path.join(self.dir, "map.sock"),
             "--robot-socket", self.robot.path, "--tof-socket", self.tof.path,
             "--map", os.path.join(self.dir, "map.json")],
            stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.t_base_ns = 1_000_000_000
        self.last_ns = 0
        self.seq = 0
        self.inputs = 0
        self._wait_connected()

    # --- the two daemons' calls -----------------------------------------------------------------
    def _robot_call(self, method, params):
        if method == "robot.mode":
            return ({"mode": "walk", "autonomous": True} if self.autonomous else {"mode": "walk"}), False
        if method == "robot.subscribe":
            return {"subscribed": True}, True
        return None, False

    def _tof_call(self, method, params):
        if method == "tof.stream":
            return {"accepted": True, "sensor": TOF_SENSOR, "rows": 8, "cols": 8, "hz": 15}, True
        return None, False

    def _wait_connected(self, timeout=15.0):
        t0 = time.time()
        while not (self.robot.subscribers and self.tof.subscribers):
            if self.proc.poll() is not None:
                raise RuntimeError(f"mapd exited ({self.proc.returncode}); log {self.log.name}")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"mapd did not subscribe in {timeout} s (calls seen: robot {self.robot.calls}, "
                                   f"tof {self.tof.calls}); log {self.log.name}")
            time.sleep(0.01)
        self.map = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        p = os.path.join(self.dir, "map.sock")
        while True:
            try:
                self.map.connect(p)
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise
                time.sleep(0.01)
        self.mapf = self.map.makefile("rb")
        self._id = 10

    # --- map.* ------------------------------------------------------------------------------------
    def call(self, method: str, params=None) -> dict:
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            req["params"] = params
        self.map.sendall(_line(req))
        resp = json.loads(self.mapf.readline())
        if resp.get("error"):
            raise RuntimeError(f"{method}: {resp['error']}")
        return resp["result"]

    def status(self) -> dict:
        return self.call("map.status")

    def _barrier(self, t_ns: int) -> dict:
        """Wait until mapd's worker has processed the input stamped t_ns."""
        t0 = time.time()
        while True:
            st = self.status()
            if st["state"]["t_ns"] == t_ns:
                return st
            if time.time() - t0 > BARRIER_TIMEOUT_S:
                raise RuntimeError(f"mapd never processed input t_ns={t_ns} (status {st}); log {self.log.name}")

    # --- feeding --------------------------------------------------------------------------------
    def clock(self, t_s: float) -> int:
        return self.t_base_ns + int(round(t_s * 1e9))

    def send_state(self, state: dict) -> dict:
        assert state["t_ns"] > self.last_ns, "the state stream must be monotonic"
        if not self.robot.push(notify("robot.state", state)):
            raise RuntimeError("mapd dropped its robot.subscribe")
        self.last_ns = state["t_ns"]
        self.inputs += 1
        return self._barrier(state["t_ns"])

    def send_tof(self, t_ns: int, distance_mm, status) -> dict:
        self.seq += 1
        if not self.tof.push(notify("tof.frame", tof_frame(t_ns, self.seq, distance_mm, status))):
            raise RuntimeError("mapd dropped its tof.stream")
        self.inputs += 1
        return self._barrier(t_ns)

    def new_run(self) -> None:
        """A fresh map for the next run, and a clock gap so no history bracket spans the two."""
        self.call("map.wipe")
        self.t_base_ns = self.last_ns + RUN_GAP_NS

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for s in (self.robot, self.tof):
            s.close()
        try:
            self.map.close()
        except Exception:
            pass
        self.log.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def offset_from(map_pose, odom_pose):
    """autod's map::offset: the rigid transform with map = offset ∘ odom."""
    oyaw = map_pose[2] - odom_pose[2]
    c, s = math.cos(oyaw), math.sin(oyaw)
    return (map_pose[0] - (c * odom_pose[0] - s * odom_pose[1]),
            map_pose[1] - (s * odom_pose[0] + c * odom_pose[1]), oyaw)


class MapdProvider(OdometryProvider):
    """Odometry + IMU (+ stride calibration), corrected by the real mapd through the offset hook.

    Every tick: the odometry integrates (as OdometryProvider). Every 2nd tick (25 Hz) a robot.state
    goes to mapd; every ToF frame the twin's sensor delivers (15 Hz, 60 ms late, all Stage 2 flaws)
    goes as a tof.frame. When mapd reports a pose, offset = map ⊖ odom at that state's t_ns, and the
    estimate becomes offset ∘ odom from the next tick on. When mapd is lost or silent the last offset
    stays (autod: "carry on with the last offset"). mapd sees the RAW (uncorrected) odometry, always.
    """

    name = "mapd"

    def __init__(self, sim, shim: MapdShim, tof, **kw):
        self.shim, self.tof_sensor = shim, tof
        self.tof_site = tof.site
        self.mapd_log = []
        self._first = True
        super().__init__(sim, **kw)

    def reset(self, sim=None, seed: int = 0) -> None:
        if not self._first:
            self.shim.new_run()
        self._first = False
        self.offset = None
        self.tick = 0
        self.sent = {}                       # t_ns -> raw odom pose sent at that tick
        self.mapd_log = []
        self.last_status = None
        self.corrections = 0
        self.relocalizations, self._reloc0 = 0, None
        self.tof_sensor.random = np.random.default_rng(seed)
        self.tof_sensor._queue.clear()
        self.next_tof_t = 0.0
        super().reset(sim, seed)

    def _integrate(self, sim, first=False):
        super()._integrate(sim, first)
        if first:
            return
        self.tick += 1
        data = sim.data
        p_t, R_true = self.frames.trunk_world(data)
        x, y, yaw = self._odom_raw
        if self.tick % (50 // STATE_HZ) == 0:
            t_ns = self.shim.clock(sim.t)
            R_imu = self._R_imu
            tof_p = R_true.T @ (data.site_xpos[self.tof_site] - p_t)
            tof_R = R_true.T @ data.site_xmat[self.tof_site].reshape(3, 3)
            gyro = data.qvel[sim.va + 3:sim.va + 6]          # free-joint angular velocity: already body frame
            st = robot_state(t_ns, sim.t, (x, y, float(self.odo.position[2])), yaw, mat2quat(R_imu), gyro,
                             tof_p, mat2quat(tof_R), sim.policy.vel_cmd[:3], self._est.tilt_deg > FALLEN_TILT_DEG,
                             head=getattr(sim.policy, "head_offset", np.zeros(4))[:4])
            self.sent[t_ns] = (x, y, yaw)
            if len(self.sent) > 64:
                self.sent.pop(next(iter(self.sent)))
            self._take(self.shim.send_state(st), sim.t)
        if sim.t + 1e-9 < self.next_tof_t:
            return
        self.next_tof_t += 1.0 / 15.0                   # the sensor's own 15 Hz, as Stage 2's Tof expects
        got = self.tof_sensor.frame_at(data, sim.t)
        if got is not None:
            dist, stat, _capture = got
            # stamped when read (as tofd does); mapd subtracts its tof_latency_ms (60) itself
            self._take(self.shim.send_tof(self.shim.clock(sim.t) + 1, dist, stat), sim.t)

    def _take(self, status: dict, t: float) -> None:
        s = status["state"]
        self.last_status = s
        n = status.get("relocalizations", 0)
        if self._reloc0 is None:
            self._reloc0 = n
        if n - self._reloc0 > self.relocalizations:
            # mapd recognised where it is in its own map (after a fall: a stop matched the pre-fall
            # keyframes). Its map frame IS the field frame (anchored at the known kickoff pose), so its
            # pose is absolute: any landmark fix layered on top is superseded.
            self.relocalizations = n - self._reloc0
            self.mapd_log.append({"t": round(t, 3), "event": "relocalized", "keyframes": s.get("keyframes")})
            self.fix = None
        pose = s.get("pose")
        odom = self.sent.get(s["t_ns"]) or self.sent.get(s["t_ns"] - 1)
        if pose is not None and odom is not None:
            off = offset_from((pose["x"], pose["y"], pose["yaw"]), odom)
            if self.offset is None or any(abs(a - b) > 1e-6 for a, b in zip(off, self.offset)):
                if self.offset is not None or any(abs(v) > 1e-6 for v in off):
                    self.corrections += 1
                    self.mapd_log.append({"t": round(t, 3), "offset": [round(v, 4) for v in off],
                                          "keyframes": s.get("keyframes"), "lost": s.get("lost")})
                self.offset = off
        elif pose is None and (not self.mapd_log or self.mapd_log[-1].get("lost") != s.get("lost")):
            self.mapd_log.append({"t": round(t, 3), "lost": s.get("lost"), "keyframes": s.get("keyframes")})

    def summary(self) -> dict:
        s = self.last_status or {}
        return {"mapd_keyframes": s.get("keyframes"), "mapd_lost": s.get("lost"),
                "mapd_corrections": self.corrections, "mapd_relocalizations": self.relocalizations,
                "mapd_offset": None if self.offset is None else [round(v, 4) for v in self.offset]}
