"""Where the supervisor thinks the duck is (rung 6 Stage 2): the estimated-pose provider seam.

The mission supervisors (goto_point / GotoSupervisor, kick_mission.Mission) take every pose and
ball reading through a provider, never from `sim.pose()` / `sim.ball()` directly. Stage 3 swaps
the ball source for the camera, and a later stage drops the real `mapd` behind the same seam,
without touching the call sites. Ground truth stays the scoring reference only.

Providers:
- `GroundTruthProvider` returns exactly what the missions read before: the same calls, the same
  arrays. Ground-truth runs are byte-identical to before the seam.
- `OdometryProvider` (this stage, odometry-only, no map correction) is the robot's contact
  odometry, `infer_policy.PyOdometry`, a line-for-line replica of
  `microduck/odometry/src/lib.rs` on the upstream alpha16 anchor set. It runs at 50 Hz on:
  - foot poses relative to the trunk. That relative geometry is what forward kinematics gives
    the real robot from its joint encoders.
  - the IMU's trunk rotation: true roll/pitch, and a heading with `ImuYawError` applied.
  - It is anchored at a known start pose (a kickoff spot); the error is drift since then.

Output speaks `mapd`'s `MapState` vocabulary (duck-ipc-proto lib.rs:6090-6109): `{t_ns,
pose | absent, lost, ...}` plus an explicit `age_s`. A pose older than `STALE_S` is reported
ABSENT, with `lost = "stale"`, never silently held. `lost` is `"boot"` before the first sample.
`mapd` documents that a client must treat any word it doesn't know as lost.

A dropout window models the pose LINK going down: no estimates reach the supervisor. The
robot's own odometry keeps integrating onboard, as `robotd`'s does, so the first estimate after
the link returns is current.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STALE_S = 0.10          # 5 control ticks; GotoSupervisor's own watchdog is 0.2 s
VEL_WINDOW_S = 0.10     # odometry velocity: displacement over the last 0.1 s

# Versioned odometry calibrations (rung 6 Stage 3, Workstream A; Dillon approved 2026-11-14).
# A calibration scales the odometry's planar displacement since the anchor; the estimator itself
# (PyOdometry, a replica of the Rust crate) is untouched.
# - "none": raw contact odometry (Stage 2).
# - "stride_v1": +1.9 %. Measured basis is Stage 2's isolated straight walks: 3.1 m with a
#   perfect IMU and no supervisor ends -1.74 / -1.86 / -1.98 % short along-track over three
#   headings, under the round-3 VelStand gait on the alpha16 anchor set.
ODOM_CALIBS = {"none": 1.0, "stride_v1": 1.019}


@dataclass
class PoseEstimate:
    t: float
    pose: tuple | None              # (x, y, yaw) in the field/world frame, or None (absent)
    lost: str | None                # None while known; "boot" | "stale" | a mapd cause
    age_s: float
    tilt_deg: float | None = None
    trunk_z: float | None = None
    v_xy: np.ndarray | None = None  # trunk planar velocity, world frame

    def map_state(self) -> dict:
        """mapd's MapState shape (plus age_s), for logs."""
        d = {"t_ns": int(round(self.t * 1e9)), "age_s": round(self.age_s, 3)}
        if self.pose is not None:
            d["pose"] = {"x": round(self.pose[0], 4), "y": round(self.pose[1], 4), "yaw": round(self.pose[2], 4)}
        if self.lost is not None:
            d["lost"] = self.lost
        return d


class ImuYawError:
    """Heading error of an integrated gyro: a constant bias plus a random walk.

    Defaults are measured on the real duck by `scripts/odom_dataset_check.py`, over 20 s windows
    of walking in Pollen's `SLAM_JJ_Salon_with_GT` s2 + s3 (19 windows):
    - The signed fit gives a bias of +2.02 deg/min, plus a residual of 2.14 deg std per window.
    - Removing the ground truth's own heading noise (~1 deg per pose, so ~1.4 deg on a
      difference) in quadrature leaves 1.62 deg per 20 s, a random walk of 0.36 deg/sqrt(s).
    - The bias sign is drawn per run, because one robot's sign is not every robot's.
    - The model reproduces the measured median |heading error| of 3.3 deg/min:
      0.67 * sqrt(0.67^2 + 1.62^2) = 1.17 deg per 20 s.
    - Turn-proportional (scale) error was fitted at -0.3 % of net turn: not modelled.
    """

    BIAS_DEG_PER_MIN = 2.0
    RW_DEG_PER_SQRT_S = 0.36

    def __init__(self, seed: int = 0, bias_deg_per_min: float | None = None,
                 rw_deg_per_sqrt_s: float | None = None, enabled: bool = True):
        self.enabled = enabled
        self.bias_mag = self.BIAS_DEG_PER_MIN if bias_deg_per_min is None else bias_deg_per_min
        self.rw = self.RW_DEG_PER_SQRT_S if rw_deg_per_sqrt_s is None else rw_deg_per_sqrt_s
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(10_000 + seed)
        self.sign = 1.0 if self.rng.random() < 0.5 else -1.0
        self.err = 0.0                      # rad

    @property
    def bias_rad_s(self) -> float:
        return self.sign * math.radians(self.bias_mag) / 60.0

    def step(self, dt: float) -> float:
        if self.enabled:
            self.err += self.bias_rad_s * dt + math.radians(self.rw) * math.sqrt(dt) * self.rng.standard_normal()
        return self.err


class _Provider:
    name = "base"

    def __init__(self, dropouts=()):
        self.dropouts = list(dropouts)      # [(t_start, t_end)] link down, sim time
        self._last = None                   # last delivered PoseEstimate

    def link_down(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.dropouts)

    def _deliver(self, est: PoseEstimate) -> PoseEstimate:
        if not self.link_down(est.t):
            self._last = est
            return est
        if self._last is None:
            return PoseEstimate(est.t, None, "boot", math.inf)
        age = est.t - self._last.t
        if age > STALE_S:
            return PoseEstimate(est.t, None, "stale", age)
        return PoseEstimate(self._last.t, self._last.pose, None, age, self._last.tilt_deg,
                            self._last.trunk_z, self._last.v_xy)



class GroundTruthProvider(_Provider):
    """Today's reads, through the seam: `sim.pose()` and the trunk's qvel, computed at call time."""

    name = "gt"

    def __init__(self, sim, dropouts=()):
        super().__init__(dropouts)
        self.sim = sim

    def reset(self, sim=None, seed: int = 0) -> None:
        self._last = None

    def update(self, sim=None) -> None:
        pass

    def get(self, t: float) -> PoseEstimate:
        s = self.sim
        pose, tilt, z = s.pose()
        est = PoseEstimate(t, pose, None, 0.0, tilt, z, s.data.qvel[s.va:s.va + 2])
        return est if not self.dropouts else self._deliver(est)


class OdometryProvider(_Provider):
    """Contact odometry + IMU (with heading error), anchored at a known start pose."""

    name = "odom"

    def __init__(self, sim, anchor_set: str = "alpha16", imu: ImuYawError | None = None, dropouts=(), dt=0.02,
                 calib: str = "none"):
        super().__init__(dropouts)
        if calib not in ODOM_CALIBS:
            raise SystemExit(f"unknown odometry calibration {calib!r} ({' | '.join(ODOM_CALIBS)})")
        self.calib = calib
        self.stride_scale = ODOM_CALIBS[calib]
        self.offset = None                  # map<->odom correction (x, y, yaw), applied on top; None = none
        import infer_policy as ip
        self._ip = ip
        self.sim = sim
        self.frames = ip.FootFrames(sim.model)
        self.corners = ip.load_odom_anchor_sets(anchor_set)[anchor_set]
        self.anchor_set = anchor_set
        self.imu = ImuYawError() if imu is None else imu
        self.dt = dt
        self.reset(sim)

    def reset(self, sim=None, seed: int = 0) -> None:
        """(Re-)anchor at the duck's current TRUE pose: the known start of a mission."""
        sim = sim or self.sim
        self.imu.reset(seed)
        self.odo = self._ip.PyOdometry(self.anchor_set, self.corners)
        p_t, _ = self.frames.trunk_world(sim.data)
        self.p0 = p_t[:2].copy()
        self.hist = deque()
        self._last = None
        self._est = None
        self._integrate(sim, first=True)

    def _integrate(self, sim, first=False):
        data = sim.data
        _, R_true = self.frames.trunk_world(data)
        e = self.imu.err if first else self.imu.step(self.dt)
        c, s = math.cos(e), math.sin(e)
        R_imu = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R_true
        self.odo.update(self.frames.feet_in_trunk(data), R_imu)
        self._R_imu = R_imu
        xy = self.p0 + self.stride_scale * self.odo.position[:2]
        yaw = math.atan2(R_imu[1, 0], R_imu[0, 0])
        self._odom_raw = (float(xy[0]), float(xy[1]), float(yaw))
        if self.offset is not None:
            xy, yaw = apply_offset(self.offset, xy, yaw)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, R_imu[2, 2]))))
        t = sim.t
        self.hist.append((t, xy.copy()))
        while len(self.hist) > 1 and t - self.hist[0][0] > VEL_WINDOW_S + 1e-9:
            self.hist.popleft()
        t0, xy0 = self.hist[0]
        v = (xy - xy0) / (t - t0) if t > t0 else np.zeros(2)
        self._est = PoseEstimate(t, (float(xy[0]), float(xy[1]), float(yaw)), None, 0.0, tilt,
                                 float(self.odo.position[2]), v)

    def update(self, sim=None) -> None:
        """Call once per control tick, after sim.step(): the robot's odometry never stops."""
        self._integrate(sim or self.sim)

    def get(self, t: float) -> PoseEstimate:
        if self._est is None:
            return PoseEstimate(t, None, "boot", math.inf)
        est = PoseEstimate(t, self._est.pose, None, t - self._est.t, self._est.tilt_deg,
                           self._est.trunk_z, self._est.v_xy)
        if est.age_s > STALE_S:
            return PoseEstimate(t, None, "stale", est.age_s)
        return self._deliver(est)


class GroundTruthBall:
    """The ball, through the seam (ground truth this stage; the camera channel is Stage 3)."""

    name = "gt"

    def __init__(self, sim):
        self.sim = sim

    def get(self, t: float = 0.0):
        return self.sim.ball()


def apply_offset(off, xy, yaw):
    """Map pose = offset ∘ odom pose (autod's layering: map->odom rigid transform)."""
    ox, oy, oyaw = off
    c, s = math.cos(oyaw), math.sin(oyaw)
    x, y = float(xy[0]), float(xy[1])
    return np.array([ox + c * x - s * y, oy + s * x + c * y]), yaw + oyaw


def make_provider(kind: str, sim, seed: int = 0, dropouts=(), imu_error: bool = True, calib: str = "none"):
    if kind == "gt":
        return GroundTruthProvider(sim, dropouts)
    if kind == "odom":
        return OdometryProvider(sim, imu=ImuYawError(seed, enabled=imu_error), dropouts=dropouts, calib=calib)
    if kind == "mapd":   # Stage 4: odometry + the real mapd's map<-odom offset (mapd_shim.py); close() when done
        import mujoco
        from mapd_shim import MapdProvider, MapdShim
        from mjlab_microduck.sim.tof import Tof
        tof = Tof(sim.model, mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "tof"), seed=seed)
        return MapdProvider(sim, MapdShim(), tof, imu=ImuYawError(seed, enabled=imu_error), dropouts=dropouts,
                            calib=calib)
    raise SystemExit(f"unknown pose source {kind!r} (gt | odom | mapd)")


def pose_error(est_pose, true_pose):
    """(xy error m, signed yaw error rad), estimate minus truth."""
    dx, dy = est_pose[0] - true_pose[0], est_pose[1] - true_pose[1]
    dyaw = (est_pose[2] - true_pose[2] + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(dx, dy), dyaw
