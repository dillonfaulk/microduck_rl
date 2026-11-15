"""Camera object channel (rung 6 Stage 3, Workstream B): the ball measured in the duck's own frame.

A head-camera frame is rendered under a board profile (`sim/board_profiles.py`, face-board-v1 by
default), the ball is segmented by colour, and its position is ranged two ways:
- **ray -> plane** (primary, upstream's soccer method): the blob centroid's pixel ray, from the
  camera pose, intersected with the plane z = ball radius.
- **apparent size** (cross-check): range from the blob's equivalent diameter against the known
  70 mm ball, along the same ray.

**Intrinsics come from the rendered camera** (`pinhole_intrinsics`, i.e. the profile's fovy at the
render size), never a hard-coded field of view. The sim renders a pinhole; the profile's distortion
(GC2093 k1 -0.369) is carried for the arrival era and not undone here.

What the robot really knows, and what this module uses:
- camera pose relative to the trunk = head FK (joint encoders): read from the sim's camera frame;
- trunk roll/pitch from the IMU (accurate); trunk yaw is NOT used: the output is in the duck's yaw
  frame, so world heading error cancels;
- floor height: the flat floor at z = 0 (on hardware: stance-foot FK). Stated, not hidden.

`CameraBall` is a ball source for `kick_mission.Mission`: it reports the ball in the ESTIMATED
world frame, i.e. estimated pose ⊕ camera-relative ball. Every line-up quantity the mission derives
(ball - pose, rotated by estimated yaw) is then camera-relative: the world-pose error cancels.
Between looks the estimate is held in the estimated frame (it moves with odometry), and each new
look re-bases it.
"""

from __future__ import annotations

import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mjlab_microduck.sim import board_profiles  # noqa: E402
from mjlab_microduck.sim.camera import apply_profile, pinhole_intrinsics  # noqa: E402

BALL_RADIUS = 0.035
BALL_DIAMETER = 2 * BALL_RADIUS
MIN_PIXELS = 12             # smaller blobs are "not found"
LOOK_PERIOD_S = 0.2         # 5 Hz looks while approaching (the real mediad runs 15 fps)
EDGE_MARGIN_PX = 2          # a blob touching the frame edge is truncated: centroid unreliable


def orange_mask(rgb: np.ndarray) -> np.ndarray:
    """The sim ball's orange (rgba 1 0.55 0), lit: red high, green mid, blue low. The real duck's
    detector is tuned for a purple ball; colour is an arrival calibration item."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (r > 120) & (r - b > 80) & (g > 0.25 * r) & (g < 0.80 * r) & (b < 0.45 * r)


class BallCamera:
    """Renders the head camera under a board profile and measures the ball."""

    def __init__(self, model: mujoco.MjModel, profile: str = board_profiles.PRIMARY, width: int = 640,
                 height: int = 360, cam_name: str = "head_camera"):
        self.model = model
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self.profile = board_profiles.get(profile)
        apply_profile(model, self.cam, self.profile)
        self.rotate = self.profile.camera.rotate_deg
        self.w, self.h = width, height
        self.K = pinhole_intrinsics(model, self.cam, width, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.looks = 0

    def render(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.cam)
        return self.renderer.render()

    def pixel_ray_world(self, data, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """World origin and unit direction of pixel (u, v). MuJoCo cameras look down -z, +x right,
        +y up; image rows grow downward."""
        K = self.K
        d_cam = np.array([(u - K["cx"]) / K["fx"], -(v - K["cy"]) / K["fy"], -1.0])
        R = data.cam_xmat[self.cam].reshape(3, 3)
        d = R @ d_cam
        return data.cam_xpos[self.cam].copy(), d / np.linalg.norm(d)

    def measure(self, data, rgb: np.ndarray | None = None) -> dict | None:
        """Ball measurement, or None (not found). Positions in the duck's yaw frame, metres, from the
        trunk: {"ray": (x, y), "size": (x, y), "pixels": n, "edge": bool, "u", "v"}."""
        self.looks += 1
        rgb = self.render(data) if rgb is None else rgb
        m = orange_mask(rgb)
        n = int(m.sum())
        if n < MIN_PIXELS:
            return None
        vs, us = np.nonzero(m)
        edge = bool(us.min() < EDGE_MARGIN_PX or vs.min() < EDGE_MARGIN_PX
                    or us.max() >= self.w - EDGE_MARGIN_PX or vs.max() >= self.h - EDGE_MARGIN_PX)
        u, v = float(us.mean()) + 0.5, float(vs.mean()) + 0.5
        o, d = self.pixel_ray_world(data, u, v)
        out = {"pixels": n, "edge": edge, "u": u, "v": v}
        if d[2] < -1e-3:                          # ray -> plane z = ball radius (floor at 0)
            s = (BALL_RADIUS - o[2]) / d[2]
            out["ray_w"] = o + s * d
        # apparent size: equivalent diameter in pixels -> range along the ray to the ball centre
        diam_px = 2.0 * math.sqrt(n / math.pi)
        rng = self.K["fx"] * BALL_DIAMETER / max(diam_px, 1e-6)
        out["size_w"] = o + rng * d
        R_t = data.xmat[self.trunk].reshape(3, 3)
        p_t = data.xpos[self.trunk]
        yaw = math.atan2(R_t[1, 0], R_t[0, 0])  # only to express the result in the yaw frame
        c, s_ = math.cos(yaw), math.sin(yaw)
        for k in ("ray_w", "size_w"):
            if k in out:
                dd = out[k][:2] - p_t[:2]
                out[k[:-2]] = (c * dd[0] + s_ * dd[1], -s_ * dd[0] + c * dd[1])
                out[k] = out[k].tolist()
        return out


class CameraBall:
    """Ball source for Mission: estimated pose ⊕ camera-relative ball, re-based at each look.

    get(t) -> (ball_xy in the estimated world frame, ball velocity estimate). `found` is False until
    a look succeeds; while not found the mission must hold and re-scan (never kick blind).
    """

    name = "camera"

    def __init__(self, sim, pose_src, method: str = "ray", profile: str = board_profiles.PRIMARY):
        self.sim, self.pose_src, self.method = sim, pose_src, method
        self.cam = BallCamera(sim.model, profile)
        self.reset()

    def reset(self, hide_s: float = 0.0):
        self.hide_s, self.hide_until = hide_s, None
        self.ball = None            # estimated world frame
        self.v = np.zeros(2)
        self.t_look = -math.inf
        self.t_seen = None
        self.last_meas = None
        self.stats = {"looks": 0, "found": 0, "edge": 0}

    @property
    def found(self) -> bool:
        return self.ball is not None

    def look(self, t: float, force: bool = False) -> bool:
        """Take a look if one is due (or forced). Returns True if the ball was seen this look."""
        if not force and t - self.t_look < LOOK_PERIOD_S - 1e-9:
            return False
        self.t_look = t
        self._hide_tick(t, force)
        est = self.pose_src.get(t)
        if est.pose is None:
            return False
        self.stats["looks"] += 1
        m = self.cam.measure(self.sim.data)
        if m is None or self.method not in m:
            return False
        if m["edge"]:
            self.stats["edge"] += 1
            return False
        self.stats["found"] += 1
        lx, ly = m[self.method]
        x, y, yaw = est.pose
        c, s = math.cos(yaw), math.sin(yaw)
        new = np.array([x + c * lx - s * ly, y + s * lx + c * ly])
        if self.ball is not None and self.t_seen is not None and t > self.t_seen:
            self.v = (new - self.ball) / (t - self.t_seen)
        self.ball, self.t_seen, self.last_meas = new, t, m
        return True

    # behaviour test: grey the ball out in the SCENE (the detector genuinely fails), starting at the
    # first forced look after the scan (the re-observe), for hide_s seconds
    def _hide_tick(self, t, force):
        g = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        if self.hide_s > 0 and self.hide_until is None and force and self.ball is not None:
            self._rgba = self.sim.model.geom_rgba[g].copy()
            self.sim.model.geom_rgba[g] = (0.5, 0.5, 0.5, 1.0)
            self.hide_until = t + self.hide_s
        elif self.hide_until is not None and t >= self.hide_until:
            self.unhide()

    def unhide(self):
        if getattr(self, "_rgba", None) is not None:
            g = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
            self.sim.model.geom_rgba[g] = self._rgba
            self._rgba = None
        self.hide_until = math.inf if self.hide_until is not None else None

    def get(self, t: float = 0.0):
        if self.ball is None:
            return None, np.zeros(2)
        v = self.v if self.t_seen is not None and t - self.t_seen < 2 * LOOK_PERIOD_S + 1e-9 else np.zeros(2)
        return self.ball.copy(), v.copy()
