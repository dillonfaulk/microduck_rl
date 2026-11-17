"""Goal perception (rung 6 Stage 4, Workstream A): the goal measured in the duck's own frame.

The goal is two white posts (goal.xml: 20 mm diameter, 120 mm tall, centres 230 mm apart = the
210 mm opening + one post diameter). The head camera (board profile, face-board-v1 default, the same
camera object the ball channel and the duck-cam use) is rendered and the goal fitted to it:

1. **Detect** white, unsaturated blobs (the posts; the floor's checker edges are a duller grey and
   the boards are tan) and keep only blobs **the size a post would be where they stand**: the blob's
   base row is ranged onto the floor (ray -> plane z = 0), and its pixel height and width must match
   a 120 x 20 mm post at that range. A pair whose bases are about a goal-width apart is the goal.
2. **Fit** the goal pose (centre x, y and facing) to both posts by least squares: each post's
   column (the bearing, precise) and its base and top rows (range, weaker), projected through the
   rendered camera's own intrinsics. The known 230 mm post spacing is the landmark: at range it,
   not the base row, is what pins distance.

Output in the duck's yaw frame from the trunk (as camera_ball.py), so world heading error cancels.
What the robot knows that this uses: camera pose from head FK + IMU roll/pitch (read from the sim's
camera frame, as the ball channel does), the flat floor at z = 0, and the goal's dimensions.
Ground truth (the goal's placed pose) is used by camera_goal_check.py to score ranging, never here.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy import ndimage

from camera_ball import BallCamera

POST_HEIGHT = 0.120
POST_DIAMETER = 0.020
POST_SPACING = 0.230            # centre to centre: 210 mm opening + one post diameter (goal.xml)
SPACING_TOL = 0.08              # a candidate pair's base-ranged spacing must be within this of 0.230
MIN_POST_PX = 6
EDGE_MARGIN_PX = 2
TARGET_GRID_M = 0.25            # the target-goal assignment's precision (an instruction, not a measurement)
TARGET_GATE_M = 0.6             # a seen goal must lie within this of the assignment to be the target
GOAL_LOOK_N = 3                 # looks averaged at the re-observe
SIGMA_U, SIGMA_V = 0.5, 1.5     # px: a post's column is sharp; its base/top rows are not


def white_mask(rgb: np.ndarray) -> np.ndarray:
    """Bright and unsaturated: the posts (rgba 0.95 grey-white, lit 115-215 on the min channel), not
    the floor's 0.8 checker edges at that brightness, not the tan boards (saturated), not the sky."""
    c = rgb.astype(np.int16)
    mn, mx = c.min(axis=2), c.max(axis=2)
    return (mn > 135) & (mx - mn < 45)


class GoalCamera(BallCamera):
    """The ball channel's camera (same profile, renderer, intrinsics, pixel rays), measuring the goal."""

    def project(self, data, p_world) -> tuple[float, float, float]:
        """World point -> (u, v, depth along the optical axis)."""
        R = data.cam_xmat[self.cam].reshape(3, 3)
        pc = R.T @ (np.asarray(p_world, float) - data.cam_xpos[self.cam])
        z = -pc[2]
        return (self.K["cx"] + self.K["fx"] * pc[0] / z, self.K["cy"] - self.K["fy"] * pc[1] / z, z)

    def _floor(self, data, u, v):
        o, d = self.pixel_ray_world(data, u, v)
        if d[2] > -1e-3:
            return None
        return o + (-o[2] / d[2]) * d

    def posts(self, data, rgb) -> list[dict]:
        """Post-shaped blobs: column centroid, base and top rows, base point on the floor."""
        lab, n = ndimage.label(white_mask(rgb), structure=np.ones((3, 3), int))
        out = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            blob = lab[sl] == i
            npx = int(blob.sum())
            if npx < MIN_POST_PX:
                continue
            v0, v1, u0, u1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            if u0 < EDGE_MARGIN_PX or v0 < EDGE_MARGIN_PX or u1 > self.w - EDGE_MARGIN_PX or v1 > self.h - EDGE_MARGIN_PX:
                continue
            vs, us = np.nonzero(blob)
            us, vs = us + u0 + 0.5, vs + v0 + 0.5
            # bearing from the upper 80 % of the blob: the base blurs into the floor reflection
            keep = vs <= v0 + 0.8 * (v1 - v0) + 0.5
            u = float(us[keep].mean())
            v_base, v_top = float(v1), float(v0)
            base = self._floor(data, u, v_base)
            if base is None:
                continue
            ub, vb, depth = self.project(data, base)
            ut, vt, _ = self.project(data, base + np.array([0, 0, POST_HEIGHT]))
            h_exp, w_exp = vb - vt, self.K["fx"] * POST_DIAMETER / depth
            h_obs, w_obs = v1 - v0, npx / max(v1 - v0, 1)
            if not (0.55 < h_obs / max(h_exp, 1e-6) < 1.5 and w_obs < 2.5 * w_exp + 2.0):
                continue
            out.append({"u": u, "v_base": v_base, "v_top": v_top, "base": base, "px": npx,
                        "h_ratio": h_obs / h_exp})
        return out

    def _residuals(self, data, params, pl, pr):
        gx, gy, th = params
        side = np.array([-math.sin(th), math.cos(th)]) * POST_SPACING / 2   # +side = the goal's left
        res = []
        for post, sgn in ((pl, 1.0), (pr, -1.0)):
            c = np.array([gx, gy]) + sgn * side
            u_mid, _, _ = self.project(data, [c[0], c[1], POST_HEIGHT / 2])
            _, v_b, _ = self.project(data, [c[0], c[1], 0.0])
            _, v_t, _ = self.project(data, [c[0], c[1], POST_HEIGHT])
            res += [(u_mid - post["u"]) / SIGMA_U, (v_b - post["v_base"]) / SIGMA_V, (v_t - post["v_top"]) / SIGMA_V]
        return np.array(res)

    def fit(self, data, pl, pr):
        """Gauss-Newton on (centre x, y, facing) from the two posts' bearings and rows."""
        c0 = (pl["base"][:2] + pr["base"][:2]) / 2
        d = pl["base"][:2] - pr["base"][:2]                 # right -> left post
        th0 = math.atan2(d[1], d[0]) - math.pi / 2          # the goal's +x (into the goal) from its +y (left)
        p = np.array([c0[0], c0[1], th0])
        for _ in range(12):
            r = self._residuals(data, p, pl, pr)
            J = np.zeros((len(r), 3))
            for k, eps in enumerate((1e-4, 1e-4, 1e-4)):
                dp = np.zeros(3)
                dp[k] = eps
                J[:, k] = (self._residuals(data, p + dp, pl, pr) - r) / eps
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            p = p + step
            if np.abs(step).max() < 1e-6:
                break
        return p, float(np.sqrt(np.mean(self._residuals(data, p, pl, pr) ** 2)))

    def measure_all(self, data, rgb: np.ndarray | None = None) -> list[dict]:
        """Every goal in view (the field may hold more than one: the field's own end goals and a
        placed goal), each in the duck's yaw frame: {"centre": (x, y), "into": yaw of the goal's +x
        (into the goal) relative to the duck, "base_centre": ray->plane-only centre (the cross-check),
        "spacing_m": base-ranged post spacing, "rms": fit residual (sigma units)}. Posts are paired
        greedily by how close their base-ranged spacing is to 230 mm; each post is used once."""
        self.looks += 1
        rgb = self.render(data) if rgb is None else rgb
        cands = self.posts(data, rgb)
        pairs = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                sep = float(np.linalg.norm(cands[i]["base"][:2] - cands[j]["base"][:2]))
                if abs(sep - POST_SPACING) < SPACING_TOL:
                    pairs.append((abs(sep - POST_SPACING), i, j, sep))
        used, goals = set(), []
        R_t = data.xmat[self.trunk].reshape(3, 3)
        p_t = data.xpos[self.trunk]
        yaw = math.atan2(R_t[1, 0], R_t[0, 0])   # only to express the result in the duck's yaw frame
        c, s = math.cos(yaw), math.sin(yaw)

        def rel(w):
            dd = np.asarray(w[:2]) - p_t[:2]
            return (float(c * dd[0] + s * dd[1]), float(-s * dd[0] + c * dd[1]))

        cam = data.cam_xpos[self.cam][:2]
        for _, i, j, sep in sorted(pairs):
            if i in used or j in used:
                continue
            a, b = cands[i], cands[j]
            # which post is the goal's LEFT depends on its facing: take the fit facing AWAY from the
            # camera (the duck kicks into the goal from the field side)
            p, rms = None, None
            for pl, pr in ((a, b), (b, a)):
                q, e = self.fit(data, pl, pr)
                into = np.array([math.cos(q[2]), math.sin(q[2])])
                if (q[:2] - cam) @ into > 0 and (rms is None or e < rms):
                    p, rms = q, e
            if p is None:
                continue
            used |= {i, j}
            base_c = (a["base"] + b["base"]) / 2
            goals.append({"centre": rel(p[:2]), "into": float((p[2] - yaw + math.pi) % (2 * math.pi) - math.pi),
                          "base_centre": rel(base_c[:2]), "spacing_m": round(sep, 4), "rms": round(rms, 3),
                          "posts_u": sorted([round(a["u"], 1), round(b["u"], 1)])})
        return goals

    def measure(self, data, rgb: np.ndarray | None = None, near_rel=None, gate: float = math.inf) -> dict | None:
        """The goal nearest `near_rel` (duck yaw frame; None = nearest the duck) within `gate` m."""
        goals = self.measure_all(data, rgb)
        ref = np.zeros(2) if near_rel is None else np.asarray(near_rel, float)
        goals = [g for g in goals if np.linalg.norm(np.asarray(g["centre"]) - ref) <= gate]
        return min(goals, key=lambda g: np.linalg.norm(np.asarray(g["centre"]) - ref)) if goals else None


class CameraGoal:
    """Goal source for Mission: estimated pose ⊕ camera-relative goal, re-based at each look.

    get() -> (centre_xy, into_yaw, t_seen) in the estimated world frame, or (None, None, None)."""

    name = "camera"

    def __init__(self, sim, pose_src, profile: str | None = None):
        from mjlab_microduck.sim import board_profiles
        self.sim, self.pose_src = sim, pose_src
        self.cam = GoalCamera(sim.model, profile or board_profiles.PRIMARY)
        self.reset()

    def reset(self, target=None, hide: str | None = None):
        """target: WHICH goal to attack, as a coarse field position (an instruction, like a waypoint:
        `assign_target` rounds it to a 0.25 m grid); the goal's pose itself comes only from the camera.
        hide (behaviour test): "all" = the posts are dark from the start; "look" = dark from the first
        look after the goal was found (i.e. the pre-stance re-observe). The detector genuinely fails."""
        self.unhide()
        self.hide = hide
        self.centre, self.into, self.t_seen, self.last_meas = None, None, None, None
        self.target = None if target is None else np.asarray(target, float)
        self.stats = {"looks": 0, "found": 0}

    @property
    def found(self) -> bool:
        return self.centre is not None

    def _posts(self):
        m = self.sim.model
        return [i for i in range(m.ngeom) if (m.geom(i).name or "").endswith(("_post_left", "_post_right"))]

    def _set_hidden(self, on: bool):
        m = self.sim.model
        if on and getattr(self, "_rgba", None) is None:
            self._rgba = {g: m.geom_rgba[g].copy() for g in self._posts()}
            for g in self._rgba:
                m.geom_rgba[g] = (0.15, 0.25, 0.15, 1.0)
        elif not on and getattr(self, "_rgba", None) is not None:
            for g, c in self._rgba.items():
                m.geom_rgba[g] = c
            self._rgba = None

    def unhide(self):
        self._set_hidden(False)

    def look(self, t: float) -> dict | None:
        """One look; returns the measurement in the ESTIMATED world frame (and keeps it) or None."""
        if self.hide == "all" or (self.hide == "look" and self.found):
            self._set_hidden(True)
        est = self.pose_src.get(t)
        if est.pose is None:
            return None
        self.stats["looks"] += 1
        x, y, yaw = est.pose
        c, s = math.cos(yaw), math.sin(yaw)
        near = None
        if self.target is not None:      # the assignment, into the duck's frame (estimated pose)
            d = self.target - np.array([x, y])
            near = (c * d[0] + s * d[1], -s * d[0] + c * d[1])
        m = self.cam.measure(self.sim.data, near_rel=near, gate=TARGET_GATE_M if near is not None else math.inf)
        if m is None:
            return None
        self.stats["found"] += 1
        lx, ly = m["centre"]
        centre = np.array([x + c * lx - s * ly, y + s * lx + c * ly])
        into = (yaw + m["into"] + math.pi) % (2 * math.pi) - math.pi
        self.centre, self.into, self.t_seen, self.last_meas = centre, into, t, m
        return {"centre": centre, "into": into, "t": t, "meas": m}

    def landmark_sightings(self) -> list[dict]:
        """Every goal in view, in the duck's yaw frame (continuous play, fall re-localization: the
        field's FIXED goals are landmarks; which one is which is the caller's business)."""
        if self.hide == "all" or (self.hide == "look" and self.found):
            self._set_hidden(True)
        self.stats["landmark_looks"] = self.stats.get("landmark_looks", 0) + 1
        return self.cam.measure_all(self.sim.data)

    def get(self, t: float = 0.0):
        if self.centre is None:
            return None, None, None
        return self.centre.copy(), self.into, self.t_seen


def assign_target(xy):
    """The coarse target assignment handed to the mission: the goal's nominal spot on a 0.25 m grid."""
    return tuple(round(float(v) / TARGET_GRID_M) * TARGET_GRID_M for v in xy)
