"""Corner-post perception (4 m field, 2026-11-18): the field's four blue corner posts, measured in the
duck's own frame with the same head camera (board profile) the ball and goal channels use.

A corner post is a 75 mm x 0.40 m matte-blue cylinder (field4.xml). Detection:
  1. saturated-blue blobs (the field's only blue: the boundary wall is visual-only in geom group 5, which
     the camera does not draw; the floor and the sky are a duller, greyer blue);
  2. the blob's base row is ranged onto the floor (ray -> plane z = 0): that is the post's FRONT surface,
     so the axis is one radius further along the ray;
  3. kept only if its pixel width matches a 75 mm post at that range and it is post-tall (its top may be
     cut by the frame at close range; its base must be in frame).
The bearing (the blob's column) is the precise part; the range (the base row) is the weak part.
Output is in the duck's yaw frame from the trunk, like camera_ball / camera_goal. Which corner a sighting
is, is the caller's business (the estimated pose + the field map).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from camera_ball import BallCamera

CORNER_RADIUS = 0.0375
CORNER_HEIGHT = 0.40
MIN_PX = 12
EDGE_MARGIN_PX = 2


def blue_mask(rgb: np.ndarray) -> np.ndarray:
    """The corner posts' saturated blue (rgba 0.10 0.25 0.85, lit), not the grey-blue floor or the sky."""
    c = rgb.astype(np.int16)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return (b > 110) & (b - r > 90) & (b - g > 60)


class CornerCamera(BallCamera):
    """The ball channel's camera (same profile, renderer, intrinsics), measuring the corner posts."""

    def _floor(self, data, u, v):
        o, d = self.pixel_ray_world(data, u, v)
        if d[2] > -1e-3:
            return None, None
        return o + (-o[2] / d[2]) * d, d

    def measure(self, data, rgb: np.ndarray | None = None) -> list[dict]:
        rgb = self.render(data) if rgb is None else rgb
        lab, n = ndimage.label(blue_mask(rgb), structure=np.ones((3, 3), int))
        R_t = data.xmat[self.trunk].reshape(3, 3)
        p_t = data.xpos[self.trunk]
        yaw = math.atan2(R_t[1, 0], R_t[0, 0])            # only to express the result in the yaw frame
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            blob = lab[sl] == i
            npx = int(blob.sum())
            if npx < MIN_PX:
                continue
            v0, v1, u0, u1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            if v1 > self.h - EDGE_MARGIN_PX or u0 < EDGE_MARGIN_PX or u1 > self.w - EDGE_MARGIN_PX:
                continue                                 # base or a side cut by the frame: no range
            vs, us = np.nonzero(blob)
            low = vs >= (v1 - v0) * 0.7                  # the column from the lower 30 % (near the base row)
            u = float(us[low].mean()) + u0 + 0.5 if low.any() else float(us.mean()) + u0 + 0.5
            base, ray = self._floor(data, u, float(v1))
            if base is None:
                continue
            h = np.array([ray[0], ray[1]]) / max(np.linalg.norm(ray[:2]), 1e-9)
            axis = base[:2] + CORNER_RADIUS * h          # the front surface -> the post's axis
            o = data.cam_xpos[self.cam]
            rng_cam = float(np.linalg.norm(axis - o[:2]))
            w_exp = self.K["fx"] * 2 * CORNER_RADIUS / max(rng_cam, 1e-6)
            w_obs = npx / max(v1 - v0, 1)
            # expected pixel height of the post (cut at the top of the frame when close)
            top_ang = math.atan2(CORNER_HEIGHT - o[2], rng_cam)
            base_ang = math.atan2(-o[2], rng_cam)
            h_exp = self.K["fy"] * (math.tan(top_ang) - math.tan(base_ang))
            if not (0.4 < w_obs / max(w_exp, 1e-6) < 2.2) or (v1 - v0) < 0.35 * min(h_exp, v1):
                continue
            d = axis - p_t[:2]
            rel = (float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1]))
            out.append({"rel": rel, "rng": math.hypot(*rel), "u": round(u, 1), "px": npx,
                        "w_ratio": round(w_obs / w_exp, 2)})
        return out


def corner_map(model) -> list[tuple]:
    """The field map's corner posts (x, y), read back from the compiled model (field4.xml)."""
    return [(float(model.geom(g).pos[0]), float(model.geom(g).pos[1])) for g in range(model.ngeom)
            if (model.geom(g).name or "").startswith("field_corner")]
