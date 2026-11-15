"""A simulated VL53L5CX / VL53L8CX, so the duck can see a wall.

8x8 zones over a 45-degree square field of view, out to 4 m, at 15 Hz. That is the real sensor's
shape: `tofd` publishes frames of exactly that, and `maploc` reprojects them assuming it.

Modelled on `~/MISC/microduck_maploc`'s `sim/tof_sensor.py`, which had these numbers from the
datasheet and from the sensor on a desk: noise that grows with distance (millimetres up close,
centimetres out near the limit), and a status per zone rather than a distance alone.

**The status byte matters as much as the distance.** A zone that returned nothing is no
information, and `maploc` believes only statuses 5 and 9. A simulator reporting only distances
would let a bug through that hardware finds.

**The real sensor's flaws, behind switches, default ON** (rung 6 Stage 2, 2026-11-14). Values
come from upstream `maploc` (`project.rs:57-75`, `sim.rs:378-401`). They were re-measured on
Pollen's recordings by `scripts/tof_dataset_check.py`
(`logs/evals/vision/tof_dataset_check_2026-11-14/`):

- `axial_range`: the sensor reports distance along its optical axis, not along each zone's ray.
  A corner zone reads up to ~11 % short. Measured: with slant range the corner zones read
  27-38 mm shorter than the centre; with axial range, within +-10 mm.
- `latency_s` 0.060: a frame's light was collected ~60 ms before its stamp. Only `frame_at`
  applies it, because a delay needs a clock.
- `pitch_offset_deg` 3.0: the sensor sits ~3 deg further down than the head's kinematics say.
  Positive tips the beams down, as `ProjectConfig::pitch_offset_rad` does. Measured on one
  alpha, VL53L5CX. No VL53L8CX measurement exists (see `board_profiles`).
- `dropout`: the share of zones that come back valid (status 5/9) falls with range. That's the
  "reliable to ~1 m" of the spec sheet, measured per range bin on s2 + s3 (pooled, 0.25 m bins,
  in `DROPOUT_VALID`). A dropped zone reports status 0 with its distance kept, as the real wire
  does. With dropout ON, a zone with nothing in range also reports status 0: the real
  recordings contain no 255 at all. OFF restores the legacy 255 "no target".

The noise model is unchanged. The recordings can't separate sensor noise from pose and map
error: the inlier spread of 23-69 mm is an upper bound. `TofFlaws.none()` is the old ideal
sensor, bit for bit.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

# The sensor, as `tof/src/lib.rs` publishes it.
ROWS = 8
COLS = 8
ZONES = ROWS * COLS
STATUS_VALID = 5
STATUS_INVALID = 0
STATUS_NO_TARGET = 255

# VL53L5CX: 45 degrees per axis (63 on the diagonal), 4 m of range.
FOV_DEG = 45.0
MAX_RANGE = 4.0

# P(status 5/9) by reported range, s2 + s3 pooled (tof_dataset_check, 2026-11-14): upper bin
# edges in metres and the valid share in each bin. Past 3 m the scan is incomplete (glass,
# doorway, above the scan), so the last bins are the least trustworthy numbers here.
DROPOUT_EDGES = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00)
DROPOUT_VALID = (0.955, 0.940, 0.680, 0.400, 0.355, 0.240, 0.225, 0.345)


@dataclass(frozen=True)
class TofFlaws:
    axial_range: bool = True
    latency_s: float = 0.060
    pitch_offset_deg: float = 3.0
    dropout: bool = True

    @classmethod
    def none(cls) -> "TofFlaws":
        """The ideal sensor this module modelled before Stage 2."""
        return cls(axial_range=False, latency_s=0.0, pitch_offset_deg=0.0, dropout=False)


def valid_probability(reported_m: float) -> float:
    i = bisect.bisect_left(DROPOUT_EDGES, reported_m)
    return DROPOUT_VALID[min(i, len(DROPOUT_VALID) - 1)]


class Tof:
    """The 8x8 sensor on one duck's `tof` site.

    Rays are cast in the site's frame (+x forward, +y left, +z up), so a head that turns takes the
    sensor with it. That is the whole point of `robot.look` scanning a room.
    """

    def __init__(self, model: mujoco.MjModel, site: int, seed: int = 0, flaws: TofFlaws | None = None):
        self.model = model
        self.site = site
        self.random = np.random.default_rng(seed)
        self.flaws = TofFlaws() if flaws is None else flaws
        self._queue: deque = deque()

        # Zone centres, once. Azimuth is about +z (positive toward +y, the duck's left), elevation
        # is up, and both span the field of view. Row 0 is the top of the frame and column 0 the
        # sensor's LEFT, as `kinematics::tof` reads the real sensor's buffer. With column 0 on the
        # right, every frame reached the mapper mirrored: oblique walls were inked at their mirror
        # image across the head's axis, and no loop ever closed on the twin.
        half = np.radians(FOV_DEG) / 2.0
        edges = np.linspace(-half, half, COLS + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        self.directions = np.zeros((ZONES, 3))
        for row in range(ROWS):
            elevation = -centres[row]
            for col in range(COLS):
                azimuth = -centres[col]
                self.directions[row * COLS + col] = [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]
        p = np.radians(self.flaws.pitch_offset_deg)
        tip = np.array([[np.cos(p), 0.0, np.sin(p)], [0.0, 1.0, 0.0], [-np.sin(p), 0.0, np.cos(p)]])
        self._beams = self.directions @ tip.T          # the beams as the mounted sensor really casts them
        self._geomid = np.zeros(ZONES, dtype=np.int32)
        self._dist = np.zeros(ZONES)

    def frame(self, data: mujoco.MjData) -> tuple[list[int], list[int]]:
        """One capture at this instant: distances in millimetres and a status per zone.

        Self-hits are reported, not filtered. A real sensor sees the duck's own beak when the beak
        is in front of it, and a simulator that quietly skipped its own geometry would hide exactly
        the kind of mounting problem this is here to catch.
        """
        no_target = STATUS_INVALID if self.flaws.dropout else STATUS_NO_TARGET
        distance_mm = [0] * ZONES
        status = [no_target] * ZONES

        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        world = rotation @ self._beams.T  # (3, ZONES)

        # A zero-length direction makes MuJoCo's ray cast abort the whole process ("vector length
        # is too small"), so it is checked rather than trusted. It happens for a real reason:
        # before the first forward pass every site orientation is zero, and a client can connect
        # before then.
        if not np.isfinite(world).all() or np.linalg.norm(world) < 1e-9:
            return distance_mm, status

        # One mj_multiRay call, with the same per-zone semantics as a mj_ray loop: every geom
        # group, static geoms included, nothing excluded. A miss is -1.
        mujoco.mj_multiRay(self.model, data, origin, np.ascontiguousarray(world.T).reshape(-1), None, 1, -1,
                           self._geomid, self._dist, None, ZONES, MAX_RANGE + 1.0)
        for zone in range(ZONES):
            hit = self._dist[zone]
            if hit < 0 or hit > MAX_RANGE:
                continue
            # Noise that grows with range, as the datasheet has it: a few millimetres up close, a
            # couple of centimetres at the far end. Without it a simulated map is suspiciously
            # crisp, and every filter downstream goes untested.
            sigma = 0.003 + 0.02 * (hit / MAX_RANGE)
            reported = hit * self._beams[zone, 0] if self.flaws.axial_range else hit
            measured = max(0.0, reported + self.random.normal(0.0, sigma))
            distance_mm[zone] = int(measured * 1000.0)
            status[zone] = STATUS_VALID
            if self.flaws.dropout and self.random.random() >= valid_probability(reported):
                status[zone] = STATUS_INVALID
        return distance_mm, status

    def frame_at(self, data: mujoco.MjData, t: float) -> tuple[list[int], list[int], float] | None:
        """Capture now, and return the newest capture at least `latency_s` old, with its capture time.

        That is the real sensor's delay. Called at 15 Hz, the frame handed out is the previous
        one, about 67 ms old. Returns None until a capture is old enough.
        """
        q = self._queue
        q.append((t, self.frame(data)))
        cut = t - self.flaws.latency_s + 1e-9
        while len(q) > 1 and q[1][0] <= cut:   # keep only the newest capture that is old enough
            q.popleft()
        if q[0][0] > cut:
            return None
        tc, (distance_mm, status) = q[0]
        return distance_mm, status, tc
