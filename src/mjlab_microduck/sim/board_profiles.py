"""Which sensors a duck has, by electronics board: the one place the sim's sensor numbers live.

Rung 6 Stage 2 (Dillon, 2026-11-14): sensor models are parameterized by board profile.
- **Primary:** face-board-v1 (GC2093 camera, VL53L8CX ToF).
- **Alternate:** zero3 (IMX219 camera, VL53L5CX ToF).
- Which board d10de ships with is still an open VERIFY. Answering it later is a profile flip, not a redesign.

Sources (upstream `pollen-robotics/microduck` at 094d4bb4):
- Board -> camera: `robotd-params/src/board.rs:89-100`. Zero3 -> IMX219, mounted a quarter turn
  off (`rotate` 90); Beta / FaceBoardV1 -> GC2093, upright (`rotate` 0).
- Camera family solves at 1280x720: `robotd-params/src/lib.rs:856-912`.
  - IMX219 (alpha): fx 1061.8, fy 1062.2 -> 62.2 x 37.4 deg.
  - GC2093 (beta): fx 695.6, fy 696.0 -> 85.2 x 54.7 deg.
  - Distortion is recorded here, not rendered: a rendered pinhole has none, so a Stage 3
    detector reads intrinsics rather than assuming them.
- Board -> ToF: `tof/src/link.rs:4-5`, `tof/src/main.rs:113`. Zero3 = VL53L5CX (or L8CX) on
  I2C; beta = VL53L8CX on SPI.
- ToF geometry is the same for both chips: 8x8 zones over a 45 deg square, 15 Hz
  (`kinematics/src/tof.rs:27-34`; `tof/src/main.rs:118-121`).
- The ToF flaw values were measured on ONE alpha (graphite-bang, a VL53L5CX), and reproduced on
  Pollen's recordings by `scripts/tof_dataset_check.py`. **No measurement of a VL53L8CX exists**,
  so the L8CX profile carries the L5CX values as an explicit assumption, not a measured
  difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraProfile:
    sensor: str
    width: int                 # delivered frame, unrotated (mediad's 1280x720 default)
    height: int
    fx: float                  # family solve, pixels at width x height
    fy: float
    cx: float
    cy: float
    rotate_deg: int            # clockwise mount angle from upright (mediad `rotate`)
    distortion: tuple[float, ...] = ()

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / 2 / self.fx))

    @property
    def vfov_deg(self) -> float:
        """Vertical field of the UNROTATED frame: what MuJoCo's `fovy` means for the camera."""
        return math.degrees(2 * math.atan(self.height / 2 / self.fy))


@dataclass(frozen=True)
class TofProfile:
    chip: str
    bus: str
    rows: int = 8
    cols: int = 8
    fov_deg: float = 45.0
    hz: float = 15.0
    flaws_measured_on: str = "VL53L5CX (graphite-bang, alpha), 2026-10-14"


@dataclass(frozen=True)
class BoardProfile:
    name: str
    camera: CameraProfile
    tof: TofProfile
    note: str = ""


GC2093 = CameraProfile("GC2093", 1280, 720, 695.6216231288809, 696.0209167299262, 627.4740538073111,
                       363.60566030048255, rotate_deg=0,
                       distortion=(-0.368607360841971, 0.13550162561831905, 0.00012682873881955374,
                                   0.0008719998338708391, -0.022368269599378002))
IMX219 = CameraProfile("IMX219", 1280, 720, 1061.8060025020175, 1062.193713957031, 596.7776848217006,
                       474.52348043048636, rotate_deg=90,
                       distortion=(-0.34695388691888, 0.14615866153945595, -0.0007449157712424215,
                                   -0.0022167170528743837, -0.016437266141048814))

PROFILES = {
    "face-board-v1": BoardProfile(
        "face-board-v1", GC2093, TofProfile("VL53L8CX", "SPI3"),
        note="primary (Dillon 2026-11-14); upstream calls its image 'production'. ToF flaw values are the "
             "L5CX's: no L8CX measurement exists."),
    "zero3": BoardProfile(
        "zero3", IMX219, TofProfile("VL53L5CX", "I2C"),
        note="alternate: the alpha prototype (Radxa Zero 3W), the board Pollen's recordings come from."),
}
PROFILES["beta"] = BoardProfile("beta", GC2093, TofProfile("VL53L8CX", "SPI3"),
                                note="same sensors as face-board-v1 (board.rs:91)")
PRIMARY = "face-board-v1"


def get(name: str = PRIMARY) -> BoardProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise SystemExit(f"unknown board profile {name!r}; have {sorted(PROFILES)}") from None
