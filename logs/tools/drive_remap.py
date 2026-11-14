"""infer_policy.py with a runtime velocity-command remap for velstand_model_3000.

The keyboard keeps setting TARGET speeds; only the twist slot the walking policy sees is
replaced by the FED command that produces that speed (inverse of the CPU BAM response
measured by logs/tools/cmd_calibrate.py, 2026-10-30). Fed commands stay inside the trained
ranges (x ±0.4, y ±0.3, wz ±1.0). Use exactly like infer_policy.py:
  python logs/tools/drive_remap.py --new-cmd-obs --scene <scene_allcollisions.xml> --walking <onnx>
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
import infer_policy as ip  # noqa: E402

DEADBAND = 0.02
# (achieved |v|, fed |v|) knots per axis and direction, from cmd_calibrate.py
X_FWD = ((0.0, 0.0), (0.111, 0.20), (0.143, 0.25), (0.179, 0.30), (0.212, 0.35), (0.236, 0.40))
X_BACK = ((0.0, 0.0), (0.110, 0.25), (0.141, 0.30), (0.172, 0.35), (0.210, 0.40))
Y_MAX = 0.3      # sideways barely responds (≤0.08 m/s): any request → trained max
WZ_MIN, WZ_MAX = 0.7, 1.0   # turning is on/off: nothing below 0.7, ~1:1 above


def _axis(t, table):
    a = abs(t)
    if a < DEADBAND:
        return 0.0
    ach, fed = zip(*table)
    f = fed[1] if a <= ach[1] else float(np.interp(a, ach, fed))
    return float(np.sign(t) * f)


def remap(cmd):
    x, y, wz = (float(v) for v in cmd)
    fx = _axis(x, X_FWD if x > 0 else X_BACK)
    fy = 0.0 if abs(y) < DEADBAND else float(np.sign(y) * Y_MAX)
    fw = 0.0 if abs(wz) < DEADBAND else float(np.sign(wz) * np.clip(abs(wz), WZ_MIN, WZ_MAX))
    return np.array([fx, fy, fw], dtype=np.float32)


_orig_update = ip.PolicyInference._update_command


def _update_command(self):
    _orig_update(self)
    if self.new_cmd_obs and self.current_policy == "walking" and self.command.shape[0] == 13:
        fed = remap(self.vel_cmd)
        self.command[0:3] = fed
        if not np.allclose(fed, self.vel_cmd):
            print(f"  remap: target [{self.vel_cmd[0]:+.2f}, {self.vel_cmd[1]:+.2f}, {self.vel_cmd[2]:+.2f}]"
                  f" -> fed [{fed[0]:+.2f}, {fed[1]:+.2f}, {fed[2]:+.2f}]")


ip.PolicyInference._update_command = _update_command

if __name__ == "__main__":
    print("[drive_remap] velocity command remap ON (see logs/tools/drive_remap.py)")
    ip.main()
