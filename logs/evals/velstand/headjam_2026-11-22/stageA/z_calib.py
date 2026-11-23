"""Measure the floor clearance of head-under spawns vs trunk pitch (CPU MuJoCo, field scene) to set the spawn z."""
import math, sys
import numpy as np, mujoco
sys.path.insert(0, "scripts")
from kick_battery import SCENES, KickSim
S = np.load("logs/evals/velstand/headjam_2026-11-22/stall_state_6m_look_run5_t60.npz")
sim = KickSim("logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx", SCENES["field"])
m, d, p = sim.model, sim.data, sim.policy
qi = list(p.joint_qpos_indices)
fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[1]]
def quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2), math.sin(pitch/2), math.cos(yaw/2), math.sin(yaw/2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
def min_clear(z, roll, pitch, joints):
    d.qpos[:] = S["qpos"]; d.qpos[2] = z; d.qpos[3:7] = quat(roll, pitch, 0.0)
    for k, q in enumerate(qi): d.qpos[q] = joints[k]
    mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
    best = 1.0
    for g in robot_geoms:
        best = min(best, mujoco.mj_geomDistance(m, d, g, fl, 0.5, None))
    return best
rng = np.random.default_rng(0)
base = np.array([S["qpos"][q] for q in qi])
for pd in (25, 35, 42.7, 50, 60):
    need = []
    for _ in range(40):
        j = base + rng.uniform(-0.15, 0.15, 14)
        j[5] = rng.uniform(-1.57, -1.1); j[7] = rng.uniform(-1.1, 1.1)
        roll = math.radians(rng.uniform(-20, 20))
        c = min_clear(0.30, roll, math.radians(pd), j)
        need.append(0.30 - c)       # trunk z at which the lowest point touches the floor
    need = np.array(need)
    print(f"pitch {pd:5.1f}: touch-z min {need.min():.3f} med {np.median(need):.3f} max {need.max():.3f}")
print("stall z", S["qpos"][2])
