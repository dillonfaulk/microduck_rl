"""Compare the 61-D actor observation the TRAINING sim (warp) and the CPU viewer
(infer_policy.PolicyInference) produce for the SAME robot state.

Run from repo root: python logs/tools/obs_compare.py
"""
import sys

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

sys.path.insert(0, "scripts")
import infer_policy as ip  # noqa: E402

TASK = "Mjlab-VelStand-Flat-MicroDuck"
SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
s = 2.0 ** -0.5
POSES = {  # name: (z, quat wxyz)
    "standing": (0.118, (1.0, 0.0, 0.0, 0.0)),
    "face-down": (0.07, (s, 0.0, s, 0.0)),
    "face-up": (0.07, (s, 0.0, -s, 0.0)),
    "left side": (0.07, (s, s, 0.0, 0.0)),
}
LABELS = ["ang_vel"] * 3 + ["gravity"] * 3 + ["joint_pos"] * 14 + ["joint_vel"] * 14 + ["last_act"] * 14 + ["cmd"] * 13

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env.reset()
robot = env.scene["robot"]
print("warp joint order:", robot.joint_names)
print("warp default joint pos:", np.round(robot.data.default_joint_pos[0].cpu().numpy(), 4))
print("obs terms:", env.observation_manager.active_terms["actor"])

ip.MICRODUCK_XML = SCENE
bam = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(SCENE, bam, 0.005, 0.1, ip.BAM_VIN_MIN)
pol = ip.PolicyInference(model, data, bam_ctrl=bam_ctrl, new_cmd_obs=True, walking_onnx_path=sys.argv[1])
print("cpu default pose:     ", np.round(pol.default_pose, 4))
fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
qa = model.jnt_qposadr[fj]

for name, (z, quat) in POSES.items():
    q_off = torch.tensor(np.r_[0.0, 0.0, z, quat], dtype=torch.float32, device="cuda:0")
    origin = env.scene.env_origins[0]
    env.sim.data.qpos[0, :7] = q_off + torch.cat([origin, torch.zeros(4, device="cuda:0")])
    env.sim.data.qpos[0, 7:7 + 14] = robot.data.default_joint_pos[0]
    env.sim.data.qvel[0] = 0.0
    env.sim.forward()
    w = env.observation_manager.compute()["actor"][0].cpu().numpy()

    data.qpos[qa:qa + 3] = [0, 0, z]
    data.qpos[qa + 3:qa + 7] = quat
    data.qpos[pol.joint_qpos_indices] = pol.default_pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    c = pol.get_observations()

    d = np.abs(w[:48] - c[:48])
    print(f"\n== {name}: max |warp-cpu| over ang_vel/gravity/joints = {d.max():.4f}")
    print("   warp gravity", np.round(w[3:6], 3), " cpu gravity", np.round(c[3:6], 3))
    print("   warp ang_vel", np.round(w[0:3], 3), " cpu ang_vel", np.round(c[0:3], 3))
    for i in np.where(d > 1e-3)[0]:
        print(f"   [{i:2d}] {LABELS[i]:9s} warp {w[i]:+.4f}  cpu {c[i]:+.4f}")
env.close()
