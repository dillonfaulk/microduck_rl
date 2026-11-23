"""Build the StandUp env with MICRODUCK_HEADJAM=1, reset 2048 envs, and check the head-under share and pose."""
import os, math, torch
os.environ["MICRODUCK_HEADJAM"] = "1"
import mjlab_microduck.tasks  # noqa: registers
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
cfg = load_env_cfg("Mjlab-StandUp-Flat-MicroDuck"); cfg.scene.num_envs = 2048
env = ManagerBasedRlEnv(cfg, device="cuda:0")
env.reset()
r = env.scene["robot"]
neck = r.data.joint_pos[:, r.find_joints("neck_pitch")[0][0]]
q = env.sim.data.qpos[:, 3:7]; w, x, y, z = q.unbind(1)
pitch = torch.rad2deg(torch.asin((2 * (w * y - z * x)).clamp(-1, 1)))
hu = neck < -1.09
print("head-under share", hu.float().mean().item(), "count", env.__dict__.get("_head_under_count"))
print("pitch of head-under: min %.1f max %.1f" % (pitch[hu].min(), pitch[hu].max()))
print("trunk z of head-under: %.3f..%.3f" % (env.sim.data.qpos[hu, 2].min(), env.sim.data.qpos[hu, 2].max()))
for _ in range(50):
    env.step(torch.zeros(2048, env.action_manager.total_action_dim, device="cuda:0"))
print("after 1 s zero action: alive", (~env.termination_manager.terminated).float().mean().item())
print("action term clip:", env.action_manager.get_term("joint_pos")._clip[0].tolist() if hasattr(env.action_manager.get_term("joint_pos"), "_clip") else None)
