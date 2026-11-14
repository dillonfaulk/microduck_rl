"""Score a VelStand checkpoint's get-ups in the TRAINING simulator (MuJoCo Warp).

Spawns every env with StandUp's set_random_ground_state (face-down / face-up with
±90° roll noise / standing control, the stand teacher's spawn ranges), runs one
episode under the training command mix, and reports per start pose:
  recovered = tilt < 25° and z > 90 mm held 1 s, before any reset;
  for the standing control, also how many fell (tilt > 60°) and got back up.

Run from repo root: python logs/tools/warp_velstand_eval.py <model_N.pt> [num_envs]
"""
import math
import sys
from dataclasses import asdict

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks.mdp import set_random_ground_state

TASK = "Mjlab-VelStand-Flat-MicroDuck"
ckpt, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024
UP_TILT_DEG, UP_Z, FALL_TILT_DEG = 25.0, 0.09, 60.0

env_cfg = load_env_cfg(TASK, play=False)
env_cfg.scene.num_envs = n
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env.common_step_counter = 4000 * 24  # curricula at their final stage
vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or OnPolicyRunner)(vec, asdict(agent_cfg), device="cuda:0")
runner.load(ckpt, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]


def state():
    g = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.arccos((-g[:, 2]).clamp(-1, 1)))
    return tilt, robot.data.root_link_pos_w[:, 2].clone()


vec.reset()
env.common_step_counter = 4000 * 24
ids = torch.arange(n, device="cuda:0")
torch.manual_seed(0)
set_random_ground_state(env, ids, face_down_prob=0.4, face_up_prob=0.4, sitting_prob=0.0,
                        standing_prob=0.2, prone_z_min=0.05, prone_z_max=0.09,
                        face_up_roll_max=math.radians(90), sitting_tilt_max=math.radians(10))
env.sim.forward()
obs = vec.get_observations()
tilt0, _ = state()
g0 = robot.data.projected_gravity_b.clone()
kind = torch.zeros(n, dtype=torch.long, device="cuda:0")      # 0 standing control
kind[(tilt0 > 45) & (g0[:, 0] < 0)] = 1                        # 1 prone g_x<0
kind[(tilt0 > 45) & (g0[:, 0] >= 0)] = 2                       # 2 prone g_x>=0

alive = torch.ones(n, dtype=torch.bool, device="cuda:0")      # no reset yet
up_steps = torch.zeros(n, device="cuda:0")
first_up = torch.full((n,), -1.0, device="cuda:0")
fell = torch.zeros(n, dtype=torch.bool, device="cuda:0")
rec_after_fall = torch.zeros(n, dtype=torch.bool, device="cuda:0")
peak_w = torch.zeros(n, device="cuda:0")
steps = int(env_cfg.episode_length_s * 50) - 5
for k in range(steps):
    with torch.no_grad():
        obs, _, dones, _ = vec.step(policy(obs))
    alive &= ~dones.bool()
    tilt, z = state()
    peak_w = torch.where(alive, torch.maximum(peak_w, robot.data.root_link_ang_vel_b.norm(dim=-1)), peak_w)
    up = (tilt < UP_TILT_DEG) & (z > UP_Z) & alive
    up_steps = torch.where(up, up_steps + 1, torch.zeros_like(up_steps))
    held = up_steps >= 50
    first_up = torch.where(held & (first_up < 0), torch.full_like(first_up, k / 50 - 1.0), first_up)
    rec_after_fall |= fell & held
    fell |= (tilt > FALL_TILT_DEG) & alive & (first_up >= 0)

print(f"checkpoint {ckpt}  envs={n}  episode {steps / 50:.0f}s, training command mix")
for i, name in enumerate(["standing (control)", "prone g_x<0", "prone g_x>=0"]):
    m = kind == i
    if m.sum() == 0:
        continue
    rec = first_up[m] >= 0
    t = first_up[m][rec]
    line = (f"{name:19s} n={int(m.sum()):4d}  got up (held 1s) {rec.float().mean() * 100:5.1f}%  "
            f"t_med={t.median().item() if len(t) else float('nan'):.2f}s  "
            f"recycled before up {(~alive[m] & ~rec).float().mean() * 100:5.1f}%  "
            f"peak|w| med {peak_w[m].median():4.1f} rad/s")
    if i == 0:
        f = fell[m]
        line += (f"  | fell later {f.float().mean() * 100:5.1f}%"
                 f", of those back up {rec_after_fall[m][f].float().mean() * 100 if f.any() else float('nan'):5.1f}%")
    print(line)
env.close()
