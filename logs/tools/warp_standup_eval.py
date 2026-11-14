"""Roll a StandUp checkpoint for one episode in the TRAINING simulator (MuJoCo Warp) and score get-ups.

Run from repo root: python logs/tools/warp_standup_eval.py <model_N.pt> [num_envs]
"""
import sys
from dataclasses import asdict

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-StandUp-Flat-MicroDuck"
ckpt, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024
# --mask: emulate distill.py's joint-masked teacher (velstand TEACHER_MASK_COLS / _TILT_DEG)
MASK = "--mask" in sys.argv
MASK_COLS, MASK_TILT_DEG = slice(6, 48), 75.0
UP_TILT_DEG, UP_Z = 25.0, 0.09

env_cfg = load_env_cfg(TASK, play=False)
env_cfg.scene.num_envs = n
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env.common_step_counter = 4000 * 24  # curricula at their final stage, as at the end of training
vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or OnPolicyRunner)(vec, asdict(agent_cfg), device="cuda:0")
try:
    runner.load(ckpt, map_location="cuda:0")
except RuntimeError:
    # VelStand checkpoint: same 61-D actor, wider critic -> rise-only test of the actor
    runner.alg.actor.load_state_dict(torch.load(ckpt, map_location="cuda:0")["actor_state_dict"])
    print("loaded actor only (critic shape differs)")
policy = runner.get_inference_policy(device="cuda:0")

robot = env.scene["robot"]


def state():
    g = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.arccos((-g[:, 2]).clamp(-1, 1)))
    return tilt, robot.data.root_link_pos_w[:, 2].clone()


obs, _ = vec.reset()
env.common_step_counter = 4000 * 24
obs, _ = vec.reset()
tilt0, z0 = state()
kind = torch.zeros(n, dtype=torch.long)  # 0 standing, 1 sitting, 2/3 prone split by gravity-x sign
kind[(z0 < 0.105).cpu() & (tilt0 < 30).cpu()] = 1
g0 = robot.data.projected_gravity_b.clone()
prone = (tilt0 > 60).cpu()
kind[prone & (g0[:, 0] < 0).cpu()] = 2
kind[prone & (g0[:, 0] >= 0).cpu()] = 3

ep_len = env.episode_length_buf.clone()
up_steps = torch.zeros(n, device="cuda:0")
first_up = torch.full((n,), -1.0, device="cuda:0")
peak_w = torch.zeros(n, device="cuda:0")
for k in range(int(env_cfg.episode_length_s * 50) - 5):
    with torch.no_grad():
        if MASK:
            masked = state()[0] > MASK_TILT_DEG
            obs["actor"][masked, MASK_COLS] = 0.0
        obs, _, _, _ = vec.step(policy(obs))
    tilt, z = state()
    peak_w = torch.maximum(peak_w, robot.data.root_link_ang_vel_b.norm(dim=-1))
    up = (tilt < UP_TILT_DEG) & (z > UP_Z)
    up_steps = torch.where(up, up_steps + 1, torch.zeros_like(up_steps))
    first_up = torch.where((up_steps >= 50) & (first_up < 0), torch.full_like(first_up, k / 50 - 1.0), first_up)
reset_mid = (env.episode_length_buf < ep_len).cpu()
tilt, z = state()
nonfoot = (env.scene["nonfoot_ground_contact"].data.found.sum(-1) > 0).cpu() \
    if "nonfoot_ground_contact" in env.scene.sensors else None
names = ["standing", "sitting", "prone g_x<0", "prone g_x>=0"]
print(f"checkpoint {ckpt}  envs={n}  (mid-episode resets: {int(reset_mid.sum())}, excluded)"
      + ("  [JOINT-MASKED >75deg]" if MASK else ""))
for i, name in enumerate(names):
    m = (kind == i) & ~reset_mid
    if m.sum() == 0:
        continue
    rec = (first_up.cpu()[m] >= 0)
    t = first_up.cpu()[m][rec]
    print(f"{name:15s} n={int(m.sum()):4d}  recovered(held 1s) {rec.float().mean()*100:5.1f}%  "
          f"t_med={t.median().item() if len(t) else float('nan'):.2f}s  "
          f"final tilt med {tilt.cpu()[m].median():5.1f}deg  final z med {z.cpu()[m].median()*1000:5.1f}mm"
          f"  peak|w| med/p90 {peak_w.cpu()[m].median():4.1f}/{peak_w.cpu()[m].quantile(0.9):4.1f} rad/s"
          + (f"  non-foot floor contact {nonfoot[m].float().mean()*100:5.1f}%" if nonfoot is not None else ""))
env.close()
