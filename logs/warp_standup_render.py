"""Render a StandUp checkpoint from a forced spawn type in the training sim; saves PNG frames.

Run from repo root: python logs/warp_standup_render.py <model_N.pt> <face_up|face_down|sitting> <out_dir>
"""
import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-StandUp-Flat-MicroDuck"
ckpt, mode, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

env_cfg = load_env_cfg(TASK, play=False)
env_cfg.scene.num_envs = 4
env_cfg.viewer.distance, env_cfg.viewer.elevation, env_cfg.viewer.azimuth = 0.45, -15.0, 135.0
env_cfg.viewer.width, env_cfg.viewer.height, env_cfg.viewer.max_extra_envs = 640, 480, 0
probs = {k: 0.0 for k in ("standing_prob", "sitting_prob", "face_down_prob", "face_up_prob")}
probs[f"{mode}_prob"] = 1.0
env_cfg.events["set_ground_state"].params.update(probs)
env_cfg.curriculum.pop("ground_state_mix", None)  # keep the forced spawn mix
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env.common_step_counter = 4000 * 24
vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or OnPolicyRunner)(vec, asdict(agent_cfg), device="cuda:0")
runner.load(ckpt, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")

obs, _ = vec.reset()
for k in range(int(env_cfg.episode_length_s * 50) - 5):
    with torch.no_grad():
        obs, _, _, _ = vec.step(policy(obs))
    if k in (0, 25, 50, 100, 200, 290):
        imageio.imwrite(out / f"{mode}_t{k/50:.1f}s.png", env.render())
print("frames written to", out)
env.close()
