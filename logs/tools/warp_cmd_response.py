"""Fed-command -> achieved-velocity response of a checkpoint in the TRAINING sim (MuJoCo Warp).

Every env holds one fixed command (no resampling, no heading control, no pushes, upright
HOME spawn) for 8 s; achieved body-frame (vx, vy, wz) is averaged over 3-8 s on envs that
never tilted past 60 deg. Use it on the walk anchor (model_4750) and on VelStand students.

Run from repo root: python logs/tools/warp_cmd_response.py <model_N.pt> [reps] > table.txt
"""
import sys
from dataclasses import asdict

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-VelStand-Flat-MicroDuck"
REMAP = "--remap" in sys.argv  # cells are TARGETS; the policy is fed LOCAL_ANCHOR_TWIST_REMAP(target)
TARGET_GRID = REMAP or "--target-grid" in sys.argv  # the target grid, fed as-is (for students)
argv = [a for a in sys.argv if a not in ("--remap", "--target-grid")]
ckpt, reps = argv[1], int(argv[2]) if len(argv) > 2 else 20

def sym(mags):
    return [-m for m in reversed(mags)] + list(mags)


X = sym([round(0.05 * i, 2) for i in range(1, 13)])          # to ±0.6 (trained ±0.4)
Y = sym([round(0.05 * i, 2) for i in range(2, 10)])          # to ±0.45 (trained ±0.3)
W = sym([round(0.1 * i, 1) for i in range(1, 17)])           # to ±1.6 (trained ±1.0)
cells = ([(x, 0.0, 0.0) for x in X] + [(0.0, y, 0.0) for y in Y] + [(0.0, 0.0, w) for w in W]
         + [(vx, 0.0, w) for vx in (0.2, 0.3, 0.4) for w in sym((0.3, 0.5, 0.7, 0.9, 1.1, 1.3))]
         + [(0.0, 0.0, 0.0)])
if TARGET_GRID:
    cells = ([(x, 0.0, 0.0) for x in sym((0.05, 0.1, 0.15, 0.2, 0.25))] + [(0.0, y, 0.0) for y in sym((0.05, 0.1))]
             + [(0.0, 0.0, w) for w in sym((0.2, 0.4, 0.6, 0.8, 1.0))]
             + [(vx, 0.0, w) for vx in (0.1, 0.2) for w in sym((0.3, 0.6, 1.0))] + [(0.0, 0.0, 0.0)])
n = len(cells) * reps

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = n
env_cfg.events["random_prone_init"].params.update(prone_prob=0.0, crouch_prob=0.0, sit_prob=0.0)  # upright HOME spawns
for name in [k for k, e in env_cfg.events.items() if e.mode == "interval"]:
    env_cfg.events.pop(name)  # no pushes / interval disturbances
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or OnPolicyRunner)(vec, asdict(agent_cfg), device="cuda:0")
runner.load(ckpt, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]

cmd = torch.tensor(cells, dtype=torch.float32, device="cuda:0").repeat_interleave(reps, dim=0)
if REMAP:
    from mjlab_microduck.tasks.distill import remap_anchor_twist
    from mjlab_microduck.tasks.microduck_velstand_env_cfg import LOCAL_ANCHOR_TWIST_REMAP
    fake = torch.zeros(len(cells), 61)
    fake[:, 48:51] = torch.tensor(cells)
    fed_cells = remap_anchor_twist(fake, (48, 51), LOCAL_ANCHOR_TWIST_REMAP)[:, 48:51]
    cmd = fed_cells.to("cuda:0").repeat_interleave(reps, dim=0)
tw = env.command_manager.get_term("twist")
tw._resample_command = lambda env_ids: None
tw._update_command = lambda: None
for name in env.command_manager.active_terms:  # body/head pose commands -> zero (plain walking)
    if name != "twist":
        t = env.command_manager.get_term(name)
        t._resample_command = lambda env_ids: None
        t._update_command = lambda: None


def hold():
    tw.vel_command_b[:] = cmd
    tw.is_standing_env[:] = False
    tw.is_heading_env[:] = False
    for name in env.command_manager.active_terms:
        if name != "twist":
            env.command_manager.get_term(name).command.zero_()


obs, _ = vec.reset()
hold()
obs = vec.get_observations()
g0 = robot.data.projected_gravity_b
start_up = (-g0[:, 2]) > np.cos(np.radians(25))
print(f"started upright {int(start_up.sum())}/{n}")
ok = start_up.clone()
acc = torch.zeros(n, 3, device="cuda:0")
k_steps = 0
for k in range(8 * 50):
    with torch.no_grad():
        obs, _, dones, _ = vec.step(policy(obs))
    hold()
    g = robot.data.projected_gravity_b
    ok &= ~dones.bool() & ((-g[:, 2]) > np.cos(np.radians(60)))
    if k >= 150:
        v = robot.data.root_link_lin_vel_b
        acc += torch.stack([v[:, 0], v[:, 1], robot.data.root_link_ang_vel_b[:, 2]], dim=1)
        k_steps += 1

ach = (acc / k_steps).view(len(cells), reps, 3)
okc = ok.view(len(cells), reps)
fell = (start_up & ~ok).view(len(cells), reps).sum(1)
print(f"started upright but fell/reset: {int(fell.sum())} total")
print(f"checkpoint {ckpt}  reps={reps}  8 s fixed command, achieved = mean over 3-8 s of upright envs")
print(f"{'target (vx, vy, wz)' if REMAP else 'fed (vx, vy, wz)':>24s}   {'achieved (vx, vy, wz)':>24s}  upright  fell")
for i, c in enumerate(cells):
    m = okc[i]
    a = ach[i][m].mean(0).tolist() if m.any() else [float("nan")] * 3
    print(f"({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})      ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})   {int(m.sum())}/{reps}  {int(fell[i])}")
