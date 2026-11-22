#!/bin/bash
# Finalists 1875 / 1999: calibrated missions (stance 0, aim = -uncal drift), batteries as v2 (kick_battery.py,
# 50 each: standing, handoff, push 0.3 m/s at 1.0 / 3.5 s), and the closing-work GoalKick method (goal 0.8 m,
# field scene, 50 runs, gt + full stack) for the lateral-sensitivity record. v2 right 1250 rerun on the GoalKick rows.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
R=logs/rsl_rl/ball_kick_v3_right/2026-11-22_14-22-48_v3_right_ft
D=logs/evals/ballkick/ballkick_v3_eval_2026-11-22
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
KB="env -u PYTHONPATH uv run python scripts/kick_battery.py"
for pair in "1875 -4.3" "1999 -4.9"; do
  set -- $pair; N=$1; A=$2; O=$R/v3_right_ft_model_$N.onnx
  for s in 0 1; do
    KICK_STANCE_SHIFT=0 KICK_AIM_OFFSET_DEG=$A env -u PYTHONPATH uv run python scripts/kick_mission.py $V $O --foot right \
      --precision $P --runs 30 --seed $s > $D/v3R${N}_cal_s$s.txt 2>&1 &
  done
  $KB $O --episodes 50 --target 0.25 --foot right --out $D/v3R${N}_battery.csv > $D/v3R${N}_battery.txt 2>&1 &
  H="--scene allcollisions_ball --prestand $V --prestand-s 1.5 --handoff $V --handoff-at 2.0"
  $KB $O --episodes 50 --target 0.25 --foot right $H --out $D/v3R${N}_handoff.csv > $D/v3R${N}_handoff.txt 2>&1 &
  for pa in 1.0 3.5; do
    $KB $O --episodes 50 --target 0.25 --foot right $H --push-at $pa --push-v 0.3 --out $D/v3R${N}_push$pa.csv > $D/v3R${N}_push$pa.txt 2>&1 &
  done
  G="--foot right --precision $P --goal-dist 0.8 --scene field --runs 50"
  KICK_STANCE_SHIFT=0 KICK_AIM_OFFSET_DEG=$A env -u PYTHONPATH uv run python scripts/kick_mission.py $V $O $G --pose gt > $D/v3R${N}_goal_gt.txt 2>&1 &
  KICK_STANCE_SHIFT=0 KICK_AIM_OFFSET_DEG=$A env -u PYTHONPATH uv run python scripts/kick_mission.py $V $O $G --pose mapd --odom-calib stride_v1 \
    --ball-src camera --goal-src camera --estimated-only > $D/v3R${N}_goal_full.txt 2>&1 &
done
wait
echo FINALISTS_DONE
