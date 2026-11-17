#!/bin/bash
# Stage 4 paths after the play hooks (Mission play=None): must be line-for-line identical to Stage 4.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
KR=logs/rsl_rl/ball_kick_v2_right/2026-11-13_18-51-32_v2_right_chunk1/2026-11-13_18-51-32_v2_right_chunk1_model_1250.onnx
D=logs/evals/play/continuous_2026-11-16/regression
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
B="$V $KR --foot right --calib v2 --precision $P --goal-dist 0.8 --scene field --runs 50"
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose gt > $D/right_goal0.8_field_gt.txt 2>&1 &
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose odom > $D/right_goal0.8_field_odom_stage2.txt 2>&1 &
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose mapd --odom-calib stride_v1 --ball-src camera --goal-src camera --estimated-only > $D/right_goal0.8_full.txt 2>&1 &
wait
S=logs/evals/vision/stage4_2026-11-16
for pair in "right_goal0.8_field_gt.txt regression/right_goal0.8_field_gt.txt" "right_goal0.8_field_odom_stage2.txt regression/right_goal0.8_field_odom_stage2.txt" "right_goal0.8_full.txt full/right_goal0.8.txt"; do
  set -- $pair
  if diff <(grep "^  run " $D/$1) <(grep "^  run " $S/$2) > /dev/null; then echo "$1: IDENTICAL ($(grep -c '^  run ' $D/$1) runs)"; else echo "$1: DIFFERS"; fi
done
