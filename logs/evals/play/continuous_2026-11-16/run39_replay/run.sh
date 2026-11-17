#!/bin/bash
# Exact replay of Stage 4's odom-only run 39 (the PrecisionStep fall), without and with the fall protocol.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
KR=logs/rsl_rl/ball_kick_v2_right/2026-11-13_18-51-32_v2_right_chunk1/2026-11-13_18-51-32_v2_right_chunk1_model_1250.onnx
KL=logs/rsl_rl/ball_kick_v2_left/2026-11-13_19-50-52_v2_left_chunk1/2026-11-13_19-50-52_v2_left_chunk1_model_500.onnx
O=logs/evals/play/continuous_2026-11-16/run39_replay
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
for row in "right $KR 0.5" "right $KR 0.8" "left $KL 0.5" "left $KL 0.8"; do
  set -- $row
  for mode in ${MODES:-plain reloc}; do
    extra=""; [ $mode = reloc ] && extra="--reloc"
    env -u PYTHONPATH uv run python scripts/kick_mission.py $V $2 --foot $1 --calib v2 --precision $P --goal-dist $3 \
      --scene field --pose odom --odom-calib stride_v1 --ball-src camera --goal-src camera --estimated-only \
      --runs 40 --only-runs 39 $extra > $O/${1}_goal$3_$mode.txt 2>&1 &
  done
done
wait
echo ALL_DONE
