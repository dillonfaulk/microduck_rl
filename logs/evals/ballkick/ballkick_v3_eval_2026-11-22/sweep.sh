#!/bin/bash
# BallKick v3 right (run plan microduck-ballkick-v3-run-plan-2026-11-22.md, approved 2026-11-22): per-checkpoint sweep.
#   uncal missions 2 x 30 (seeds 0, 1; aim 0, stance 0) -- v2's protocol (ballkick_v2_eval_2026-11-13)
#   lateral sweep: KICK_LATERAL_SHIFT -8/-4/0/+4/+8 mm x 20 missions (gt pose: align_ey = TRUE placement)
# Usage: sweep.sh <tag> <onnx> <foot>    (references: v2 right 1250, v2 left 500 get the same lateral sweep)
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
D=logs/evals/ballkick/ballkick_v3_eval_2026-11-22
tag=$1; onnx=$2; foot=${3:-right}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
if [ "$foot" = right ]; then U="KICK_STANCE_SHIFT=0 KICK_AIM_OFFSET_DEG=0"; else U="KICK_STANCE_SHIFT_LEFT=0 KICK_AIM_OFFSET_DEG_LEFT=0"; fi
m() { env $U "$@"; }
for s in 0 1; do
  m env -u PYTHONPATH uv run python scripts/kick_mission.py $V $onnx --foot $foot --precision $P --runs 30 --seed $s \
    > $D/${tag}_uncal_s$s.txt 2>&1 &
done
for sh in -0.008 -0.004 0.0 0.004 0.008; do
  m KICK_LATERAL_SHIFT=$sh env -u PYTHONPATH uv run python scripts/kick_mission.py $V $onnx --foot $foot --precision $P \
    --runs 20 --seed 7 > $D/${tag}_lat${sh}.txt 2>&1 &
done
wait
