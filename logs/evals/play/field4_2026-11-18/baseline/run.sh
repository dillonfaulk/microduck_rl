#!/bin/bash
# WS2 RE-BASELINE on the 4 m field (field4.xml), BEFORE the drift / 60 s fix / get-up / bump work. One wave.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
KR=logs/rsl_rl/ball_kick_v2_right/2026-11-13_18-51-32_v2_right_chunk1/2026-11-13_18-51-32_v2_right_chunk1_model_1250.onnx
KL=logs/rsl_rl/ball_kick_v2_left/2026-11-13_19-50-52_v2_left_chunk1/2026-11-13_19-50-52_v2_left_chunk1_model_500.onnx
D=${D:-logs/evals/play/field4_2026-11-18/baseline}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
R="$V $KR --foot right --calib v2 --precision $P --estimated-only --field 4"
L="$V $KL --foot left --calib v2 --precision $P --estimated-only --field 4"
run() { name=$1; shift; env -u PYTHONPATH uv run python scripts/play.py "$@" --out $D/logs/$name > $D/$name.txt 2>&1; }
mkdir -p $D/logs
run B_right_odom $R --pose odom --runs 30 &
run B_left_odom  $L --pose odom --runs 30 &
run B_right_mapd $R --pose mapd --runs 30 &
run B_left_mapd  $L --pose mapd --runs 30 &
for ph in NAVIGATE LOOK PRECISE WATCH; do run C_fall_${ph}_right $R --pose odom --runs 20 --time-cap 150 --fall-in $ph & done
run C_fall_PRECISE_left $L --pose odom --runs 20 --time-cap 150 --fall-in PRECISE &
run C_fall_PRECISE_right_mapd $R --pose mapd --runs 20 --time-cap 150 --fall-in PRECISE &
run R_retrieval_right $R --pose odom --runs 20 --ball-out-every 30 &
wait
echo ALL_DONE
