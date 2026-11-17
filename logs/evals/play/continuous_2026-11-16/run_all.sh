#!/bin/bash
# Continuous-play batteries (brief: microduck-continuous-play-brief.md, Workstreams A-C). All --estimated-only.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
KR=logs/rsl_rl/ball_kick_v2_right/2026-11-13_18-51-32_v2_right_chunk1/2026-11-13_18-51-32_v2_right_chunk1_model_1250.onnx
KL=logs/rsl_rl/ball_kick_v2_left/2026-11-13_19-50-52_v2_left_chunk1/2026-11-13_19-50-52_v2_left_chunk1_model_500.onnx
D=logs/evals/play/continuous_2026-11-16
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
R="$V $KR --foot right --calib v2 --precision $P --estimated-only"
L="$V $KL --foot left --calib v2 --precision $P --estimated-only"
run() { name=$1; shift; env -u PYTHONPATH uv run python scripts/play.py "$@" --out $D/logs/$name > $D/$name.txt 2>&1; }
mkdir -p $D/logs
# B: continuous play, 30 runs x 240 s
run B_right_odom $R --pose odom --runs 30 &
run B_left_odom  $L --pose odom --runs 30 &
run B_right_odom_noroutine $R --pose odom --runs 30 --no-routine-fix &
run B_right_mapd $R --pose mapd --runs 30 &
run B_left_mapd  $L --pose mapd --runs 30 &
# C: scripted falls, 20 runs x 150 s each
for ph in NAVIGATE LOOK PRECISE WATCH; do run C_fall_${ph}_right $R --pose odom --runs 20 --time-cap 150 --fall-in $ph & done
run C_fall_PRECISE_left $L --pose odom --runs 20 --time-cap 150 --fall-in PRECISE &
run C_fall_PRECISE_right_mapd $R --pose mapd --runs 20 --time-cap 150 --fall-in PRECISE &
wait
# behaviour cases, 10 runs each
run beh_ball_hidden_after_respawn $R --pose odom --runs 10 --hide-ball-after-respawn &
run beh_goal_hidden $R --pose odom --runs 10 --hide-goal all &
run beh_pose_lost_permanent $R --pose odom --runs 10 --dropout 60,100000 &
run beh_pose_dropout_2s $R --pose odom --runs 10 --dropout 60,2 &
wait
MODES=reloc logs/evals/play/continuous_2026-11-16/run39_replay/run.sh > /dev/null 2>&1
echo ALL_DONE
