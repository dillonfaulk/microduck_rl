#!/bin/bash
# FINAL on the 4 m field: WS3 (corner posts, 60 s timer fix, pre-kick look, post detours), WS4 (get-up failure),
# WS5 (bump persistence) all on. One wave, then the regression checks.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
KR=logs/rsl_rl/ball_kick_v2_right/2026-11-13_18-51-32_v2_right_chunk1/2026-11-13_18-51-32_v2_right_chunk1_model_1250.onnx
KL=logs/rsl_rl/ball_kick_v2_left/2026-11-13_19-50-52_v2_left_chunk1/2026-11-13_19-50-52_v2_left_chunk1_model_500.onnx
D=logs/evals/play/field4_2026-11-18/final
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
R="$V $KR --foot right --calib v2 --precision $P --estimated-only --field 4"
L="$V $KL --foot left --calib v2 --precision $P --estimated-only --field 4"
run() { name=$1; shift; env -u PYTHONPATH uv run python scripts/play.py "$@" --out $D/logs/$name > $D/$name.txt 2>&1; }
mkdir -p $D/logs
run B_right_odom $R --pose odom --runs 30 &
run B_left_odom  $L --pose odom --runs 30 &
run B_right_mapd $R --pose mapd --runs 30 &
run B_left_mapd  $L --pose mapd --runs 30 &
run A_right_odom_no_corners $R --pose odom --runs 30 --no-corners &
run A_right_odom_no_timer_prekick $R --pose odom --runs 30 --no-timer-fix --no-pre-kick-fix &
for ph in NAVIGATE LOOK PRECISE WATCH; do run C_fall_${ph}_right $R --pose odom --runs 20 --time-cap 150 --fall-in $ph & done
run C_fall_PRECISE_left $L --pose odom --runs 20 --time-cap 150 --fall-in PRECISE &
run C_fall_PRECISE_right_mapd $R --pose mapd --runs 20 --time-cap 150 --fall-in PRECISE &
run R_retrieval_right $R --pose odom --runs 20 --ball-out-every 30 &
run G_getup_NAVIGATE_right $R --pose odom --runs 30 --fall-in NAVIGATE --falls-per-run 5 &
run G_getup_PRECISE_left $L --pose odom --runs 30 --fall-in PRECISE --falls-per-run 5 &
run beh_ball_hidden_after_respawn $R --pose odom --runs 10 --hide-ball-after-respawn &
run beh_goal_hidden $R --pose odom --runs 10 --hide-goal all &
run beh_pose_lost_permanent $R --pose odom --runs 10 --dropout 60,100000 &
run beh_pose_dropout_2s $R --pose odom --runs 10 --dropout 60,2 &
wait
echo WAVE_DONE
# regression: Stage 4 paths (must be identical to Stage 4) + run-39 replays with the protocol (re-based)
RD=$D/regression; mkdir -p $RD
B="$V $KR --foot right --calib v2 --precision $P --goal-dist 0.8 --scene field --runs 50"
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose gt > $RD/right_goal0.8_field_gt.txt 2>&1 &
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose odom > $RD/right_goal0.8_field_odom_stage2.txt 2>&1 &
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose mapd --odom-calib stride_v1 --ball-src camera --goal-src camera --estimated-only > $RD/right_goal0.8_full.txt 2>&1 &
for row in "right $KR 0.5" "right $KR 0.8" "left $KL 0.5" "left $KL 0.8"; do
  set -- $row
  env -u PYTHONPATH uv run python scripts/kick_mission.py $V $2 --foot $1 --calib v2 --precision $P --goal-dist $3 \
    --scene field --pose odom --odom-calib stride_v1 --ball-src camera --goal-src camera --estimated-only \
    --runs 40 --only-runs 39 --reloc > $RD/run39_${1}_goal$3_reloc.txt 2>&1 &
done
wait
S=logs/evals/vision/stage4_2026-11-16
for pair in "right_goal0.8_field_gt.txt regression/right_goal0.8_field_gt.txt" "right_goal0.8_field_odom_stage2.txt regression/right_goal0.8_field_odom_stage2.txt" "right_goal0.8_full.txt full/right_goal0.8.txt"; do
  set -- $pair
  if diff <(grep "^  run " $RD/$1 | sed 's/, mapd_relocalizations=0//') <(grep "^  run " $S/$2) > /dev/null; then echo "$1: IDENTICAL ($(grep -c '^  run ' $RD/$1) runs)"; else echo "$1: DIFFERS"; fi
done > $RD/result.txt
echo ALL_DONE
