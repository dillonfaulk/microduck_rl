#!/bin/bash
# Near-post aim fix (NP1-NP3, approved 2026-11-22) on the 4 m field, v2 kicks (BallKick v3 not trained yet).
# Wave 1: targeted near-post batteries (--spawn-near-post: every ball spawns in the attacked goal's near-post
# zone), NP off vs on and per part; the four shipping rows with NP on (the NP-off reference = the closing-work
# FINAL battery, same seeds); one NP-off shipping row that must reproduce FINAL B_right_odom exactly.
# Wave 2: the Stage 4 regression + run-39 replays, as in field4_2026-11-18/final.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
KR=logs/rsl_rl/ball_kick_v2_right/2026-11-13_18-51-32_v2_right_chunk1/2026-11-13_18-51-32_v2_right_chunk1_model_1250.onnx
KL=logs/rsl_rl/ball_kick_v2_left/2026-11-13_19-50-52_v2_left_chunk1/2026-11-13_19-50-52_v2_left_chunk1_model_500.onnx
D=logs/evals/play/near_post_2026-11-22
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
R="$V $KR --foot right --calib v2 --precision $P --estimated-only --field 4"
L="$V $KL --foot left --calib v2 --precision $P --estimated-only --field 4"
run() { name=$1; shift; env -u PYTHONPATH uv run python scripts/play.py "$@" --out $D/logs/$name > $D/$name.txt 2>&1; }
mkdir -p $D/logs
NP="--pose odom --runs 30 --spawn-near-post"
run T_right_off  $R $NP --no-near-post &
run T_right_on   $R $NP &
run T_right_np1  $R $NP --no-np-setup --no-np-line &
run T_right_np12 $R $NP --no-np-line &
run T_left_off   $L $NP --no-near-post &
run T_left_on    $L $NP &
run B_right_odom $R --pose odom --runs 30 &
run B_left_odom  $L --pose odom --runs 30 &
run B_right_mapd $R --pose mapd --runs 30 &
run B_left_mapd  $L --pose mapd --runs 30 &
run B_right_odom_npoff $R --pose odom --runs 30 --no-near-post &
wait
echo WAVE_DONE
F=logs/evals/play/field4_2026-11-18/final
if diff <(grep "^  run " $D/B_right_odom_npoff.txt) <(grep "^  run " $F/B_right_odom.txt) > /dev/null; then
  echo "B_right_odom_npoff vs FINAL B_right_odom: IDENTICAL"; else echo "B_right_odom_npoff vs FINAL B_right_odom: DIFFERS"; fi > $D/npoff_identity.txt
RD=$D/regression; mkdir -p $RD
B="$V $KR --foot right --calib v2 --precision $P --goal-dist 0.8 --scene field --runs 50"
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose gt > $RD/right_goal0.8_field_gt.txt 2>&1 &
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose odom > $RD/right_goal0.8_field_odom_stage2.txt 2>&1 &
env -u PYTHONPATH uv run python scripts/kick_mission.py $B --pose mapd --odom-calib stride_v1 --ball-src camera --goal-src camera --estimated-only > $RD/right_goal0.8_full.txt 2>&1 &
wait
FR=$F/regression
for f in right_goal0.8_field_gt.txt right_goal0.8_field_odom_stage2.txt right_goal0.8_full.txt; do
  if diff <(grep "^  run " $RD/$f) <(grep "^  run " $FR/$f) > /dev/null; then echo "$f: IDENTICAL to final/regression"; else echo "$f: DIFFERS"; fi
done > $RD/result.txt
echo ALL_DONE
