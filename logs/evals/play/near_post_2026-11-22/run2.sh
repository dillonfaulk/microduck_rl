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
# Wave 2 (tuning the NP2 cut-off, Dillon: "keep 50 deg and adjust if necessary"): wave 1 showed NP1 alone scoring
# the most near-post goals on the right (89 vs 57 with NP1-3 at 50 deg): the taps cost a whole approach each.
NP="--pose odom --runs 30 --spawn-near-post"
run T_left_np1        $L $NP --no-np-setup --no-np-line &
run T_right_on70      $R $NP --np-oblique-deg 70 &
run T_left_on70       $L $NP --np-oblique-deg 70 &
run T_right_stance    $R $NP --np-oblique-deg 180 &
run T_left_stance     $L $NP --np-oblique-deg 180 &
run B_right_odom_np1  $R --pose odom --runs 30 --no-np-setup --no-np-line &
run B_left_odom_np1   $L --pose odom --runs 30 --no-np-setup --no-np-line &
run B_right_mapd_np1  $R --pose mapd --runs 30 --no-np-setup --no-np-line &
run B_left_mapd_np1   $L --pose mapd --runs 30 --no-np-setup --no-np-line &
wait
echo WAVE2_DONE
