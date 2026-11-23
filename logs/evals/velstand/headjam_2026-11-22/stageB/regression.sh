#!/bin/bash
# HeadJamRecover stage B winner regression. usage: regression.sh <velstand winner .onnx> <winner .pt> <tag>
# 1. natural falls: the closing-work final wave's C rows (6 x 20, one fall each) + G rows (2 x 30 x 5 falls) -- 417 falls;
#    bar 0 get-up failures. Today's stack: v3 right (--calib v3), v2 left, NP1 defaults, runtime neck clip on.
# 2. soccer B rows (4 x 30 x 240 s) vs today's NP1 rows (v3 right / v2 left)
# 3. CPU rehearsal fallen suite (round 3: 48/48), training-sim get-ups (warp_velstand_eval), rise-only spin
#    (warp_standup_eval actor-only, face-up peak <= 10.7), walking response (warp_cmd_response, +-10% of round 3)
cd ~/microduck_rl
V=$1; PT=$2; TAG=$3
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
K3=logs/rsl_rl/ball_kick_v3_right/2026-11-22_14-22-48_v3_right_ft/v3_right_ft_model_1875.onnx
KL=logs/rsl_rl/ball_kick_v2_left/2026-11-13_19-50-52_v2_left_chunk1/2026-11-13_19-50-52_v2_left_chunk1_model_500.onnx
D=logs/evals/velstand/headjam_2026-11-22/stageB/regression_$TAG
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2 MICRODUCK_NECK_CLIP=1
R="$V $K3 --foot right --calib v3 --precision $P --estimated-only --field 4"
L="$V $KL --foot left --calib v2 --precision $P --estimated-only --field 4"
run() { name=$1; shift; env -u PYTHONPATH uv run python scripts/play.py "$@" --out $D/logs/$name > $D/$name.txt 2>&1; }
mkdir -p $D/logs
for ph in NAVIGATE LOOK PRECISE WATCH; do run C_fall_${ph}_right $R --pose odom --runs 20 --time-cap 150 --fall-in $ph & done
run C_fall_PRECISE_left $L --pose odom --runs 20 --time-cap 150 --fall-in PRECISE &
run C_fall_PRECISE_right_mapd $R --pose mapd --runs 20 --time-cap 150 --fall-in PRECISE &
run G_getup_NAVIGATE_right $R --pose odom --runs 30 --fall-in NAVIGATE --falls-per-run 5 &
run G_getup_PRECISE_left $L --pose odom --runs 30 --fall-in PRECISE --falls-per-run 5 &
run B_right_odom $R --pose odom --runs 30 &
run B_right_mapd $R --pose mapd --runs 30 &
run B_left_odom  $L --pose odom --runs 30 &
run B_left_mapd  $L --pose mapd --runs 30 &
env -u PYTHONPATH uv run python logs/tools/headless_rehearsal.py $V --kind walking --suite all --delay 1 --train-solver \
  --scene src/mjlab_microduck/robot/microduck/scene_allcollisions.xml > $D/cpu_rehearsal.txt 2>&1 &
wait
export MICRODUCK_HEADJAM=1 CUDA_VISIBLE_DEVICES=0
env -u PYTHONPATH uv run python logs/tools/warp_velstand_eval.py $PT 1024 > $D/warp_velstand_eval.txt 2>&1
env -u PYTHONPATH uv run python logs/tools/warp_standup_eval.py $PT 1024 > $D/warp_standup_eval_riseonly.txt 2>&1
env -u PYTHONPATH uv run python logs/tools/warp_cmd_response.py $PT 20 > $D/response.txt 2>&1
echo ALL_DONE
