#!/bin/bash
# SCAN clearance (default on, 4 m field): 4 soccer rows x {4250 + neck clip, round 3}, same seeds as the regression
# (4250, no clearance) and today's reference rows (round 3, no clearance).
cd ~/microduck_rl
H=logs/rsl_rl/velstand/2026-11-22_19-47-54_headjamB/2026-11-22_19-47-54_headjamB_model_4250.onnx
R3=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
K3=logs/rsl_rl/ball_kick_v3_right/2026-11-22_14-22-48_v3_right_ft/v3_right_ft_model_1875.onnx
KL=logs/rsl_rl/ball_kick_v2_left/2026-11-13_19-50-52_v2_left_chunk1/2026-11-13_19-50-52_v2_left_chunk1_model_500.onnx
D=logs/evals/velstand/headjam_2026-11-22/stageB/scan_clear
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
run() { name=$1; clip=$2; shift 2; MICRODUCK_NECK_CLIP=$clip env -u PYTHONPATH uv run python scripts/play.py "$@" --out $D/logs/$name > $D/$name.txt 2>&1; }
for pol in "hj4250 $H 1" "round3 $R3 0"; do
  set -- $pol; tag=$1; V=$2; clip=$3
  run ${tag}_B_right_odom $clip $V $K3 --foot right --calib v3 --precision $P --estimated-only --field 4 --pose odom --runs 30 &
  run ${tag}_B_right_mapd $clip $V $K3 --foot right --calib v3 --precision $P --estimated-only --field 4 --pose mapd --runs 30 &
  run ${tag}_B_left_odom  $clip $V $KL --foot left --calib v2 --precision $P --estimated-only --field 4 --pose odom --runs 30 &
  run ${tag}_B_left_mapd  $clip $V $KL --foot left --calib v2 --precision $P --estimated-only --field 4 --pose mapd --runs 30 &
done
wait
echo ALL_DONE
