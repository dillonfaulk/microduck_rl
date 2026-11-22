#!/bin/bash
# BallKick v3 right 1875 (--calib v3) in the soccer stack on the 4 m field, NP1 default; compare to
# near_post_2026-11-22/B_right_{odom,mapd}_np1 (v2 right, same seeds, same NP1 defaults). Right foot only changes.
cd ~/microduck_rl
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step_v2/2026-11-13_16-23-21_v2_chunk1/2026-11-13_16-23-21_v2_chunk1_model_2500.onnx
K3=logs/rsl_rl/ball_kick_v3_right/2026-11-22_14-22-48_v3_right_ft/v3_right_ft_model_1875.onnx
D=logs/evals/play/v3_soccer_2026-11-22
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
R="$V $K3 --foot right --calib v3 --precision $P --estimated-only --field 4"
run() { name=$1; shift; env -u PYTHONPATH uv run python scripts/play.py "$@" --out $D/logs/$name > $D/$name.txt 2>&1; }
mkdir -p $D/logs
run B_right_odom $R --pose odom --runs 30 &
run B_right_mapd $R --pose mapd --runs 30 &
run T_right_np1  $R --pose odom --runs 30 --spawn-near-post &
wait
echo DONE
