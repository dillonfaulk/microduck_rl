#!/bin/bash
# Per checkpoint as training writes it: export (scripts/export.py) then sweep.sh. Report at the end.
cd ~/microduck_rl
R=logs/rsl_rl/ball_kick_v3_right/2026-11-22_14-22-48_v3_right_ft
D=logs/evals/ballkick/ballkick_v3_eval_2026-11-22
for N in 1375 1500 1625 1750 1875 1999; do
  until [ -f $R/model_$N.pt ]; do sleep 20; done
  sleep 15
  [ -f $R/v3_right_ft_model_$N.onnx ] || CUDA_VISIBLE_DEVICES=0 MICRODUCK_BALLKICK_V3=1 MICRODUCK_BALL_TARGET_SPEED=0.25 \
    env -u PYTHONPATH uv run scripts/export.py Mjlab-BallKick-Flat-MicroDuck --checkpoint-file $R/model_$N.pt \
    --onnx-file $R/v3_right_ft_model_$N.onnx > $D/export_$N.log 2>&1
  bash $D/sweep.sh v3R$N $R/v3_right_ft_model_$N.onnx right
  python3 $D/report.py v2R1250 v2L500 v3R1375 v3R1500 v3R1625 v3R1750 v3R1875 v3R1999 > $D/SWEEP.txt 2>&1
  echo "swept $N"
done
