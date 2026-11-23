#!/bin/bash
# Stage B sweep: per checkpoint, export ONNX, head-under battery (clamp on, as trained), and round 3's
# training-sim get-up eval (warp_velstand_eval, neck clip on, head-under spawns removed).
cd ~/microduck_rl
B=logs/evals/velstand/headjam_2026-11-22
OUT=$B/stageB
RUN=$(ls -td logs/rsl_rl/velstand/*_headjamB | head -1)
echo "run $RUN"
mkdir -p $OUT/battery $OUT/warp
for it in 4250 4500 4750 5000 5250 5498; do
  until [ -f $RUN/model_$it.pt ] || ! pgrep -f "run-name headjamB" >/dev/null; do sleep 30; done
  [ -f $RUN/model_$it.pt ] || { echo "training gone before model_$it"; exit 1; }
  sleep 20
  onnx=$(EXPORT_TASK=Mjlab-VelStand-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $it) || { echo "export $it failed: $onnx"; continue; }
  env -u PYTHONPATH uv run python $B/battery.py $onnx --clamp --workers 8 --out $OUT/battery/B_$it.json 2>&1 | tail -2 | tee -a $OUT/SWEEP.txt
  CUDA_VISIBLE_DEVICES=0 MICRODUCK_HEADJAM=1 env -u PYTHONPATH uv run python logs/tools/warp_velstand_eval.py $RUN/model_$it.pt 1024 2>&1 \
    | grep -A4 "^checkpoint" > $OUT/warp/B_$it.txt
  cat $OUT/warp/B_$it.txt | tee -a $OUT/SWEEP.txt
  echo "=== done $it" | tee -a $OUT/SWEEP.txt
done
