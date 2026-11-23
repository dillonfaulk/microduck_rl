#!/bin/bash
# Stage A sweep: for each saved checkpoint, export ONNX, run the head-under battery (clamp on, as trained) and
# StandUp's own get-up eval in the training sim (neck clip on, head-under spawns removed).
cd ~/microduck_rl
B=logs/evals/velstand/headjam_2026-11-22
OUT=$B/stageA
RUN=$(ls -td logs/rsl_rl/microduck_stand/*_headjamA | head -1)
echo "run $RUN"
mkdir -p $OUT/battery $OUT/warp
for hj in; do CUDA_VISIBLE_DEVICES=0 MICRODUCK_HEADJAM=$hj env -u PYTHONPATH uv run python logs/tools/warp_standup_eval.py logs/rsl_rl/microduck_stand/2026-11-01_17-21-18_spincap_chunk2/model_11000.pt 1024 2>&1 | grep -A6 "^checkpoint" > $OUT/warp/ref11000_headjam$hj.txt; done
for it in 11250 11500 11750 12000 12250 12499; do
  until [ -f $RUN/model_$it.pt ] || ! pgrep -f "run-name headjamA" >/dev/null; do sleep 30; done
  [ -f $RUN/model_$it.pt ] || { echo "training gone before model_$it"; exit 1; }
  sleep 20
  onnx=$(EXPORT_TASK=Mjlab-StandUp-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $it) || { echo "export $it failed"; continue; }
  env -u PYTHONPATH uv run python $B/battery.py $onnx --clamp --workers 8 --out $OUT/battery/A_$it.json 2>&1 | tail -2 | tee -a $OUT/SWEEP.txt
  CUDA_VISIBLE_DEVICES=0 MICRODUCK_HEADJAM=1 env -u PYTHONPATH uv run python logs/tools/warp_standup_eval.py $RUN/model_$it.pt 1024 2>&1 \
    | grep -A6 "^checkpoint" > $OUT/warp/A_$it.txt
  cat $OUT/warp/A_$it.txt | tee -a $OUT/SWEEP.txt
  echo "=== done $it" | tee -a $OUT/SWEEP.txt
done
