#!/bin/bash
# FineTurn per-checkpoint evals (2026-11-24). Waits for the training run to exit, then:
#  1. export every checkpoint (4500..5500 + final) to ONNX
#  2. CPU low-end command sweep (sweep.py), all checkpoints + shipped 4250 as baseline, friction x1.0 and x1.5
#  3. head-under battery (exact stall + 199 spawns, clip on) per checkpoint
#  4. training-sim get-ups (warp_velstand_eval) + rise-only spin (warp_standup_eval; face-up peak vs 10.7) per checkpoint
cd ~/microduck_rl
B=logs/evals/velstand/fineturn_2026-11-23
OUT=$B/ckpts
RUN=$(ls -td logs/rsl_rl/velstand/*_fineturn | head -1)
echo "run $RUN"
while pgrep -f "run-name fineturn " >/dev/null; do sleep 30; done
mkdir -p $OUT/battery $OUT/warp
ITS=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | awk '$1>4250')
echo "checkpoints: $ITS"
POLS='{"hj4250": "logs/rsl_rl/velstand/2026-11-22_19-47-54_headjamB/2026-11-22_19-47-54_headjamB_model_4250.onnx"'
for it in $ITS; do
  onnx=$(EXPORT_TASK=Mjlab-VelStand-Flat-MicroDuck MICRODUCK_HEADJAM=1 MICRODUCK_FINETURN=1 bash logs/tools/export_ckpt.sh $RUN $it) || { echo "export $it failed"; continue; }
  # the ONNX name must contain "headjam" for the runtime neck-clip auto-switch: rename
  hj=${onnx%.onnx}; hj=$(dirname $onnx)/fineturn_headjam_model_$it.onnx; mv $onnx $hj
  POLS="$POLS, \"ft$it\": \"$hj\""
done
POLS="$POLS}"
echo "$POLS" > $OUT/pols.json
SWEEP_POLS="$POLS" SWEEP_OUT=$OUT/sweep_f1.0 env -u PYTHONPATH uv run python $B/sweep.py > $OUT/sweep_f1.0.log 2>&1
SWEEP_POLS="$POLS" SWEEP_FRICTION=1.5 SWEEP_OUT=$OUT/sweep_f1.5 env -u PYTHONPATH uv run python $B/sweep.py > $OUT/sweep_f1.5.log 2>&1
echo "sweeps done"
for it in $ITS; do
  hj=$RUN/fineturn_headjam_model_$it.onnx
  env -u PYTHONPATH uv run python logs/evals/velstand/headjam_2026-11-22/battery.py $hj --clamp --workers 12 --out $OUT/battery/ft_$it.json 2>&1 | tail -2 > $OUT/battery/ft_$it.txt
  CUDA_VISIBLE_DEVICES=0 MICRODUCK_HEADJAM=1 env -u PYTHONPATH uv run python logs/tools/warp_velstand_eval.py $RUN/model_$it.pt 1024 > $OUT/warp/getup_$it.log 2>&1
  CUDA_VISIBLE_DEVICES=0 MICRODUCK_HEADJAM=1 env -u PYTHONPATH uv run python logs/tools/warp_standup_eval.py $RUN/model_$it.pt 1024 > $OUT/warp/spin_$it.log 2>&1
  echo "=== done $it"
done
echo ALL_DONE
