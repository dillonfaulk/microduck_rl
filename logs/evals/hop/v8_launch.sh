#!/bin/bash
# OmniHop v8 (Dillon + n3z-jr, 2026-11-25): hold-still penalties after ANY touchdown once the window closes, plain resume
# of v7 model_6500 (the banked candidate), +1000 iterations @2048 envs. Re-gate: per-direction battery
# (both request timings, drift on/off, end-position fallback) + the phase diagnostic (slide = end - lift-off -
# flight) on checkpoints 6500 and final.
cd ~/microduck_rl
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run train Mjlab-OmniHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 --agent.logger tensorboard --video True --agent.run-name v8 \
  --agent.resume True --agent.load-run 2026-11-25_19-09-19_v7 --agent.load-checkpoint model_6500.pt > logs/omnihop_v8.log 2>&1
echo "v8 exit $? $(date)"
RUN=$(ls -td logs/rsl_rl/omnihop/*_v8 | head -1)
LAST=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
for IT in 7000 $LAST; do
  onnx=$(EXPORT_TASK=Mjlab-OmniHop-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $IT) || { echo "export $IT failed"; continue; }
  for dl in 0 0.25; do
    env -u PYTHONPATH uv run python scripts/hop_battery.py $onnx --episodes 20 --drift both --workers 12 --pulse-delay $dl \
      --out logs/evals/hop/v8_model_${IT}_delay$dl > logs/evals/hop/v8_${IT}_delay$dl.log 2>&1
    env -u PYTHONPATH uv run python logs/evals/hop/phase_diag.py $onnx $dl > logs/evals/hop/v8_model_${IT}_delay$dl/PHASES.txt 2>&1
  done
  echo "battery $IT done"
done
echo "ALL_DONE"
