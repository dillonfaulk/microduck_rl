#!/bin/bash
# OmniHop v9 chunk 2 (Dillon 2026-11-27: LAST chunk on the back question; stop condition back < ~10 mm at the feet): same recipe,
# (feet), trunk-motion penalty dropped, progress paid on airborne frames only. Plain resume of v6c2 model_6000
# (resume of v9 model_6999), +1000 iterations @2048 envs.
# Re-gate: per-direction battery (feet yardstick; feet-vs-trunk + touchdown/end/steps phases), both request
# timings, drift on and off, on checkpoints 6500 and final. Fallback: v7 model_6500 @ immediate.
cd ~/microduck_rl
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run train Mjlab-OmniHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 --agent.logger tensorboard --video True --agent.run-name v9c2 \
  --agent.resume True --agent.load-run 2026-11-27_15-23-25_v9 --agent.load-checkpoint model_6999.pt > logs/omnihop_v9c2.log 2>&1
echo "v9c2 exit $? $(date)"
RUN=$(ls -td logs/rsl_rl/omnihop/*_v9c2 | head -1)
LAST=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
for IT in 7500 $LAST; do
  onnx=$(EXPORT_TASK=Mjlab-OmniHop-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $IT) || { echo "export $IT failed"; continue; }
  for dl in 0 0.25; do
    env -u PYTHONPATH uv run python scripts/hop_battery.py $onnx --episodes 20 --drift both --workers 12 --pulse-delay $dl \
      --out logs/evals/hop/v9c2_model_${IT}_delay$dl > logs/evals/hop/v9c2_${IT}_delay$dl.log 2>&1
  done
  echo "battery $IT done"
done
echo "ALL_DONE"
