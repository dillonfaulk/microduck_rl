#!/bin/bash
# OmniHop v6 (Dillon 2026-11-25): plain resume of the widening run's model_4496, +1000 iterations @2048 envs,
# full direction range (curriculum already at its last stage). Then the per-direction battery (conditioning
# guard, end-displacement fallback) on the final checkpoint at the immediate request (runtime default) and the
# 0.25 s settle fallback, drift on and off. Re-gate before going further.
cd ~/microduck_rl
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run train Mjlab-OmniHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 --agent.logger tensorboard --video True --agent.run-name v6 \
  --agent.resume True --agent.load-run 2026-11-25_16-25-46_widen --agent.load-checkpoint model_4496.pt > logs/omnihop_v6.log 2>&1
echo "v6 exit $? $(date)"
RUN=$(ls -td logs/rsl_rl/omnihop/*_v6 | head -1)
LAST=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
for IT in 5000 $LAST; do
  onnx=$(EXPORT_TASK=Mjlab-OmniHop-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $IT) || { echo "export $IT failed"; continue; }
  for dl in 0 0.25; do
    env -u PYTHONPATH uv run python scripts/hop_battery.py $onnx --episodes 20 --drift both --workers 12 --pulse-delay $dl \
      --out logs/evals/hop/v6_model_${IT}_delay$dl > logs/evals/hop/v6_${IT}_delay$dl.log 2>&1
  done
  echo "battery $IT done"
done
echo "ALL_DONE"
