#!/bin/bash
# OmniHop widening (Dillon 2026-11-25): plain resume of v5 model_2997, +1500 iterations @2048 envs. The target
# curriculum widens at iteration 3000 (half range, 40 % in place) and 3500 (fwd/back <= 40 mm, side <= 30 mm,
# 25 % in place). Then the per-direction battery (conditioning guard) on checkpoints 3500 / 4000 / final at the
# runtime-default immediate request (delay 0) and the 0.25 s settle fallback.
cd ~/microduck_rl
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run train Mjlab-OmniHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 1500 --agent.logger tensorboard --video True --agent.run-name widen \
  --agent.resume True --agent.load-run 2026-11-24_23-55-04_v5 --agent.load-checkpoint model_2997.pt > logs/omnihop_widen.log 2>&1
echo "widen exit $? $(date)"
RUN=$(ls -td logs/rsl_rl/omnihop/*_widen | head -1)
LAST=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
for IT in 3500 4000 $LAST; do
  onnx=$(EXPORT_TASK=Mjlab-OmniHop-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $IT) || { echo "export $IT failed"; continue; }
  for dl in 0 0.25; do
    env -u PYTHONPATH uv run python scripts/hop_battery.py $onnx --episodes 20 --drift both --workers 12 --pulse-delay $dl \
      --out logs/evals/hop/widen_model_${IT}_delay$dl > logs/evals/hop/widen_${IT}_delay$dl.log 2>&1
  done
  echo "battery $IT done"
done
echo "ALL_DONE"
