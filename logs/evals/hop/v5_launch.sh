#!/bin/bash
# OmniHop v5 (Dillon 2026-11-25): plain resume of v4 model_1998 (crouch fix + stronger imitation), +1000 in-place iterations @2048 envs
# (IN_PLACE_ITERS now 3000, so it never widens), then the gate battery TWICE: request immediately at the
# swap (--pulse-delay 0, tests the trained swap-in fix) and after the 0.25 s runtime settle (the runtime path).
cd ~/microduck_rl
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run train Mjlab-OmniHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 --agent.logger tensorboard --video True --agent.run-name v5 \
  --agent.resume True --agent.load-run 2026-11-24_23-05-33_v4 --agent.load-checkpoint model_1998.pt > logs/omnihop_v5.log 2>&1
echo "v5 exit $? $(date)"
RUN=$(ls -td logs/rsl_rl/omnihop/*_v5 | head -1)
IT=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
onnx=$(EXPORT_TASK=Mjlab-OmniHop-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $IT) || { echo "export failed"; exit 1; }
echo "exported $onnx"
for dl in 0 0.25; do
  env -u PYTHONPATH uv run python scripts/hop_battery.py $onnx --episodes 20 --drift both --workers 12 --pulse-delay $dl \
    --out logs/evals/hop/v5_gate_model_${IT}_delay$dl > logs/evals/hop/v5_gate_delay$dl.log 2>&1
  echo "battery delay $dl done"
done
echo "GATE_DONE model_$IT"
