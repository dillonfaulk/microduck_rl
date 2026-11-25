#!/bin/bash
# v2 rewards (fork fb95489a): launched after the v1 chunk 1a gate confirmed 0/200 hops (2026-11-24, Dillon).
# OmniHop chunk 1a (2026-11-24): waits until the FineTurn evals free GPU0, then trains the IN-PLACE phase
# only (1000 iterations from scratch; the target curriculum widens at iteration 1000, so this run never
# widens), exports the final checkpoint and runs the gate battery (all 5 directions, drift ON and OFF,
# with the direction check -- at this point directions were never trained, so it is the baseline the
# post-widening checks are read against). The resume to 3000 is a separate step after the gate.
cd ~/microduck_rl
E=logs/evals/velstand/fineturn_2026-11-23/ckpt_evals.log
true
while pgrep -f "Mjlab-OmniHop-Flat" >/dev/null; do sleep 30; done
echo "GPU0 free $(date)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run train Mjlab-OmniHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 --agent.logger tensorboard --video True --agent.run-name v2_chunk1a > logs/omnihop_v2_chunk1a.log 2>&1
echo "v2_chunk1a exit $? $(date)"
RUN=$(ls -td logs/rsl_rl/omnihop/*_v2_chunk1a | head -1)
IT=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
onnx=$(EXPORT_TASK=Mjlab-OmniHop-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $IT) || { echo "export failed"; exit 1; }
echo "exported $onnx"
env -u PYTHONPATH uv run python scripts/hop_battery.py $onnx --episodes 20 --drift both --workers 12 \
  --out logs/evals/hop/v2_chunk1a_gate_model_$IT > logs/evals/hop/v2_chunk1a_gate.log 2>&1
echo "GATE_DONE model_$IT"
