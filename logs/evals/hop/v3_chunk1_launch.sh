#!/bin/bash
# OmniHop v3 chunk 1a (fork v3 commit): waits for the v2 chunk 1a gate battery (GATE_DONE), then the same
# split as before -- 1000 in-place iterations @2048 envs from scratch, export, gate battery.
cd ~/microduck_rl
until grep -q GATE_DONE logs/evals/hop/v2_chunk1_launch.log 2>/dev/null; do sleep 20; done
echo "v2 gate done, GPU0 $(date)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run train Mjlab-OmniHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 --agent.logger tensorboard --video True --agent.run-name v3_chunk1a > logs/omnihop_v3_chunk1a.log 2>&1
echo "v3_chunk1a exit $? $(date)"
RUN=$(ls -td logs/rsl_rl/omnihop/*_v3_chunk1a | head -1)
IT=$(ls $RUN | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
onnx=$(EXPORT_TASK=Mjlab-OmniHop-Flat-MicroDuck bash logs/tools/export_ckpt.sh $RUN $IT) || { echo "export failed"; exit 1; }
echo "exported $onnx"
env -u PYTHONPATH uv run python scripts/hop_battery.py $onnx --episodes 20 --drift both --workers 12 \
  --out logs/evals/hop/v3_chunk1a_gate_model_$IT > logs/evals/hop/v3_chunk1a_gate.log 2>&1
echo "GATE_DONE model_$IT"
