#!/bin/bash
# usage: logs/tools/export_ckpt.sh <run_dir> <iteration> [env vars...]: export model_<it>.pt -> <run_dir>/<run>_model_<it>.onnx
set -e
d=$1; it=$2; name=$(basename $d)
tmp=$(mktemp -d); cd ~/microduck_rl
exec 9>logs/.export.lock; flock 9   # export.py always writes ./output.onnx: serialize concurrent exports
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" uv run scripts/export.py Mjlab-BallKick-Flat-MicroDuck --checkpoint-file $d/model_$it.pt > $tmp/export.log 2>&1 || { tail -5 $tmp/export.log; exit 1; }
mv output.onnx $d/${name}_model_$it.onnx && echo $d/${name}_model_$it.onnx
