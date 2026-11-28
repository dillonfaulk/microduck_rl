#!/bin/bash
# OmniHop shared regression on v9c2 model_7998 (Dillon 2026-11-27). CPU only.
cd ~/microduck_rl
R=logs/rsl_rl/omnihop/2026-11-27_16-38-02_v9c2
O=$R/2026-11-27_16-38-02_v9c2_model_7998.onnx
D=logs/evals/hop/regression_7998
mkdir -p $D
env -u PYTHONPATH uv run python -c "
import onnxruntime as ort, numpy as np, onnx
s = ort.InferenceSession('$O'); i, o = s.get_inputs()[0], s.get_outputs()[0]
m = onnx.load('$O'); names = [n.op_type for n in m.graph.node]
print('ONNX', i.shape, '->', o.shape, '| zero-obs out max', float(np.abs(s.run(None, {i.name: np.zeros((1, 61), np.float32)})[0]).max()),
      '| normalizer ops in graph:', sum(t in ('Sub', 'Div') for t in names))" > $D/ONNX.txt 2>&1
for dl in 0 0.25; do
  env -u PYTHONPATH uv run python scripts/hop_battery.py $O --episodes 50 --drift both --workers 12 --pulse-delay $dl --out $D/battery_delay$dl > $D/battery_delay$dl.log 2>&1
done
env -u PYTHONPATH uv run python scripts/hop_regression.py $O --episodes 20 --workers 12 --out $D > $D/regression.log 2>&1
echo ALL_DONE
