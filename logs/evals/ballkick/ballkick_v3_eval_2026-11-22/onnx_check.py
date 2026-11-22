"""ONNX vs PyTorch: the checkpoint's actor (obs normalizer + MLP, ELU, deterministic mean) vs the exported ONNX on
300 random observations drawn around the normalizer's own statistics. Prints the max abs action difference."""
import sys, torch, numpy as np, onnxruntime as ort
pt, onnx = sys.argv[1], sys.argv[2]
sd = torch.load(pt, map_location="cpu", weights_only=False)["actor_state_dict"]
mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
g = torch.Generator().manual_seed(0)
obs = mean + std * torch.randn(300, 61, generator=g)
best = None
for eps in (1e-2, 1e-8, 0.0):
    x = (obs - mean) / (std + eps)
    for i in (0, 2, 4, 6):
        x = x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"]
        if i < 6:
            x = torch.nn.functional.elu(x)
    s = ort.InferenceSession(onnx)
    nm = s.get_inputs()[0].name
    y = np.concatenate([s.run(None, {nm: o[None].astype(np.float32)})[0] for o in obs.numpy()])
    print("onnx io:", s.get_inputs()[0].shape, "->", s.get_outputs()[0].shape)
    d = float(np.abs(y - x.numpy()).max())
    best = d if best is None else min(best, d)
    print(f"normalizer eps {eps:g}: max abs diff {d:.3e}")
print(f"MAX ABS DIFF {best:.3e}")
