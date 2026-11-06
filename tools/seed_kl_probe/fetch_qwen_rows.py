# Fetch a few expert rows of layer 20 of Qwen3.8-Flash-Next through HTTP ranges (BF16 -> f32 .bin).
import json, struct, subprocess, sys, numpy as np
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
BASE = f"https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{REV}/"
out = sys.argv[1]
def rng(f, a, b):
    return subprocess.run(["curl", "-sSL", "-r", f"{a}-{b}", BASE + f], check=True, capture_output=True).stdout
def tensor(f, name):
    n = struct.unpack("<Q", rng(f, 0, 7))[0]
    h = json.loads(rng(f, 8, 8 + n - 1))
    t = h[name]
    return 8 + n, t
for f, name, tag in [("model-00061-of-00131.safetensors", "model.language_model.layers.20.mlp.experts.gate_up_proj", "gu"),
                     ("model-00062-of-00131.safetensors", "model.language_model.layers.20.mlp.experts.down_proj", "dn")]:
    base, t = tensor(f, name)
    print(tag, t["dtype"], t["shape"], t["data_offsets"])
    E, R, C = t["shape"]
    rows = []
    for e in (0, 1, 2, 3):
        # 8 rows from each expert, spread over the matrix (contiguous row = C weights)
        for r in np.linspace(0, R - 1, 8).astype(int):
            a = base + t["data_offsets"][0] + ((e * R + r) * C) * 2
            raw = rng(f, a, a + C * 2 - 1)
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            rows.append(u.view(np.float32))
    W = np.stack(rows).astype(np.float32)
    print(tag, W.shape, float(W.std()), float(np.abs(W).max()))
    W.tofile(f"{out}/{tag}.f32")
