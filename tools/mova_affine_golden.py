#!/usr/bin/env python3
"""tools/mova_affine_golden.py - golden affine-quantization samples from MLX for tests/test_affine.c.

  mova_affine_golden.py [--out out/test/affine]

For a few real MoVA tensors (BF16) and a synthetic one with edge cases, writes mx.quantize(w, group_size=64, bits=b)
for b in {8, 4}: NAME.bf16 (rows x cols u16), NAME.q{b}.w (the packed uint32 words, as MLX stores them),
NAME.q{b}.s / .b (the BF16 scales and biases), NAME.q{b}.deq (mx.dequantize as BF16), and an index.txt with
"name rows cols".
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mova_common as mc  # noqa: E402

TENSORS = [
    ("model.layers.3.mlp.experts.0.gate_proj.weight", None),
    ("model.layers.20.mlp.experts.57.down_proj.weight", None),
    ("model.layers.10.self_attn.v_experts.3.weight", None),
    ("model.layers.0.self_attn.q_proj.weight", None),
    ("model.embed_tokens.weight", (0, 512)),          # rows 0..511 (incl. unused tokens: near-zero rows)
    ("lm_head.weight", (100000, 100256)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/test/affine")
    a = ap.parse_args()
    out = mc.ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    import json
    idx = json.load(open(mc.MOVA / "model.safetensors.index.json"))["weight_map"]
    items = []
    for name, rows in TENSORS:
        w = mx.load(str(mc.MOVA / idx[name]))[name]
        if rows:
            w = w[rows[0]:rows[1]]
        items.append((name.replace("model.", "").replace(".weight", ""), w))
    rng = np.random.default_rng(7)
    syn = rng.standard_normal((64, 256)).astype(np.float32) * 0.02
    syn[0, :64] = 0.0                      # an all-zero group
    syn[1, :64] = 0.5                      # a constant group
    syn[2, :64] = -np.abs(syn[2, :64])     # all-negative group
    syn[3, :64] = np.abs(syn[3, :64])      # all-positive group
    syn[4, 3] = 7.0                        # a large outlier
    syn[5, :64] = np.linspace(-1e-6, 1e-6, 64)   # tiny range
    items.append(("synthetic", mx.array(syn).astype(mx.bfloat16)))
    lines = []
    for name, w in items:
        w = w.astype(mx.bfloat16)
        mx.eval(w)
        rows, cols = w.shape
        np.array(w.view(mx.uint16)).tofile(out / f"{name}.bf16")
        for bits in (8, 4):
            q, s, b = mx.quantize(w, group_size=64, bits=bits)
            d = mx.dequantize(q, s, b, group_size=64, bits=bits).astype(mx.bfloat16)
            mx.eval(q, s, b, d)
            assert s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16 and q.dtype == mx.uint32, (s.dtype, q.dtype)
            np.array(q).tofile(out / f"{name}.q{bits}.w")
            np.array(s.view(mx.uint16)).tofile(out / f"{name}.q{bits}.s")
            np.array(b.view(mx.uint16)).tofile(out / f"{name}.q{bits}.b")
            np.array(d.view(mx.uint16)).tofile(out / f"{name}.q{bits}.deq")
        lines.append(f"{name} {rows} {cols}")
    (out / "index.txt").write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} tensors")


if __name__ == "__main__":
    main()
