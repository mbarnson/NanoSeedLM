#!/usr/bin/env python3
"""X^T X of MoVA's dense projection inputs (GPTQ / AW seed search), from the BF16 model on a calibration text.

  mova_capture_dense.py --text CALIB.txt --out DIR [--ctx 2048] [--holdout 2] [--dump-layers 20]

DIR/L<l>_<site>.bin: "NSLMXTX1", int32 dim, int64 rows, dim x dim f32 sum (BOS rows excluded). Sites: 0 attention
input (q / k / v / gate), 1 o_proj input, 2 MLP input (shared or dense gate / up), 3 their down_proj input;
L<n_layers>_0 the LM head input. The last --holdout windows are left out; for --dump-layers and the LM head their
rows go to DIR/held_L<l>_<site>.f32 ("NSLMXR01", int32 dim, int64 rows, rows x dim f32). DIR/tokens.bin: "NSLMTOK1",
int32 vocab, int64 count[vocab] over all windows.
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mova_common as mc  # noqa: E402

STATE = {"mode": None}   # None, "acc" or "held"
H, ROWS, HELD = {}, {}, {}


class Cap(nn.Linear):
    def __call__(self, x):
        key, mode = self._cap, STATE["mode"]
        if mode:
            xs = x[0, 1:].astype(mx.float32)
            if mode == "acc":
                h = xs.T @ xs
                H[key] = h if key not in H else H[key] + h
                ROWS[key] = ROWS.get(key, 0) + xs.shape[0]
            elif key in HELD:
                HELD[key].append(xs)
        return super().__call__(x)


def tag(m, key):
    m.__class__ = Cap
    m._cap = key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--holdout", type=int, default=2)
    ap.add_argument("--dump-layers", default="20")
    a = ap.parse_args()
    model, tok = mc.load()
    n = len(model.layers)
    for l, layer in enumerate(model.layers):
        at, mlp = layer.self_attn, layer.mlp
        tag(at.q_proj, (l, 0))
        tag(at.o_proj, (l, 1))
        dense = getattr(mlp, "shared_experts", mlp)
        tag(dense.gate_proj, (l, 2))
        tag(dense.down_proj, (l, 3))
    tag(model.lm_head, (n, 0))
    for l in (int(x) for x in a.dump_layers.split(",") if x):
        for s in range(4):
            HELD[(l, s)] = []
    HELD[(n, 0)] = []

    ids = mc.encode(tok, open(a.text).read())
    step = a.ctx - 1
    nw = len(ids) // step
    print(f"{a.text}: {len(ids)} tokens, {nw} windows of BOS + {step}, last {a.holdout} held out", flush=True)
    t0 = time.time()
    for w in range(nw):
        STATE["mode"] = "acc" if w < nw - a.holdout else "held"
        mx.eval(model(mx.array([[mc.BOS] + ids[w * step:(w + 1) * step]])), list(H.values()), [v for v in HELD.values() if v])
        if w % 10 == 0:
            print(f"  window {w + 1}/{nw}  {time.time() - t0:.0f}s", flush=True)
    STATE["mode"] = None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for (l, s), h in H.items():
        h = np.array(h, np.float32)
        with open(out / f"L{l}_{s}.bin", "wb") as f:
            f.write(b"NSLMXTX1" + struct.pack("<iq", h.shape[0], ROWS[(l, s)]) + h.tobytes())
    for (l, s), xs in HELD.items():
        x = np.array(mx.concatenate(xs), np.float32)
        with open(out / f"held_L{l}_{s}.f32", "wb") as f:
            f.write(b"NSLMXR01" + struct.pack("<iq", x.shape[1], x.shape[0]) + x.tobytes())
    counts = np.bincount(np.array(ids[:nw * step]), minlength=model.lm_head.weight.shape[0]).astype(np.int64)
    with open(out / "tokens.bin", "wb") as f:
        f.write(b"NSLMTOK1" + struct.pack("<i", counts.size) + counts.tobytes())
    print(f"wrote {len(H)} X^T X, {len(HELD)} held-out sets, token counts to {out}; {time.time() - t0:.0f} s")
    print("state", mc.machine_state())


if __name__ == "__main__":
    main()
