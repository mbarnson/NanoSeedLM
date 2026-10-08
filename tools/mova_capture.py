#!/usr/bin/env python3
"""tools/mova_capture.py - per-expert calibration statistics for the activation-weighted seed search (nslm-moe --act).

  mova_capture.py --text CALIB.txt [--out out/actsq_moe.bin] [--ctx 2048] [--max-windows N]

Runs MoVA BF16 (MLX reference implementation) over the calibration text in windows of BOS + (ctx - 1) tokens.  For every
sparse layer (3-47) it accumulates, over all positions but the BOS:
  - the MLP input x (post-attention norm): sum x^2 over all tokens, and per routed expert (top-8 of 100) the count,
    sum x^2 and sum w^2 x^2 (w = the expert's normalized, scaled routing weight);
  - the experts' down_proj input (silu(gate x) * up x) per routed expert: sum of squares;
  - the attention input for the value experts (top-4 of 64): count and sum x^2 per value expert, and over all tokens.
The routing is the model's own (route()), computed on the BF16 activations.

Output "NSLMMOE1": int32 first_layer, n_layers, E, D, F, EV, ctx; int64 tokens; then per layer:
  int64 count[E]; f64 sum_x2_all[D]; f64 sum_x2[E][D]; f64 sum_w2x2[E][D]; f64 sum_down[E][F];
  int64 vcount[EV]; f64 vsum_all[D]; f64 vsum[EV][D]
nslm/moe.c reads it.
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

E, EV, D, F = mc.N_EXPERTS, mc.N_VEXPERTS, mc.D, mc.F
L0, NL = 3, 45


class Acc:
    def __init__(self):
        self.count = np.zeros((NL, E), np.int64)
        self.x2_all = np.zeros((NL, D))
        self.x2 = np.zeros((NL, E, D))
        self.w2x2 = np.zeros((NL, E, D))
        self.down = np.zeros((NL, E, F))
        self.vcount = np.zeros((NL, EV), np.int64)
        self.v_all = np.zeros((NL, D))
        self.v = np.zeros((NL, EV, D))


ACC = Acc()
STATE = {"on": False}


def onehot(inds, n):
    return (inds[..., None] == mx.arange(n)).astype(mx.float32)          # [N, k, n]


def install(mod):
    moe_call = mod.SparseMoeBlock.__call__
    values = mod.Attention._values

    def moe(self, x):
        if STATE["on"]:
            li = self._cap_layer - L0
            inds, weights = mod.route(x, self.gate.weight, self.expert_bias, self.top_k, self.scaling_factor,
                                      partitions=self.router_partitions)
            xs = x[0, 1:].astype(mx.float32)                              # [N, D], BOS excluded
            ii, ww = inds[0, 1:], weights[0, 1:]
            oh = onehot(ii, E)                                            # [N, k, E]
            sel = oh.sum(1)                                               # [N, E]
            w2 = (oh * (ww[..., None] ** 2)).sum(1)                       # [N, E]
            x2 = xs * xs
            ex = self.experts
            xe = mx.expand_dims(x[:, 1:], (-2, -3))
            up = ex.up_proj(xe, inds[:, 1:])
            gate = ex.gate_proj(xe, inds[:, 1:])
            inter = (nn.silu(gate) * up)[0, :, :, 0, :].astype(mx.float32)   # [N, k, F]
            dn = mx.einsum("nke,nkf->ef", oh, inter * inter)
            r = [sel.sum(0), x2.sum(0), sel.T @ x2, w2.T @ x2, dn]
            mx.eval(r)
            ACC.count[li] += np.array(r[0]).round().astype(np.int64)
            ACC.x2_all[li] += np.array(r[1])
            ACC.x2[li] += np.array(r[2])
            ACC.w2x2[li] += np.array(r[3])
            ACC.down[li] += np.array(r[4])
        return moe_call(self, x)

    def vals(self, x):
        if STATE["on"] and self.mova:
            li = self._cap_layer - L0
            inds, _ = mod.route(x, self.v_router.weight, self.v_expert_bias, self.top_k, self.scaling_factor)
            xs = x[0, 1:].astype(mx.float32)
            sel = onehot(inds[0, 1:], EV).sum(1)
            x2 = xs * xs
            r = [sel.sum(0), x2.sum(0), sel.T @ x2]
            mx.eval(r)
            ACC.vcount[li] += np.array(r[0]).round().astype(np.int64)
            ACC.v_all[li] += np.array(r[1])
            ACC.v[li] += np.array(r[2])
        return values(self, x)

    mod.SparseMoeBlock.__call__ = moe
    mod.Attention._values = vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="plain-text calibration corpus")
    ap.add_argument("--out", default="out/actsq_moe.bin")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=0)
    a = ap.parse_args()
    mod = mc.register()
    install(mod)
    model, tok = mc.load()
    for i in range(L0, L0 + NL):
        model.layers[i].mlp._cap_layer = i
        model.layers[i].self_attn._cap_layer = i
    ids = mc.encode(tok, open(mc.ROOT / a.text, encoding="utf-8").read())
    step = a.ctx - 1
    nw = len(ids) // step
    if a.max_windows:
        nw = min(nw, a.max_windows)
    print(f"{a.text}: {len(ids)} tokens, {nw} windows of BOS + {step}", flush=True)
    STATE["on"] = True
    t0 = time.time()
    for w in range(nw):
        x = mx.array([[mc.BOS] + ids[w * step:(w + 1) * step]])
        mx.eval(model(x))
        if w % 10 == 0:
            print(f"  window {w + 1}/{nw}  {time.time() - t0:.0f}s", flush=True)
    tokens = nw * step
    out = mc.ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"NSLMMOE1")
        f.write(struct.pack("<7i", L0, NL, E, D, F, EV, a.ctx))
        f.write(struct.pack("<q", tokens))
        for li in range(NL):
            for arr, dt in ((ACC.count[li], np.int64), (ACC.x2_all[li], np.float64), (ACC.x2[li], np.float64),
                            (ACC.w2x2[li], np.float64), (ACC.down[li], np.float64), (ACC.vcount[li], np.int64),
                            (ACC.v_all[li], np.float64), (ACC.v[li], np.float64)):
                f.write(np.ascontiguousarray(arr, dtype=dt).tobytes())
    c = ACC.count
    print(f"wrote {out}: {tokens} tokens; routed tokens per expert: min {c.min()} median {int(np.median(c))} "
          f"max {c.max()}; experts with < 64 tokens: {(c < 64).sum()} of {c.size}; capture {time.time() - t0:.1f} s")
    print(f"value experts: min {ACC.vcount.min()} median {int(np.median(ACC.vcount))} max {ACC.vcount.max()}")
    print("state", mc.machine_state())


if __name__ == "__main__":
    main()
