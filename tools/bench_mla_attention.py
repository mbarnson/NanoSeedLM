#!/usr/bin/env python3
"""One MLA prompt-attention call (one layer, real head shapes, random BF16 inputs), timed with fixed boundaries.

  PYTHONPATH=/path/to/omlx python tools/bench_mla_attention.py [--L 2048] [--N 32768] [--variants ...] [--reps 10]

Each variant runs in a fresh process (no kernel or allocator state carried between variants): the parent starts one
child per variant and mode. A child builds the inputs, warms up (--warmup calls), then times --reps calls, each ended by
mx.eval (wall clock around the call; GPU work only, the inputs are evaluated before timing). Modes:
  attn         attention only: the inputs are the already expanded per-head keys [k_a ; k_rope] and values v_a
  expand+attn  the latent c [N, r] in: the expansion mat-muls (k_a = q_lat_a^T c, v_a = v_up_a c) plus attention
Variants:
  fused-BQ     tools/nanoseedlm_k2.py's _MLA_PREFILL with BQ query rows per threadgroup (e.g. fused-64)
  mlx-sdpa     mx.fast.scaled_dot_product_attention on [q_a ; q_rope] against [k_a ; k_rope] (Dqk 256, Dv 128)
               with a boolean causal mask (MLX 0.32.3: not fused at these dims; scores in memory)
  unfused      the loader's Python block loop (FUSED_PREFILL = False)
Shapes: 32 heads, head dim 128, RoPE dim 128, rank 768, L queries at the end of N keys (causal, bottom right), one
key block (all N keys). Output: one JSON line per variant and mode: median, min, max, quartiles (ms), machine state.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def machine_state():
    st = {}
    try:
        st["power"] = "AC" if "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout else "battery"
        st["load1"], st["load5"] = (round(x, 2) for x in os.getloadavg()[:2])
        top = subprocess.run(["ps", "-Ao", "pcpu,comm", "-r"], capture_output=True, text=True).stdout.splitlines()[1:4]
        st["top_cpu"] = [" ".join(t.split()) for t in top]
    except Exception as e:  # noqa: BLE001
        st["error"] = str(e)
    return st


def child(a):
    from omlx.patches.k2_horizon import apply_k2_horizon_patch

    apply_k2_horizon_patch()
    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(TOOLS))
    import nanoseedlm_k2 as ns

    H, hd, dr, r, L, N = 32, 128, 128, a.rank, a.L, a.N
    rng = np.random.default_rng(a.seed)
    g = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)
    q_lat, v_up = g(H, r, hd) * 0.05, g(H, hd, r) * 0.05
    q, qr, lat, kr = g(1, H, L, hd), g(1, H, L, dr), g(1, 1, N, r), g(1, 1, N, dr)
    scale = hd ** -0.5
    kn, vv = lat @ q_lat, lat @ v_up.swapaxes(1, 2)
    mx.eval(q, qr, lat, kr, kn, vv)
    ns.FUSED_KEYS = N                                          # one key block
    var = a.variant
    if var.startswith("fused-"):
        ns.FUSED_BQ = int(var.split("-")[1])
        if a.mode == "expand+attn":
            f = lambda: ns.mla_attention(q, qr, lat, kr, q_lat, v_up, scale, mask="causal", path="expanded")
        else:
            ns.mla_attention(q[:, :, :64], qr[:, :, :64], lat[:, :, :64], kr[:, :, :64], q_lat, v_up, scale, mask="causal",
                             path="expanded")                  # builds the kernel
            kern, bq, nt = ns._kernels["mla_prefill"], ns.FUSED_BQ, ns.FUSED_BQ * 4
            qa, qs = q * scale, qr * scale
            o0, ml0 = mx.zeros((1, H, L, hd), mx.float32), mx.zeros((1, H, L, 2), mx.float32)
            prm = mx.array([0, N, 1, 1], mx.int32)
            mx.eval(qa, qs, o0, ml0, prm)

            def f():
                o, ml = kern(inputs=[qa, qs, kn, kr, vv, o0, ml0, prm], template=[("T", mx.bfloat16), ("HD", hd), ("DR", dr), ("BQ", bq)],
                             grid=(-(-L // bq) * nt, H, 1), threadgroup=(nt, 1, 1), output_shapes=[o0.shape, ml0.shape],
                             output_dtypes=[mx.float32, mx.float32])
                return (o / ml[..., 1:]).astype(mx.bfloat16)
    elif var == "mlx-sdpa":
        mask = mx.arange(N)[None] <= mx.arange(N - L, N)[:, None]
        qcat = mx.concatenate([q, qr], -1)
        mx.eval(mask, qcat)
        if a.mode == "expand+attn":
            f = lambda: mx.fast.scaled_dot_product_attention(
                qcat, mx.concatenate([lat @ q_lat, mx.broadcast_to(kr, (1, H, N, dr))], -1), lat @ v_up.swapaxes(1, 2),
                scale=scale, mask=mask)
        else:
            kcat = mx.concatenate([kn, mx.broadcast_to(kr, (1, H, N, dr))], -1)
            mx.eval(kcat)
            f = lambda: mx.fast.scaled_dot_product_attention(qcat, kcat, vv, scale=scale, mask=mask)
    elif var == "unfused":
        if a.mode != "expand+attn":
            print(json.dumps({"variant": var, "mode": a.mode, "skipped": "the block loop expands inside"}))
            return
        ns.FUSED_PREFILL = False
        f = lambda: ns.mla_attention(q, qr, lat, kr, q_lat, v_up, scale, mask="causal", path="expanded")
    else:
        raise SystemExit(f"unknown variant {var}")
    for _ in range(a.warmup):
        mx.eval(f())
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        mx.eval(f())
        ts.append((time.perf_counter() - t0) * 1e3)
    qs4 = statistics.quantiles(ts, n=4)
    print(json.dumps({"variant": var, "mode": a.mode, "L": L, "N": N, "heads": H, "rank": r, "reps": a.reps,
                      "warmup": a.warmup, "median_ms": round(statistics.median(ts), 2), "min_ms": round(min(ts), 2),
                      "max_ms": round(max(ts), 2), "q1_ms": round(qs4[0], 2), "q3_ms": round(qs4[2], 2),
                      "mlx": mx.__version__}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--N", type=int, default=32768)
    ap.add_argument("--rank", type=int, default=768)
    ap.add_argument("--variants", default="fused-32,fused-64,fused-128,mlx-sdpa,unfused")
    ap.add_argument("--modes", default="attn,expand+attn")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--variant"); ap.add_argument("--mode")
    a = ap.parse_args()
    if a.variant:
        return child(a)
    for var in a.variants.split(","):
        for mode in a.modes.split(","):
            start = machine_state()
            cmd = [sys.executable, __file__, "--variant", var, "--mode", mode, "--L", str(a.L), "--N", str(a.N),
                   "--rank", str(a.rank), "--reps", str(a.reps), "--warmup", str(a.warmup), "--seed", str(a.seed)]
            out = subprocess.run(cmd, capture_output=True, text=True)
            line = next((x for x in out.stdout.splitlines() if x.startswith("{")), None)
            rec = json.loads(line) if line else {"variant": var, "mode": mode, "error": out.stderr[-400:]}
            rec["state_start"], rec["state_end"] = start, machine_state()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
