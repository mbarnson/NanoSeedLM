#!/usr/bin/env python3
"""tools/qwen4_tier_map.py - a per-expert tier map for nslm-qwen4-emulate --map: uint8 [layer][gate, up, down][expert],
0 for the first pack (the lower tier), 1 for the second.  The modes put the same share of the weights in the higher tier:

  freq    the most used experts of each layer, from routing traces (Q4E_ROUTES files), all three projections
  random  random experts of each layer: the control for freq
  down    the down_proj of every expert (a share of 1/3), as Unsloth gives down_proj more bits

  python3 tools/qwen4_tier_map.py --mode freq --share 0.3333 --trace A.bin [--trace B.bin ...] --out MAP
"""
import argparse

import numpy as np


def counts(paths, layers, experts, k):
    """Router choices of each expert: [layer][expert].  A trace holds each forward's int32 T, then [layers][T][k]."""
    c = np.zeros((layers, experts), dtype=np.int64)
    for p in paths:
        a, i = np.fromfile(p, dtype=np.int32), 0
        while i < len(a):
            t = int(a[i])
            r = a[i + 1:i + 1 + layers * t * k].reshape(layers, t * k)
            i += 1 + layers * t * k
            for l in range(layers):
                c[l] += np.bincount(r[l], minlength=experts)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["freq", "random", "down"], required=True)
    ap.add_argument("--share", type=float, default=1 / 3, help="share of the experts in the higher tier (freq, random)")
    ap.add_argument("--trace", action="append", default=[])
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L, E = a.layers, a.experts
    m = np.zeros((L, 3, E), dtype=np.uint8)
    n = int(round(a.share * E))
    c = counts(a.trace, L, E, a.top_k) if a.trace else None
    rng = np.random.default_rng(a.seed)
    for l in range(L):
        if a.mode == "down":
            m[l, 2, :] = 1
            continue
        hot = np.argsort(-c[l], kind="stable")[:n] if a.mode == "freq" else rng.choice(E, n, replace=False)
        m[l, :, hot] = 1
    m.tofile(a.out)
    msg = "%s: %d layers, %.4f of the weights in the higher tier" % (a.out, L, m.mean())
    if c is not None and a.mode != "down":   # the share of router choices that the higher tier serves, on these traces
        msg += ", %.1f%% of router choices" % (100 * sum(c[l][m[l, 0] == 1].sum() for l in range(L)) / c.sum())
    print(msg)


if __name__ == "__main__":
    main()
