#!/usr/bin/env python3
"""tools/qwen4_tier_sim.py - a high tier that follows the routing: H = the N experts with the largest decayed use
count, recomputed every R tokens. Prints the share of router choices that H serves, and the insertions into H for each
token (each one a high-tier expert read from NVMe). Routing traces are Q4E_ROUTES files.

  python3 tools/qwen4_tier_sim.py TRACE.bin [TRACE.bin ...]
"""
import sys

import numpy as np


def tokens(path, L=48, K=10):
    """Each token's routed experts [L][K] of a trace (each forward: int32 T, then [L][T][K])."""
    a, i = np.fromfile(path, dtype=np.int32), 0
    while i < len(a):
        t = int(a[i]); r = a[i+1:i+1+L*t*K].reshape(L, t, K); i += 1 + L*t*K
        for j in range(t): yield r[:, j, :]
def run(path, N, half, R=16, warm=2000):
    """Share of router choices in H and insertions a token, after warm tokens; H = top N every R tokens."""
    g = 0.5 ** (1.0 / half)
    c = np.zeros(48 * 512); inH = np.zeros(48 * 512, bool)
    hit = tot = ins = n = 0
    off = (np.arange(48) * 512)[:, None]
    for tk in tokens(path):
        n += 1
        idx = (tk + off).ravel()
        if n > warm: hit += inH[idx].sum(); tot += idx.size
        c *= g; np.add.at(c, idx, 1.0)
        if n % R == 0:
            new = np.zeros_like(inH); new[np.argpartition(-c, N)[:N]] = True
            if n > warm: ins += (new & ~inH).sum()
            inH = new
    return hit / tot, ins / max(1, n - warm)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--gb", type=float, nargs="+", default=[10.4, 20.0], help="high-tier budgets (GB)")
    ap.add_argument("--mb", type=float, default=2.76, help="one high-tier expert (MB): P4 2.76, P3 2.46")
    ap.add_argument("--half", type=int, nargs="+", default=[500, 2000], help="half-lives of the use count (tokens)")
    a = ap.parse_args()
    for gb in a.gb:
        n = int(gb * 1000 / a.mb)
        for half in a.half:
            for t in a.trace:
                share, ins = run(t, n, half)
                print("H %.1f GB (%d experts), half-life %d: %s share %.1f%%, insertions %.2f a token (%.1f MB)"
                      % (gb, n, half, t.split("/")[-1], 100 * share, ins, ins * a.mb))


if __name__ == "__main__":
    main()
