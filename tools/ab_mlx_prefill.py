#!/usr/bin/env python3
"""Paired prefill A/B: both folders loaded in one process, runs alternated; median of the per-round B/A ratios.

  PYTHONPATH=/path/to/omlx python tools/ab_mlx_prefill.py FOLDER_A FOLDER_B --text FILE [--rounds 5] [--ane-b]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_mlx_prefill import load, machine_state, prefill_fn, prefill_once  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--text", required=True, help="prompt text (held-out prose)")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ane-b", action="store_true", help="oMLX K2 ANE MLP prefill on B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"a": a.a, "b": a.b, "ane_b": a.ane_b, "state_start": machine_state(), "lengths": {}}
    (ma, tok), (mb, _) = load(a.a), load(a.b)
    if a.ane_b:
        from omlx.patches.k2_horizon.ane_prefill import enable_ane_prefill
        enable_ane_prefill(mb)
    fa, fb = prefill_fn(ma), prefill_fn(mb)
    filler = tok.encode(Path(a.text).read_text(), add_special_tokens=False)
    warm = mx.array([0] + filler[:511])
    prefill_once(ma, fa, warm), prefill_once(mb, fb, warm)
    for n in (int(x) for x in a.lengths.split(",")):
        ids = mx.array([0] + filler[:n - 1])
        ra, rb = [], []
        for _ in range(a.rounds):
            ra.append(prefill_once(ma, fa, ids))
            rb.append(prefill_once(mb, fb, ids))
        ta, tb = [r[0] for r in ra], [r[0] for r in rb]
        ratio = statistics.median(y / x for x, y in zip(ta, tb))
        res["lengths"][n] = {"a_tok_s": ta, "b_tok_s": tb, "ratio_b_over_a": ratio,
                             "a_peak_gb": max(r[1] for r in ra) / 1e9, "b_peak_gb": max(r[1] for r in rb) / 1e9}
        print(f"prefill {n:5d}: A {statistics.median(ta):6.1f} ({min(ta):.0f}-{max(ta):.0f})  B {statistics.median(tb):6.1f} "
              f"({min(tb):.0f}-{max(tb):.0f}) tok/s  B/A {ratio:.3f} (rounds {min(y / x for x, y in zip(ta, tb)):.3f}-"
              f"{max(y / x for x, y in zip(ta, tb)):.3f})", flush=True)
    res["state_end"] = machine_state()
    print(f"state {res['state_start']} -> {res['state_end']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
