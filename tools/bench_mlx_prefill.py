#!/usr/bin/env python3
"""MLX prefill / decode speed of a model folder loaded through oMLX (Trust Remote Code), with machine state.

  PYTHONPATH=/path/to/omlx python tools/bench_mlx_prefill.py FOLDER --text FILE [--lengths 512,2048,8192] [--ane]

Prefill: BOS + FILE's tokens in 2048-token chunks; median tok/s and peak memory over the weights. Decode: greedy.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 2048


def machine_state():
    st = {}
    try:
        st["power"] = "AC" if "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout else "battery"
        st["load1"], st["load5"] = (round(x, 2) for x in os.getloadavg()[:2])
        th = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
        st["thermal_warning"] = "No thermal warning" not in th
        ps = subprocess.run(["ps", "-A", "-o", "pid=,%cpu=,rss=,comm="], capture_output=True, text=True).stdout.splitlines()
        procs = [(float(c), int(r), Path(n).name) for p, c, r, n in (line.split(None, 3) for line in ps) if int(p) != os.getpid()]
        st["other_cpu_pct"] = round(sum(c for c, _, _ in procs), 1)
        st["top_other"] = [f"{n} {c:.0f}%" for c, _, n in sorted(procs, reverse=True)[:3]]
        st["top_rss_gb"] = [f"{n} {r / 1e6:.1f}" for _, r, n in sorted(procs, key=lambda q: -q[1])[:3]]
        st["mem_free_pct"] = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout)
    except Exception as e:  # noqa: BLE001
        st["error"] = str(e)
    return st


def load(folder):
    try:
        from omlx.utils.model_loading import load_text_model
    except ImportError as error:
        raise SystemExit("put an oMLX checkout on PYTHONPATH") from error
    model, tok = load_text_model(str(folder), model_settings=SimpleNamespace(trust_remote_code=True))[:2]
    mx.eval(model.parameters())
    return model, tok


def prefill_fn(model):
    if hasattr(model, "_omlx_prefill"):
        return lambda ids, cache: model._omlx_prefill(ids[None], cache=cache)

    def prefill(ids, cache):
        for s in range(0, ids.size, CHUNK):
            model(ids[s:s + CHUNK][None], cache=cache)
            mx.eval([c.state for c in cache])
    return prefill


def prefill_once(model, prefill, ids):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    prefill(ids, cache)
    dt = time.perf_counter() - t0
    peak = mx.get_peak_memory() - base
    del cache
    mx.clear_cache()
    return ids.size / dt, peak


def decode_once(model, tok, prompt, gen):
    from mlx_lm import stream_generate
    last = None
    for last in stream_generate(model, tok, prompt, max_tokens=gen):
        pass
    return last.generation_tps, last.generation_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--text", required=True, help="prompt text (held-out prose)")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gen", type=int, default=256, help="decode tokens (0: no decode run)")
    ap.add_argument("--ane", action="store_true", help="oMLX K2 ANE MLP prefill")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {"folder": str(a.folder), "ane": a.ane, "state_start": machine_state()}
    model, tok = load(a.folder)
    res["weights_gb"] = mx.get_active_memory() / 1e9
    if a.ane:
        from omlx.patches.k2_horizon.ane_prefill import enable_ane_prefill
        enable_ane_prefill(model)
        res["ane_programs"] = model._omlx_k2_ane_prefill_count
        res["weights_gb_ane"] = mx.get_active_memory() / 1e9
    prefill = prefill_fn(model)
    filler = tok.encode(Path(a.text).read_text(), add_special_tokens=False)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 0
    prefill_once(model, prefill, mx.array([bos] + filler[:511]))   # warm-up (kernel compilation)

    res["prefill"] = {}
    for n in (int(x) for x in a.lengths.split(",")):
        ids = mx.array([bos] + filler[:n - 1])
        runs = [prefill_once(model, prefill, ids) for _ in range(a.repeats)]
        tps, peak = [r[0] for r in runs], [r[1] / 1e9 for r in runs]
        res["prefill"][n] = {"tok_s": statistics.median(tps), "tok_s_all": tps, "peak_over_weights_gb": max(peak)}
        print(f"prefill {n:5d}: {statistics.median(tps):7.1f} tok/s ({min(tps):.1f}-{max(tps):.1f})  "
              f"peak over weights {max(peak):.2f} GB", flush=True)

    if a.gen:
        tok.eos_token_ids = []
        prompt = tok.apply_chat_template([{"role": "user", "content": "Write a short essay about the history of the bicycle."}],
                                         add_generation_prompt=True)
        decode_once(model, tok, prompt, 16)   # warm-up
        runs = [decode_once(model, tok, prompt, a.gen) for _ in range(a.repeats)]
        tps = [r[0] for r in runs]
        res["decode"] = {"tok_s": statistics.median(tps), "tok_s_all": tps, "tokens": runs[0][1]}
        print(f"decode {runs[0][1]} tokens: {statistics.median(tps):.1f} tok/s ({min(tps):.1f}-{max(tps):.1f})", flush=True)

    res["state_end"] = machine_state()
    print(f"weights {res['weights_gb']:.2f} GB; state {res['state_start']} -> {res['state_end']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
