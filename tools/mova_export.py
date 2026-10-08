#!/usr/bin/env python3
"""tools/mova_export.py - inputs for the C harness tools as raw little-endian files.

  mova_export.py windows --text TEXT --ref REF --out PRE
      PRE.ids (int32 [nw][2048]: BOS + 2047 text tokens, mova_score.py's windows), PRE.refnll (f32 [npos]),
      PRE.reftop (int32 [npos]) from REF.meta.npz; the f16 log-probs stay in REF.f16.npy.  -> nslm-mova-score
  mova_export.py routes --text TEXT --out PRE [--windows N]
      MLX BF16's routing on the same windows: PRE.routes_mlp (int32 [nw][2047][45][8], each sorted ascending),
      PRE.gap_mlp (f32 [nw][2047][45]: selection score of the k-th minus the (k+1)-th candidate), and the same for the
      value experts (top-4 of 64): PRE.routes_val, PRE.gap_val.  -> nslm-mova-score --routes
  mova_export.py prompts --prompts FILE.jsonl --out PRE
      chat prompts ({id, text, max_tokens, temperature, seed} per line; text is the rendered chat template without
      BOS) as ids: PRE.ids (int32, concatenated), PRE.index (id n_ids max_tokens temperature seed).
      -> nslm-mova-gen, nslm-mova-plcheck
  mova_export.py bench --text TEXT --out PRE
      PRE_ctx{1k,4k,12k}.ids: BOS + text filler + one chat question, exactly n tokens.  -> nslm-mova-bench
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mova_common as mc  # noqa: E402

CTX = 2048


def windows(tok, text, limit=0):
    ids = mc.encode(tok, open(mc.ROOT / text, encoding="utf-8").read())
    step = CTX - 1
    nw = len(ids) // step
    if limit:
        nw = min(nw, limit)
    return [[mc.BOS] + ids[w * step:(w + 1) * step] for w in range(nw)]


def cmd_windows(a):
    tok = mc.tokenizer()
    w = np.array(windows(tok, a.text), np.int32)
    out = mc.ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    w.tofile(str(out) + ".ids")
    meta = np.load(str(mc.ROOT / a.ref) + ".meta.npz")
    meta["nll"].astype(np.float32).tofile(str(out) + ".refnll")
    meta["top"].astype(np.int32).tofile(str(out) + ".reftop")
    print(f"{out}.ids: {w.shape[0]} windows; {meta['nll'].shape[0]} scored positions")


def cmd_routes(a):
    import mlx.core as mx
    mod = mc.register()
    model, tok = mc.load()
    wins = windows(tok, a.text, a.windows)
    NL = len(range(3, 48))
    rec = {"mlp": [], "val": []}
    orig_route = mod.route

    def cap(kind, sel_logits_fn):
        pass

    def route(x, weight, bias, top_k, scaling_factor, *, partitions=2):
        logits = mod.router_logits(x, weight, partitions)
        scores = mx.sigmoid(logits)
        sel = scores + bias.astype(mx.float32)
        srt = mx.sort(sel, axis=-1)
        gap = srt[..., -top_k] - srt[..., -top_k - 1]
        inds, w = orig_route(x, weight, bias, top_k, scaling_factor, partitions=partitions)
        rec["mlp" if weight.shape[0] == 100 else "val"].append((mx.sort(inds, axis=-1).astype(mx.int32), gap))
        return inds, w

    mod.route = route
    out = mc.ROOT / a.out
    files = {k: (open(str(out) + f".routes_{k}", "wb"), open(str(out) + f".gap_{k}", "wb")) for k in ("mlp", "val")}
    for wi, win in enumerate(wins):
        rec["mlp"].clear()
        rec["val"].clear()
        mx.eval(model(mx.array([win[:CTX - 1]])))   # rows 0..2046 (the C scorer forwards the same rows)
        for k in ("mlp", "val"):
            assert len(rec[k]) == NL, (k, len(rec[k]))
            inds = np.stack([np.array(i[0, :]) for i, _ in rec[k]], 1)     # [2047][45][k]
            gaps = np.stack([np.array(g[0, :]) for _, g in rec[k]], 1)     # [2047][45]
            files[k][0].write(inds.astype(np.int32).tobytes())
            files[k][1].write(gaps.astype(np.float32).tobytes())
        print(f"  window {wi + 1}/{len(wins)}", flush=True)
    for f in files.values():
        f[0].close()
        f[1].close()
    print(f"{out}.routes_*: {len(wins)} windows")


def cmd_prompts(a):
    tok = mc.tokenizer()
    out = mc.ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    allids, lines = [], []
    for l in open(mc.ROOT / a.prompts, encoding="utf-8"):
        p = json.loads(l)
        ids = mc.encode(tok, "<|ifm|begin_of_text|>" + p["text"])
        allids += ids
        lines.append(f"{p['id']} {len(ids)} {p['max_tokens']} {p['temperature']} {p['seed']}")
    np.array(allids, np.int32).tofile(str(out) + ".ids")
    (Path(str(out) + ".index")).write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} prompts")


BUCKETS = {"ctx1k": 1024, "ctx4k": 4096, "ctx12k": 12288}


def cmd_bench(a):
    """Matched-bench prompts: BOS + held-out filler + one chat question, exactly n tokens per bucket."""
    tok = mc.tokenizer()
    filler = mc.encode(tok, open(mc.ROOT / a.text, encoding="utf-8").read())
    q = mc.encode(tok, "<|ifm|im_start|>user\nWrite a short paragraph about the history of the bicycle.<|ifm|im_end|>"
                       "<|ifm|im_start|>assistant\n<ifm|think>\n")
    out = mc.ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, n in BUCKETS.items():
        ids = [mc.BOS] + filler[:n - 1 - len(q)] + q
        assert len(ids) == n
        np.array(ids, np.int32).tofile(str(out) + f"_{name}.ids")
        print(f"{out}_{name}.ids: {n} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["windows", "routes", "prompts", "bench"])
    ap.add_argument("--text")
    ap.add_argument("--ref")
    ap.add_argument("--prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=0)
    a = ap.parse_args()
    {"windows": cmd_windows, "routes": cmd_routes, "prompts": cmd_prompts, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
