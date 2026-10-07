#!/usr/bin/env python3
"""tools/check_mlx_loader.py - a model folder's MLX outputs against a saved reference (a loader change must not change
the model): logits at 1 and 40 tokens (relative error <= 1e-3) and greedy output for 3 prompts x 300 tokens
(identical; no EOS stop).

  PYTHONPATH=~/devel/omlx-nslm python tools/check_mlx_loader.py FOLDER --save ref.npz      (with the current loader)
  PYTHONPATH=~/devel/omlx-nslm python tools/check_mlx_loader.py FOLDER --compare ref.npz   (with the candidate)
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_mlx_prefill import FILLER, load  # noqa: E402

PROMPTS = ["Write a short essay about the history of the bicycle.",
           "Explain how a hash table handles collisions, with a small Python example.",
           "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Think step by step."]
GEN, MAX_REL = 300, 1e-3


def outputs(model, tok):
    from mlx_lm import stream_generate
    filler = tok.encode(FILLER.read_text(), add_special_tokens=False)
    out = {}
    for n in (1, 40):
        out[f"logits{n}"] = np.array(model(mx.array([[0] + filler[:n - 1]])).astype(mx.float32))
    tok.eos_token_ids = []
    for i, p in enumerate(PROMPTS):
        prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        out[f"greedy{i}"] = np.array([r.token for r in stream_generate(model, tok, prompt, max_tokens=GEN)], np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save")
    g.add_argument("--compare")
    a = ap.parse_args()
    got = outputs(*load(a.folder))
    if a.save:
        np.savez(a.save, **got)
        print(f"saved {a.save}")
        return
    ref, ok = np.load(a.compare), True
    for k in sorted(ref.files):
        if k.startswith("logits"):
            r, g_ = ref[k], got[k]
            err = float(np.sqrt(np.mean((r - g_) ** 2) / np.mean(r * r)))
            good = err <= MAX_REL
            print(f"{k}: relative error {err:.2e} {'ok' if good else 'FAIL'}")
        else:
            diff = np.flatnonzero(ref[k] != got[k])
            good = ref[k].shape == got[k].shape and diff.size == 0
            print(f"{k}: {'identical' if good else f'FAIL, first difference at token {diff[0] if diff.size else -1}'}")
        ok &= good
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
