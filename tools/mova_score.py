#!/usr/bin/env python3
"""tools/mova_score.py - KLD(MoVA BF16 || candidate) and perplexity on a text, in MLX.

  mova_score.py --write-ref --text TEXT --ref REF                     (the BF16 reference)
  mova_score.py --cand q8-gu --text TEXT --ref REF [--name NAME] [--out FILE.kv] [--windows N] [--chunk 512]

Windows of BOS + 2047 text tokens (no overlap); in each, the logits at positions 1024..2046 predict the next token
(1023 scored positions per window).  The reference pass stores every scored position's full log-softmax as f16
(REF.f16.npy, [positions][250624]) plus the reference NLL and arg max (REF.meta.npz); a candidate pass computes, per
position, the exact full-vocabulary KL(P_ref || P_cand) = sum_v p_ref (log p_ref - log p_cand) with p_ref
renormalised from the stored log-probs.  Writes key=value lines (default NAME_TEXTSTEM.kv): kld_mean, kld_se,
ppl_ratio, same_top1_pct, nll means, positions, machine state.
Candidates: mova_common.apply_candidate specs (bf16, q8-gu, q4-gu, nslm-gu:DIR, ...).
"""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mova_common as mc  # noqa: E402

CTX, FIRST = 2048, 1024
V = 250624


def windows(tok, text, limit):
    ids = mc.encode(tok, open(mc.ROOT / text).read())
    step = CTX - 1
    nw = len(ids) // step
    if limit:
        nw = min(nw, limit)
    return ids, [[mc.BOS] + ids[w * step:(w + 1) * step] for w in range(nw)]


def logprobs(model, win, chunk=0):
    if chunk:   # through the KV cache in chunks of `chunk` tokens (diagnostics)
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
        outs = []
        for s in range(0, CTX - 1, chunk):
            o = model(mx.array([win[s:min(s + chunk, CTX - 1)]]), cache=cache)[0]
            mx.eval(o)
            outs.append(o)
        lg = mx.concatenate(outs, 0)[FIRST:CTX - 1].astype(mx.float32)
    else:
        x = mx.array([win])
        lg = model(x)[0, FIRST:CTX - 1].astype(mx.float32)      # predicts tokens FIRST+1 .. CTX-1
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True, help="reference prefix (REF.f16, REF.meta.npz)")
    ap.add_argument("--write-ref", action="store_true")
    ap.add_argument("--cand", default="bf16")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--windows", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0, help="candidate pass through the KV cache in chunks (diagnostics)")
    a = ap.parse_args()
    st0 = mc.machine_state()
    model, tok = mc.load()
    ids, wins = windows(tok, a.text, a.windows)
    npos = len(wins) * (CTX - 1 - FIRST)
    ref = Path(a.ref)
    t0 = time.time()
    if a.write_ref:
        ref.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(str(ref) + ".f16.npy", mode="w+", dtype=np.float16, shape=(npos, V))
        nll = np.zeros(npos, np.float32)
        top = np.zeros(npos, np.int32)
        for w, win in enumerate(wins):
            lp = logprobs(model, win)
            tgt = mx.array(win[FIRST + 1:CTX])
            n = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
            am = mx.argmax(lp, axis=-1)
            lp16 = lp.astype(mx.float16)
            mx.eval(lp16, n, am)
            s = w * (CTX - 1 - FIRST)
            mm[s:s + lp16.shape[0]] = np.array(lp16)
            nll[s:s + lp16.shape[0]] = np.array(n)
            top[s:s + lp16.shape[0]] = np.array(am)
            if w % 8 == 0:
                print(f"  ref window {w + 1}/{len(wins)}  {time.time() - t0:.0f}s", flush=True)
        mm.flush()
        np.savez(str(ref) + ".meta.npz", nll=nll, top=top, tokens=len(ids), windows=len(wins))
        print(f"wrote {ref}.f16.npy: {len(wins)} windows, {npos} positions, {len(ids)} tokens; ref ppl {np.exp(nll.mean()):.4f}")
        return
    meta = np.load(str(ref) + ".meta.npz")
    mm = np.load(str(ref) + ".f16.npy", mmap_mode="r")
    assert mm.shape[0] >= npos, (mm.shape, npos)
    mc.apply_candidate(model, a.cand)
    kl = np.zeros(npos)
    nllc = np.zeros(npos)
    same = np.zeros(npos, bool)
    for w, win in enumerate(wins):
        lp = logprobs(model, win, a.chunk)
        s = w * (CTX - 1 - FIRST)
        r = mx.array(np.asarray(mm[s:s + lp.shape[0]])).astype(mx.float32)
        r = r - mx.logsumexp(r, axis=-1, keepdims=True)
        k = (mx.exp(r) * (r - lp)).sum(-1)
        tgt = mx.array(win[FIRST + 1:CTX])
        n = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        am = mx.argmax(lp, axis=-1)
        mx.eval(k, n, am)
        kl[s:s + lp.shape[0]] = np.array(k)
        nllc[s:s + lp.shape[0]] = np.array(n)
        same[s:s + lp.shape[0]] = np.array(am) == meta["top"][s:s + lp.shape[0]]
        if w % 8 == 0:
            print(f"  window {w + 1}/{len(wins)}  {time.time() - t0:.0f}s  kld so far {kl[:s + lp.shape[0]].mean():.5f}", flush=True)
    nllr = meta["nll"][:npos]
    res = {
        "name": a.name or a.cand, "cand": a.cand, "text": a.text, "windows": len(wins), "positions": npos,
        "kld_mean": f"{kl.mean():.7f}", "kld_se": f"{kl.std(ddof=1) / np.sqrt(npos):.7f}",
        "kld_p99": f"{np.percentile(kl, 99):.5f}", "kld_max": f"{kl.max():.4f}",
        "nll_ref": f"{nllr.mean():.6f}", "nll_cand": f"{nllc.mean():.6f}",
        "ppl_ratio": f"{np.exp(nllc.mean() - nllr.mean()):.6f}", "same_top1_pct": f"{100 * same.mean():.3f}",
        "seconds": f"{time.time() - t0:.1f}", "chunk": a.chunk,
    }
    res.update({f"start_{k}": v for k, v in st0.items()})
    res.update({f"end_{k}": v for k, v in mc.machine_state().items()})
    txt = "".join(f"{k}={v}\n" for k, v in res.items())
    print(txt, end="")
    out = Path(a.out or f"{res['name']}_{Path(a.text).stem}.kv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    np.save(str(out).replace(".kv", ".kl.npy"), kl.astype(np.float32))


if __name__ == "__main__":
    main()
