# harness.py - probe: KL of SmolLM2-135M (BF16 weights, FP32 arithmetic) with the MLP of some layers quantized, against
# the unquantized model, on wikitext-2 test (16 windows of 512 tokens).  Calibration X^T X from wikitext-2 train (64
# windows).  Results: docs/results/2026-11-06-search-objective-kl.md.
#
#   python3 harness.py SMOL_DIR LAYERS VARIANT [VARIANT ...]
#   SMOL_DIR: config.json, tokenizer.json, model.safetensors of HuggingFaceTB/SmolLM2-135M, and wt2-test.parquet,
#             wt2-train.parquet (Salesforce/wikitext, wikitext-2-raw-v1).  libseed.so (seedlib.c) and libtq.so
#             (tqlib.c + nslm/lib_trellis.c) beside this file.
#   LAYERS:   comma list, e.g. 15 or 0-29.
#   VARIANT:  q8_0 | q4_0
#             METHOD:P:K[:STRIDE]  seeds; METHOD plain (no weights), diag (weights h), diag25 (weights h^1/2),
#                                  gptq (full X^T X with error feedback by groups of 8 columns, as nslm/gptq4.h)
#             tq:K:L:POW           the repo's trellis encoder with weights h^POW (0: none; 1: the packer's choice)
#   h = diag(X^T X).  Environment: MATS=gate,up,down (the matrices to quantize), CAL_SKIP=n (drop the first n tokens of
#   each window from X^T X), DAMP=d (GPTQ damping, times the mean diagonal; 0.01).
# The MLP matrices of each listed layer are quantized in order (gate, up, then down from the quantized gate / up);
# layers in order, each from the inputs of the already-quantized model before it (sequential, as GPTQ).
import json, struct, sys, time
import numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
import os
import seedlib, ctypes
_tq = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtq.so"))
_tq.tq_encode.restype = ctypes.c_double
_tq.tq_encode.argtypes = [np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"), ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")]
def tq_block(cols): return 256 if cols % 256 == 0 else 320 if cols % 320 == 0 else cols   # nslm/moe.c
def tq_quant(w, H, K, L, pw):
    """The repo's trellis encoder; importance h = diag(X^T X)^pw per column (pw 0: none; 1: the packer's imatrix)."""
    w = np.ascontiguousarray(w, np.float32); out = np.zeros_like(w)
    h = None
    if pw > 0:
        d = np.maximum(np.diag(H), 1e-30); h = np.ascontiguousarray((d / d.mean()) ** pw, np.float32)
    _tq.tq_encode(w, h.ctypes.data if h is not None else None, w.shape[0], w.shape[1], tq_block(w.shape[1]), L, K, out)
    return out
MATS = os.environ.get("MATS", "gate,up,down").split(",")
CAL_SKIP, DAMP = int(os.environ.get("CAL_SKIP", "0")), float(os.environ.get("DAMP", "0.01"))

D = sys.argv[1]
cfg = json.load(open(f"{D}/config.json"))
HID, NH, NKV, NL = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"]
HD, EPS, THETA = HID // NH, cfg["rms_norm_eps"], cfg["rope_theta"]
T, NEVAL, NCAL = 512, 16, 64

def load_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for k, v in h.items():
        if k == "__metadata__": continue
        a, b = v["data_offsets"]
        assert v["dtype"] == "BF16"
        raw = np.frombuffer(buf[8 + n + a: 8 + n + b], dtype=np.uint16)
        out[k] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
    return out

W0 = load_st(f"{D}/model.safetensors")
E = W0["model.embed_tokens.weight"]
def wk(i, n): return f"model.layers.{i}.{n}.weight"

inv = 1.0 / THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD)
fr = np.outer(np.arange(T), inv)
COS = np.cos(np.concatenate([fr, fr], -1)).astype(np.float32)[None, :, None, :]
SIN = np.sin(np.concatenate([fr, fr], -1)).astype(np.float32)[None, :, None, :]
MASK = np.triu(np.full((T, T), -np.inf, np.float32), 1)

def rms(x, w): return (x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * w
def rot(x): return np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1)
def silu(x): return x / (1.0 + np.exp(-x))

def attn(x, i, W):
    B = x.shape[0]
    h = rms(x, W[wk(i, "input_layernorm")])
    q = (h @ W[wk(i, "self_attn.q_proj")].T).reshape(B, T, NH, HD)
    k = (h @ W[wk(i, "self_attn.k_proj")].T).reshape(B, T, NKV, HD)
    v = (h @ W[wk(i, "self_attn.v_proj")].T).reshape(B, T, NKV, HD)
    q = q * COS + rot(q) * SIN
    k = k * COS + rot(k) * SIN
    q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
    k, v = np.repeat(k, NH // NKV, 1), np.repeat(v, NH // NKV, 1)
    s = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(HD)) + MASK
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, HID)
    return x + o @ W[wk(i, "self_attn.o_proj")].T

def mlp_parts(x, i, W):   # returns (mlp input h2, down input a)
    h2 = rms(x, W[wk(i, "post_attention_layernorm")])
    a = silu(h2 @ W[wk(i, "mlp.gate_proj")].T) * (h2 @ W[wk(i, "mlp.up_proj")].T)
    return h2, a

def layer(x, i, W):
    x = attn(x, i, W)
    _, a = mlp_parts(x, i, W)
    return x + a @ W[wk(i, "mlp.down_proj")].T

def batched(f, x, bs=4):
    return np.concatenate([f(x[j:j + bs]) for j in range(0, len(x), bs)])

def logprobs(x, W):
    z = rms(x, W["model.norm.weight"]) @ E.T
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))

tok = Tokenizer.from_file(f"{D}/tokenizer.json")
def windows(path, n):
    ids = np.array(tok.encode("".join(pq.read_table(path).column("text").to_pylist())).ids, dtype=np.int64)
    return ids[: n * T].reshape(n, T)
ev, ca = windows(f"{D}/wt2-test.parquet", NEVAL), windows(f"{D}/wt2-train.parquet", NCAL)

# ---- quantizers: W [out, in] -> W_hat ----
def q8_0(w):
    b = w.reshape(w.shape[0], -1, 32)
    d = (np.abs(b).max(-1, keepdims=True) / 127).astype(np.float16).astype(np.float32)
    q = np.clip(np.round(np.divide(b, d, out=np.zeros_like(b), where=d > 0)), -127, 127)
    return (q * d).reshape(w.shape)
def q4_0(w):
    b = w.reshape(w.shape[0], -1, 32)
    idx = np.abs(b).argmax(-1)[..., None]
    d = (np.take_along_axis(b, idx, -1) / -8).astype(np.float16).astype(np.float32)
    q = np.clip(np.round(np.divide(b, d, out=np.zeros_like(b), where=d != 0)), -8, 7)
    return (q * d).reshape(w.shape)

def seed_quant(w, H, method):
    w = w.astype(np.float32).copy()
    out, n = w.shape
    Q = np.zeros_like(w)
    if method == "gptq":
        Hd = H.astype(np.float64).copy()
        Hd[np.diag_indices(n)] += DAMP * np.mean(np.diag(Hd))
        U = np.linalg.cholesky(np.linalg.inv(Hd)).T   # upper, Hinv = U^T U
    for g in range(n // 8):
        c = slice(8 * g, 8 * g + 8)
        if method == "plain": A = np.eye(8)
        elif method == "diag": A = np.diag(np.sqrt(np.maximum(np.diag(H)[c], 1e-12)))
        elif method == "diag25": A = np.diag(np.maximum(np.diag(H)[c], 1e-12) ** 0.25)
        else:
            Tm = np.linalg.inv(U[c, c])
            A = Tm.T
            A = A / np.abs(A).max()   # scale does not change the arg-min; keeps f32 well inside range
        wd, _, _, _ = seedlib.search(A, w[:, c])
        Q[:, c] = wd
        if method == "gptq" and 8 * g + 8 < n:
            Eg = (w[:, c] - wd).astype(np.float64) @ Tm
            w[:, 8 * g + 8:] -= (Eg @ U[c, 8 * g + 8:]).astype(np.float32)
    return Q

# ---- reference ----
t0 = time.time()
x_ev, x_ca = E[ev], E[ca]
LAYERS = []
for part in sys.argv[2].split(","):
    a, _, b = part.partition("-")
    LAYERS += list(range(int(a), int(b or a) + 1))
L0 = min(LAYERS)
for i in range(L0):
    x_ev = batched(lambda x: layer(x, i, W0), x_ev)
    x_ca = batched(lambda x: layer(x, i, W0), x_ca)
xr = x_ev
for i in range(L0, NL): xr = batched(lambda x: layer(x, i, W0), xr)
LPR = batched(lambda x: logprobs(x, W0), xr)        # [NEVAL, T, V]
ppl_ref = float(np.exp(-np.take_along_axis(LPR[:, :-1], ev[:, 1:, None], -1).mean()))
print(f"reference: {NEVAL * T} eval tokens, {NCAL * T} calibration tokens, layers {L0}..{max(LAYERS)} quantized, "
      f"perplexity {ppl_ref:.3f} ({time.time() - t0:.0f} s)", flush=True)

def evaluate(W):
    x = x_ev
    for i in range(L0, NL): x = batched(lambda y: layer(y, i, W), x)
    kl, nll = 0.0, 0.0
    for j in range(NEVAL):
        lp = logprobs(x[j:j + 1], W)[0]
        kl += float((np.exp(LPR[j]) * (LPR[j] - lp)).sum(-1).mean())
        nll += float(-np.take_along_axis(lp[:-1], ev[j, 1:, None], -1).mean())
    return kl / NEVAL, float(np.exp(nll / NEVAL))

def run_variant(spec):
    t = time.time()
    W = dict(W0)
    parts = spec.split(":")
    if parts[0] not in ("q8_0", "q4_0", "tq"):
        method, P, K = parts[0], int(parts[1]), int(parts[2])
        seedlib.setup(P, K, int(parts[3]) if len(parts) > 3 else 1)
    x = x_ca
    rel = []
    for i in range(L0, NL):
        if i in LAYERS:
            xa = batched(lambda y: attn(y, i, W), x)
            h2 = xa[:, CAL_SKIP:].reshape(-1, HID).astype(np.float64)
            Hg = h2.T @ h2
            for name in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                key = wk(i, name)
                if name == "mlp.down_proj":
                    a = batched(lambda y: mlp_parts(y, i, W)[1], xa)[:, CAL_SKIP:].reshape(-1, W[key].shape[1]).astype(np.float64)
                    H, X = a.T @ a, a
                else: H, X = Hg, h2
                w = W0[key]
                if name.split(".")[1].split("_")[0] not in MATS: continue
                dg = np.sort(np.diag(H))[::-1]
                if spec == sys.argv[3]: print(f"  {name}: top 1% of channels hold {dg[:max(1, len(dg) // 100)].sum() / dg.sum():.1%} of trace(X^T X); max / median {dg[0] / np.median(dg):.0f}", flush=True)
                if parts[0] == "q8_0": Wq = q8_0(w)
                elif parts[0] == "q4_0": Wq = q4_0(w)
                elif parts[0] == "tq": Wq = tq_quant(w, H, int(parts[1]), int(parts[2]), float(parts[3]))
                else: Wq = seed_quant(w, H, method)
                dW = (w - Wq).astype(np.float64)
                rel.append(float(np.einsum("ij,jk,ik->", dW, H, dW) / np.einsum("ij,jk,ik->", w.astype(np.float64), H, w.astype(np.float64))))
                W[key] = Wq.astype(np.float32)
        x = batched(lambda y: layer(y, i, W), x) if i < max(LAYERS) else x
    tq = time.time() - t
    kl, ppl = evaluate(W)
    print(f"{spec:16s} mats {'+'.join(MATS)} skip {CAL_SKIP} damp {DAMP:g}  output err {np.mean(rel):.5f}  KL {kl:.6f}  perplexity {ppl:.3f}  (quantize {tq:.0f} s, total {time.time() - t:.0f} s)", flush=True)

for spec in sys.argv[3:]: run_variant(spec)
