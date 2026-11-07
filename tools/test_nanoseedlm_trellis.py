#!/usr/bin/env python3
"""tools/nanoseedlm_trellis.py (trellis-coded experts in MLX) against nslm/lib_trellis.c, and the tiny qwen4 model with
trellis banks through mlx-vlm's loader (tools/nanoseedlm_qwen4.py).

  python tools/test_nanoseedlm_trellis.py      (needs cc, mlx and mlx-vlm with qwen4_exp)
"""
import ctypes
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import nanoseedlm_seeds as seeds  # noqa: E402
import nanoseedlm_trellis as tq  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="nslm_tq_"))
PROJ = ("gate_proj", "up_proj", "down_proj")


class Scratch(ctypes.Structure):   # NsTqScratch (nslm/trellis.h)
    _fields_ = [(n, ctypes.c_void_p) for n in ("cost", "ncost", "m", "ac", "jm", "back", "st", "rev")] + \
               [(n, ctypes.c_int) for n in ("L", "K", "T")]


def c_lib():
    lib = TMP / "libnslm_trellis.dylib"
    subprocess.run(["cc", "-std=c11", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-I", str(ROOT / "nslm"),
                    str(ROOT / "nslm" / "lib_trellis.c"), "-o", str(lib)], check=True)
    c = ctypes.CDLL(str(lib))
    f32, u32, u16 = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS") for t in (np.float32, np.uint32, np.uint16))
    c.nslm_tq_scratch_init.argtypes = [ctypes.POINTER(Scratch), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    c.nslm_tq_scratch_free.argtypes = [ctypes.POINTER(Scratch)]
    c.nslm_tq_encode_row.argtypes = [f32, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Scratch), u32, u16,
                                     ctypes.c_void_p]
    c.nslm_tq_encode_row.restype = ctypes.c_double
    c.nslm_tq_decode_row.argtypes = [u32, u16, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32]
    return c


C = c_lib()


def c_decode(words, scales, cols, L, K, T):
    """nslm_tq_decode_row of every row: words (E, rows, nw), scales (E, rows, nb) -> (E, rows, cols) float32."""
    e, n, _ = words.shape
    out = np.zeros((e, n, cols), np.float32)
    for i in range(e):
        for r in range(n):
            row = np.zeros(cols, np.float32)
            C.nslm_tq_decode_row(np.ascontiguousarray(words[i, r]), np.ascontiguousarray(scales[i, r]), cols, T, L, K, row)
            out[i, r] = row
    return out


def bf16_bits(a):
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)   # truncation: any BF16 value is fine


def random_bank(e, rows, cols, L, K, T, seed):
    """Random stream words and BF16 scales: every bit pattern is a valid trellis row."""
    rng = np.random.default_rng(seed)
    nw, nb = tq.row_words(cols, T, L, K), (cols + T - 1) // T
    words = rng.integers(0, 2**32, (e, rows, nw), dtype=np.uint64).astype(np.uint32)
    scales = bf16_bits(rng.uniform(0.5, 2.0, (e, rows, nb)).astype(np.float32) * 1e-3)
    return dict(rows=rows, cols=cols, experts=e, L=L, K=K, T=T, words=words, scales=scales)


def write_bank(path, b, rel=None):
    """A .blkt as nslm-moe and tcq_fixture write it: MoeBlkHeader, rel[E], wrel[E], words, scales."""
    with open(path, "wb") as f:
        f.write(struct.pack("<8s3I3I3ifd", b"NSLMBLKT", b["rows"], b["cols"], b["experts"], b["L"], b["K"], b["T"], 0, 0, 0,
                            0.0, 0.0))
        f.write(np.zeros(2 * b["experts"], np.float32).tobytes() if rel is None else rel.tobytes())
        f.write(np.ascontiguousarray(b["words"], np.uint32).tobytes())
        f.write(np.ascontiguousarray(b["scales"], np.uint16).tobytes())


# (L, K, T, rows, cols): the release shapes (gate / up 640 x 2560 in blocks of 256, down 2560 x 640 in blocks of 320) cut
# down in rows, K = 3 and 4, an L = 16 code, and tcq_fixture's tiny shapes (T = cols).
SHAPES = [(12, 3, 256, 24, 512), (12, 4, 256, 16, 512), (12, 3, 320, 20, 640), (12, 4, 320, 12, 640),
          (16, 3, 256, 8, 256), (12, 3, 64, 32, 64), (12, 3, 32, 64, 32)]


def test_bank_file_and_decode():
    for i, (L, K, T, rows, cols) in enumerate(SHAPES):
        b = random_bank(3, rows, cols, L, K, T, 10 + i)
        path = TMP / f"bank{i}.blkt"
        write_bank(path, b)
        r = tq.read_bank(str(path))
        assert (r["L"], r["K"], r["T"], r["rows"], r["cols"], r["experts"]) == (L, K, T, rows, cols, 3)
        assert np.array_equal(r["words"], b["words"]) and np.array_equal(r["scales"], b["scales"])
        ref = c_decode(b["words"], b["scales"], cols, L, K, T)
        got = np.array(tq.decode(mx.array(r["words"]), mx.array(r["scales"]), cols, L, K, T))
        assert got.dtype == np.float32 and np.array_equal(got, ref), (L, K, T, np.abs(got - ref).max())
    with open(TMP / "bank0.blkt", "r+b") as f:   # a wrong size is not a bank
        f.truncate(1000)
    try:
        tq.read_bank(str(TMP / "bank0.blkt"))
        raise AssertionError("a truncated bank was read")
    except ValueError:
        pass
    print("test_bank_file_and_decode: ok")


def test_encoded_rows():
    """Rows from nslm_tq_encode_row (tcq_fixture's way: L 12, K 3, T = cols) decode as nslm_tq_decode_row's."""
    for cols in (64, 32):
        z = Scratch()
        assert C.nslm_tq_scratch_init(ctypes.byref(z), cols, 12, 3) == 0
        nw = tq.row_words(cols, cols, 12, 3)
        words, scales = np.zeros((1, 8, nw), np.uint32), np.zeros((1, 8, 1), np.uint16)
        rng = np.random.default_rng(cols)
        for r in range(8):
            w = ((rng.random(cols) - 0.5) * 0.5).astype(np.float32)
            row, sc = np.zeros(nw, np.uint32), np.zeros(1, np.uint16)
            C.nslm_tq_encode_row(w, None, cols, cols, ctypes.byref(z), row, sc, None)
            words[0, r], scales[0, r] = row, sc
        C.nslm_tq_scratch_free(ctypes.byref(z))
        ref = c_decode(words, scales, cols, 12, 3, cols)
        got = np.array(tq.decode(mx.array(words), mx.array(scales), cols, 12, 3, cols))
        assert np.array_equal(got, ref)
    print("test_encoded_rows: ok")


def rel_err(a, b):
    a, b = (np.array(v.astype(mx.float32)) if isinstance(v, mx.array) else np.asarray(v, np.float32) for v in (a, b))
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def test_matvec_and_gemm():
    for i, (L, K, T, rows, cols) in enumerate(SHAPES):
        b = random_bank(4, rows, cols, L, K, T, 30 + i)
        words, scales = mx.array(b["words"]), mx.array(b["scales"])
        w = c_decode(b["words"], b["scales"], cols, L, K, T)
        rng = np.random.default_rng(40 + i)
        for dtype, tol in ((mx.float32, 2e-5), (mx.bfloat16, 1.6e-2)):
            for m in (1, 7, 150):   # the mat-vec (any count), and the GEMM's sorted pairs over several tiles
                x = rng.standard_normal((m, cols)).astype(np.float32)
                idx = np.sort(rng.integers(0, 4, m)).astype(np.uint32)
                xm = mx.array(x).astype(dtype)
                ref = np.einsum("mc,mrc->mr", np.array(xm.astype(mx.float32)), w[idx])
                mv = tq.gather_matvec(xm, mx.array(idx), words, scales, cols, L, K, T)
                assert mv.dtype == dtype and rel_err(mv, ref) < tol, ("matvec", L, K, T, m, dtype, rel_err(mv, ref))
                if cols % seeds.BK == 0:
                    gm = tq.gather_gemm(xm, mx.array(idx), words, scales, cols, L, K, T)
                    assert gm.dtype == dtype and rel_err(gm, ref) < tol, ("gemm", L, K, T, m, dtype, rel_err(gm, ref))
    print("test_matvec_and_gemm: ok")


def dense_linear(lin):
    """The same projection as mlx's SwitchLinear on the decoded float32 weights."""
    from mlx_lm.models.switch_layers import SwitchLinear
    ref = SwitchLinear(lin.input_dims, lin.output_dims, lin.num_experts, bias=False)
    ref.weight = tq.decode(lin.words, lin.block_scales, lin.input_dims, lin.L, lin.K, lin.T)
    return ref


def test_switch_linear():
    """x per token (..., 1, 1, cols) and per (token, expert) pair (..., k, 1, cols, as mlx-vlm's MTP verify gives
    down_proj), below and above the mat-vec limit, and pre-sorted rows (sorted_indices=True, oMLX's prefill)."""
    b = random_bank(16, 40, 512, 12, 4, 256, 50)
    lin, ref = tq.TrellisSwitchLinear(b), None
    ref = dense_linear(lin)
    rng = np.random.default_rng(51)
    for tokens in (1, 3, 4, 12, 40):
        idx = mx.array(np.stack([rng.choice(16, 6, replace=False) for _ in range(tokens)])[None].astype(np.uint32))
        for per_pair in (False, True):
            shape = (1, tokens, 6 if per_pair else 1, 1, 512)
            x = mx.array(rng.standard_normal(shape).astype(np.float32))
            got, want = lin(x, idx), ref(x, idx)
            assert got.shape == want.shape and rel_err(got, want) < 2e-5, (tokens, per_pair, rel_err(got, want))
    xs = mx.array(rng.standard_normal((90, 1, 512)).astype(np.float32))   # (pairs, 1, cols), as mlx-lm's _gather_sort
    si = mx.array(np.sort(rng.integers(0, 16, 90)).astype(np.uint32))
    for n in (20, 90):
        got = lin(xs[:n], si[:n], sorted_indices=True)
        want = ref(xs[:n], si[:n], sorted_indices=True)
        assert got.shape == want.shape and rel_err(got, want) < 2e-5, (n, got.shape, want.shape, rel_err(got, want))
    assert lin.to_quantized(group_size=64, bits=4) is lin   # mlx-vlm's quantize pass leaves it
    print("test_switch_linear: ok")


def test_switch_glu():
    """TrellisSwitchGLU (the fused mat-vec, the sorted GEMM, weighted) against mlx's SwitchGLU on the decoded weights."""
    from nanoseedlm_seeds import SwitchGLU
    glu, ref = SwitchGLU(512, 640, 16), SwitchGLU(512, 640, 16)
    for j, (p, (rows, cols, T)) in enumerate(zip(PROJ, ((640, 512, 256), (640, 512, 256), (512, 640, 320)))):
        lin = tq.TrellisSwitchLinear(random_bank(16, rows, cols, 12, 3, T, 60 + j))
        setattr(glu, p, lin)
        setattr(ref, p, dense_linear(lin))
    glu.__class__ = tq.TrellisSwitchGLU
    rng = np.random.default_rng(61)
    for tokens in (1, 2, 9, 33):
        x = mx.array((rng.standard_normal((1, tokens, 512)) * 0.5).astype(np.float32))
        idx = mx.array(np.stack([rng.choice(16, 4, replace=False) for _ in range(tokens)])[None].astype(np.uint32))
        wts = mx.softmax(mx.array(rng.standard_normal((1, tokens, 4)).astype(np.float32)), axis=-1)
        want = ref(x, idx)
        assert rel_err(glu(x, idx), want) < 3e-5, (tokens, rel_err(glu(x, idx), want))
        assert rel_err(glu.weighted(x, idx, wts), (want * wts[..., None]).sum(-2)) < 3e-5, tokens
    print("test_switch_glu: ok")


def tiny_folder(head: bool):
    """tcq_fixture's folder for the MLX loader: the tiny model without its routed experts, banks L0..L3 (L4: the MTP head)
    from nslm_tq_encode_row (L 12, K 3, T = cols, tcq_fixture's weights), and the loader beside them."""
    src = ROOT / "tests" / "data" / "qwen4_tiny"
    d = Path(tempfile.mkdtemp(prefix="tiny_head_" if head else "tiny_", dir=TMP))   # a new folder each call
    cfg = json.loads((src / "config.json").read_text())
    cfg["model_file"] = "nanoseedlm_qwen4.py"
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    w = mx.load(str(src / "model.safetensors"))
    mx.save_safetensors(str(d / "model.safetensors"), {k: v for k, v in w.items() if ".switch_mlp." not in k},
                        metadata={"format": "mlx"})
    if head:   # without the head's routed experts, as nslm-qwen4-pack --tq writes it
        h = mx.load(str(src / "model-mtp.safetensors"))
        mx.save_safetensors(str(d / "model-mtp.safetensors"), {k: v for k, v in h.items() if ".switch_mlp." not in k},
                            metadata={"format": "mlx"})
    for f in ("nanoseedlm_qwen4.py", "nanoseedlm_seeds.py", "nanoseedlm_trellis.py"):
        shutil.copy(TOOLS / f, d)
    t = cfg["text_config"]
    r = 777
    for layer in range(t["num_hidden_layers"] + head):
        for p in PROJ:
            rows, cols = (t["hidden_size"], t["moe_intermediate_size"]) if p == "down_proj" else \
                         (t["moe_intermediate_size"], t["hidden_size"])
            e, nw = t["num_experts"], tq.row_words(cols, cols, 12, 3)
            words, scales = np.zeros((e, rows, nw), np.uint32), np.zeros((e, rows, 1), np.uint16)
            z = Scratch()
            assert C.nslm_tq_scratch_init(ctypes.byref(z), cols, 12, 3) == 0
            for k in range(e * rows):
                v = np.zeros(cols, np.float32)
                for j in range(cols):   # tcq_fixture's LCG
                    r = (r * 1664525 + 1013904223) & 0xFFFFFFFF
                    v[j] = np.float32((np.float32(r >> 8) / np.float32(16777216.0) - np.float32(0.5)) * np.float32(0.5))
                row, sc = np.zeros(nw, np.uint32), np.zeros(1, np.uint16)
                C.nslm_tq_encode_row(v, None, cols, cols, ctypes.byref(z), row, sc, None)
                words[k // rows, k % rows], scales[k // rows, k % rows] = row, sc
            C.nslm_tq_scratch_free(ctypes.byref(z))
            write_bank(d / f"L{layer}_{p}.blkt", dict(rows=rows, cols=cols, experts=e, L=12, K=3, T=cols, words=words,
                                                       scales=scales))
    return d


def test_tiny_model():
    """mlx-vlm's load_model on the folder: every layer's experts are TrellisSwitchLinear (the strict load has their
    streams), and in float32 the logits equal those of the same model with SwitchLinear on the decoded weights."""
    from mlx_vlm.utils import load_model
    d = tiny_folder(head=False)
    model = load_model(d, strict=True)
    n = model.config.text_config.num_hidden_layers
    for layer in range(n):
        glu = model.language_model.model.layers[layer].mlp.switch_mlp
        assert type(glu) is tq.TrellisSwitchGLU and all(type(getattr(glu, p)) is tq.TrellisSwitchLinear for p in PROJ)
    model.update(tree_map(lambda v: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v, model.parameters()))
    ids = mx.array([[(37 * i + 11) % 512 for i in range(48)]])   # qwen4_golden's prompt, the model in float32
    got = model.language_model(ids).logits[0].astype(mx.float32)
    got1 = model.language_model(ids[:, :1]).logits[0].astype(mx.float32)   # 3 pairs a layer: the mat-vec path
    for layer in range(n):
        glu = model.language_model.model.layers[layer].mlp.switch_mlp
        for p in PROJ:
            setattr(glu, p, dense_linear(getattr(glu, p)))
        glu.__class__ = seeds.SwitchGLU
    want = model.language_model(ids).logits[0].astype(mx.float32)
    want1 = model.language_model(ids[:, :1]).logits[0].astype(mx.float32)
    err, scale = float(mx.abs(got - want).max()), float(mx.abs(want).max())
    err1 = float(mx.abs(got1 - want1).max())
    agree = int((mx.argmax(got, -1) == mx.argmax(want, -1)).sum())
    print(f"test_tiny_model: max |trellis - dequantized| {err:.2e} (GEMM, 48 tokens), {err1:.2e} (mat-vec, 1 token) of "
          f"max |logit| {scale:.3f}, argmax {agree} of 48: ", end="")
    assert err < 1e-4 * scale and err1 < 1e-4 * scale and agree == 48
    print("ok")


def converted(d: Path) -> Path:
    """A copy of folder d with its .blkt banks as model-tcq-*.safetensors shards: nslm-qwen4-pack --tcq-st (plan
    tcq-safetensors), the C converter that made the published packs."""
    pack = ROOT / "out" / "bin" / "nslm-qwen4-pack"
    assert pack.exists(), f"{pack}: build it first (make)"
    c = d.parent / (d.name + "_st")
    shutil.copytree(d, c)
    subprocess.run([str(pack), "--tcq-st", str(c)], check=True, stdout=subprocess.DEVNULL)
    assert not list(c.glob("*.blkt")) and list(c.glob("model-tcq-*.safetensors"))
    return c


def test_tiny_model_safetensors_banks():
    """The banks as safetensors entries (NAME.tcq_words, .tcq_scales, .tcq_params, .tcq_err) load as the .blkt banks do:
    every layer's experts are TrellisSwitchLinear, the strict load has no stray entry, and the logits are equal."""
    from mlx_vlm.utils import load_model
    d = tiny_folder(head=False)
    a, b = load_model(d, strict=True), load_model(converted(d), strict=True)
    n = b.config.text_config.num_hidden_layers
    for layer in range(n):
        glu = b.language_model.model.layers[layer].mlp.switch_mlp
        assert type(glu) is tq.TrellisSwitchGLU and all(type(getattr(glu, p)) is tq.TrellisSwitchLinear for p in PROJ)
    ids = mx.array([[(37 * i + 11) % 512 for i in range(48)]])
    for k in (48, 1):   # the GEMM path and the mat-vec path
        x, y = a.language_model(ids[:, :k]).logits, b.language_model(ids[:, :k]).logits
        assert mx.array_equal(x, y).item(), (k, float(mx.abs(x - y).max()))
    print("test_tiny_model_safetensors_banks: ok")


def test_tiny_model_safetensors_head_without_mtp():
    """The head's entries (mtp.layers.0.mlp.switch_mlp.*.tcq_*) are dropped when the model has no MTP module."""
    from mlx_vlm.utils import load_model
    model = load_model(converted(tiny_folder(head=True)), strict=True)
    names = [k for k, _ in tree_flatten(model.parameters()) if k.endswith(".words")]
    assert len(names) == 3 * model.config.text_config.num_hidden_layers and not any(k.startswith("mtp.") for k in names)
    print("test_tiny_model_safetensors_head_without_mtp: ok")


def test_tiny_model_head_bank_without_mtp():
    """A head bank (L{n_layers}) is skipped when the model has no MTP module (mlx-vlm drops mtp.*), as the strict load
    shows: no stray streams."""
    from mlx_vlm.utils import load_model
    model = load_model(tiny_folder(head=True), strict=True)
    names = [k for k, _ in tree_flatten(model.parameters()) if k.endswith(".words")]
    assert len(names) == 3 * model.config.text_config.num_hidden_layers and not any(k.startswith("mtp.") for k in names)
    print("test_tiny_model_head_bank_without_mtp: ok")


if __name__ == "__main__":
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
        print("test_nanoseedlm_trellis: PASS")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
