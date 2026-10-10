#!/usr/bin/env python3
"""tools/nanoseedlm_k2.py against a scalar reference, and a small model through mlx-lm's loader.

  OMLX_K2_MODEL=/path/to/k2_horizon_model.py python tools/test_nanoseedlm_k2.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import mova_common  # noqa: E402

mova_common.register()
import nanoseedlm_k2 as ns  # noqa: E402

R32 = np.float32(1.0) / np.float32(32767.0)


def lfsr_step(s):
    return (s >> 1) | (((s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1) << 15)


def ref_block(seed, coef, e):
    st, s = [], int(seed)
    for _ in range(32):
        s = lfsr_step(s)
        st.append(s)
    q = [((int(coef) >> (4 * p)) & 15) - (16 if (int(coef) >> (4 * p)) & 8 else 0) for p in range(4)]
    sc = np.float32(np.ldexp(R32, e))
    return np.array([np.float32(sum((st[4 * c + p] - 32768) * q[p] for p in range(4))) * sc for c in range(8)], np.float32)


def ref_decode(seeds, coefs, codes, bias):
    e, n, kb = seeds.shape
    s, c, k = seeds.reshape(-1), coefs.reshape(-1), codes.reshape(-1)
    out = np.empty((s.size, 8), np.float32)
    for g in range(s.size):
        out[g] = ref_block(s[g], c[g], int(bias[g // (n * kb)]) + int((k[g >> 1] >> ((g & 1) * 4)) & 15))
    return out.reshape(e, n, kb * 8)


def ref_decode8(seeds, coefs, codes, bias):
    """P = 8 blocks (searchp.h): 64 states, 8 int4 coefficients in a 32-bit word."""
    e, n, kb = seeds.shape
    s, c, k = seeds.reshape(-1), coefs.reshape(-1), codes.reshape(-1)
    out = np.empty((s.size, 8), np.float32)
    for g in range(s.size):
        st, v = [], int(s[g])
        for _ in range(64):
            v = lfsr_step(v)
            st.append(v)
        q = [((int(c[g]) >> (4 * p)) & 15) - (16 if (int(c[g]) >> (4 * p)) & 8 else 0) for p in range(8)]
        sc = np.float32(np.ldexp(R32, int(bias[g // (n * kb)]) + int((k[g >> 1] >> ((g & 1) * 4)) & 15)))
        out[g] = np.array([np.float32(sum((st[8 * cc + p] - 32768) * q[p] for p in range(8))) * sc for cc in range(8)], np.float32)
    return out.reshape(e, n, kb * 8)


def ref_decode3(seeds, nibbles, bias):
    """P = 3 blocks (SEED4, lfsr.h): 24 states; nibble word = exponent code | q_p << (4 p + 4)."""
    e, n, kb = seeds.shape
    s, c = seeds.reshape(-1), nibbles.reshape(-1)
    out = np.empty((s.size, 8), np.float32)
    for g in range(s.size):
        st, v = [], int(s[g])
        for _ in range(24):
            v = lfsr_step(v)
            st.append(v)
        nw = int(c[g])
        q = [((nw >> (4 * p + 4)) & 15) - (16 if (nw >> (4 * p + 4)) & 8 else 0) for p in range(3)]
        sc = np.float32(np.ldexp(R32, int(bias[g // (n * kb)]) + (nw & 15)))
        out[g] = np.array([np.float32(sum((st[3 * cc + p] - 32768) * q[p] for p in range(3))) * sc for cc in range(8)], np.float32)
    return out.reshape(e, n, kb * 8)


def random_seeds3(shape, rng, bias=-22):
    e, n, k = shape
    return (rng.integers(1, 65536, (e, n, k // 8)).astype(np.uint16), rng.integers(0, 65536, (e, n, k // 8)).astype(np.uint16),
            (bias + rng.integers(-2, 3, e)).astype(np.int32))


def p3_parts(parts):
    """SEED4 streams as the loader's modules hold them: nibble words as int16, codes a placeholder."""
    seeds, nib, bias = parts
    return mx.array(seeds), mx.array(nib).view(mx.int16), mx.zeros((1,), mx.uint8), mx.array(bias)


def random_seeds8(shape, rng, bias=-22):
    e, n, k = shape
    return (rng.integers(1, 65536, (e, n, k // 8)).astype(np.uint16), rng.integers(0, 2 ** 32, (e, n, k // 8), dtype=np.uint64).astype(np.uint32),
            rng.integers(0, 256, (e, n, k // 16)).astype(np.uint8), (bias + rng.integers(-2, 3, e)).astype(np.int32))


def bf16_bits(w):
    u = w.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def random_seeds(shape, rng, bias=-22):
    e, n, k = shape
    return (rng.integers(1, 65536, (e, n, k // 8)).astype(np.uint16), rng.integers(0, 65536, (e, n, k // 8)).astype(np.uint16),
            rng.integers(0, 256, (e, n, k // 16)).astype(np.uint8), (bias + rng.integers(-2, 3, e)).astype(np.int32))


def rel_err(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.sqrt(np.mean((a - b) ** 2) / max(np.mean(a * a), 1e-30)))


def test_table():
    t = np.array(ns.lfsr_table())
    for seed in (1, 2, 0xACE1, 0xFFFF):
        s, g = seed, 0
        for k in range(32):
            s = lfsr_step(s)
            g |= (s >> 15) << k
        assert t[seed] == g


def test_decode_bitwise():
    parts = random_seeds((3, 8, 32), np.random.default_rng(7))
    got = np.array(ns.decode(*map(mx.array, parts)).view(mx.uint16))
    assert np.array_equal(got, bf16_bits(ref_decode(*parts)))


def test_matvec():
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        rng = np.random.default_rng(5)
        parts = random_seeds((5, 37, 48), rng, -20)
        w = ref_decode(*parts)
        x = mx.array(rng.standard_normal((9, 48)).astype(np.float32)).astype(dtype)
        idx = rng.integers(0, 5, 9).astype(np.uint32)
        got = ns.gather_matvec(x, mx.array(idx), *map(mx.array, parts))
        want = np.einsum("mk,mnk->mn", np.array(x.astype(mx.float32)), w[idx])
        assert rel_err(mx.array(want), got) < (2e-2 if dtype != mx.float32 else 1e-5), dtype


def test_switch_linear():
    for tokens in (1, 3, 40):
        rng = np.random.default_rng(tokens)
        parts = random_seeds((6, 32, 64), rng, -20)
        layer = ns.SeedSwitchLinear(64, 32, 6)
        layer.seeds, layer.coefs, layer.codes, layer.exp_bias = map(mx.array, parts)
        x = mx.array(rng.standard_normal((tokens, 1, 1, 64)).astype(np.float32))
        idx = mx.array(rng.integers(0, 6, (tokens, 2)).astype(np.uint32))
        want = mx.gather_mm(x, mx.array(ref_decode(*parts)).swapaxes(-1, -2), rhs_indices=idx)
        got = layer(x, idx)
        assert got.shape == want.shape and rel_err(want, got) < 1e-2, tokens


def sorted_pairs(rng, pairs, experts, empty=()):
    live = [e for e in range(experts) if e not in empty]
    return np.sort(rng.choice(live, pairs)).astype(np.uint32)


def test_gather_gemm():
    """Expert-sorted pairs through the seeded tile GEMM: weights rebuilt as decode's BF16, f32 accumulation."""
    for dtype in (mx.bfloat16, mx.float32):
        rng = np.random.default_rng(9)
        parts = random_seeds((5, 37, 96), rng, -20)
        w = np.array(mx.array(ref_decode(*parts)).astype(mx.bfloat16).astype(mx.float32))
        for pairs in (33, 150, 700):
            idx = sorted_pairs(rng, pairs, 5, empty=(2,))
            x = mx.array(rng.standard_normal((pairs, 96)).astype(np.float32)).astype(dtype)
            got = ns.gather_gemm(x, mx.array(idx), *map(mx.array, parts))
            want = np.einsum("mk,mnk->mn", np.array(x.astype(mx.float32)), w[idx])
            assert got.shape == (pairs, 37) and got.dtype == dtype
            assert rel_err(mx.array(want), got) < (4e-3 if dtype == mx.bfloat16 else 1e-5), (dtype, pairs)


def test_combine_sorted():
    """Expert-sorted rows reduced into tokens with the router weights: bitwise K2's (y * w).sum(-2) on unsorted y."""
    for dtype in (mx.bfloat16, mx.float32):
        rng = np.random.default_rng(17)
        for tokens, k in ((5, 8), (300, 8), (37, 3)):
            idx = mx.array(np.stack([rng.permutation(12)[:k] for _ in range(tokens)]).astype(np.uint32))
            order = mx.argsort(idx.reshape(-1))
            ys = mx.array(rng.standard_normal((tokens * k, 64)).astype(np.float32)).astype(dtype)
            w = mx.array(rng.random((tokens, k)).astype(np.float32))
            y = ns._unsort(ys, order, idx.shape)
            want = (y * w.astype(dtype)[..., None]).sum(axis=-2)
            got = ns.combine_sorted(ys, order, w)
            assert got.shape == want.shape and got.dtype == dtype
            assert bool(mx.all(got == want).item()), (dtype, tokens, k, rel_err(want, got))


def test_switch_glu_without_decode():
    """SeedSwitchGLU above the mat-vec limit: matches SwitchGLU on the decoded weights, never builds a BF16 expert."""
    from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear
    real_decode, ns.decode = ns.decode, None
    try:
        for tokens in (5, 40, 300):
            rng = np.random.default_rng(tokens)
            glu, ref = SwitchGLU(64, 96, 6), SwitchGLU(64, 96, 6)
            glu.__class__ = ns.SeedSwitchGLU
            for name, (k, n) in {"gate_proj": (64, 96), "up_proj": (64, 96), "down_proj": (96, 64)}.items():
                parts = random_seeds((6, n, k), rng, -18)
                layer = ns.SeedSwitchLinear(k, n, 6)
                layer.seeds, layer.coefs, layer.codes, layer.exp_bias = map(mx.array, parts)
                setattr(glu, name, layer)
                lin = SwitchLinear(k, n, 6, bias=False)
                lin.weight = mx.array(ref_decode(*parts)).astype(mx.bfloat16)
                setattr(ref, name, lin)
            x = mx.array(rng.standard_normal((1, tokens, 64)).astype(np.float32)).astype(mx.bfloat16)
            idx = mx.array(np.stack([rng.permutation(6)[:4] for _ in range(tokens)])[None].astype(np.uint32))
            want, got = ref(x, idx), glu(x, idx)
            assert got.shape == want.shape == (1, tokens, 4, 64)
            assert rel_err(want, got) < 1e-2, tokens
    finally:
        ns.decode = real_decode


def test_seed_linear():
    """A dense seeded projection: the mat-vec for few rows, the tile GEMM above; x @ W^T on the decoded BF16."""
    for tokens in (1, 3, 40):
        rng = np.random.default_rng(100 + tokens)
        parts = random_seeds((1, 40, 64), rng, -20)
        layer = ns.SeedLinear(64, 40)
        layer.seeds, layer.coefs, layer.codes, layer.exp_bias = map(mx.array, parts)
        x = mx.array(rng.standard_normal((1, tokens, 64)).astype(np.float32)).astype(mx.bfloat16)
        want = x @ mx.array(ref_decode(*parts)[0]).astype(mx.bfloat16).T
        got = layer(x)
        assert got.shape == want.shape == (1, tokens, 40) and rel_err(want, got) < 1e-2, tokens


def test_seed_embedding():
    """Seeded embedding rows: bitwise the decoded BF16 rows."""
    rng = np.random.default_rng(7)
    parts = random_seeds((1, 50, 64), rng, -18)
    emb = ns.SeedEmbedding(50, 64)
    emb.seeds, emb.coefs, emb.codes, emb.exp_bias = map(mx.array, parts)
    ids = mx.array(rng.integers(0, 50, (2, 7)).astype(np.int32))
    got = emb(ids)
    want = bf16_bits(ref_decode(*parts)[0])[np.array(ids)]
    assert got.shape == (2, 7, 64) and got.dtype == mx.bfloat16
    assert np.array_equal(np.array(got.view(mx.uint16)), want)


def test_p8_dense():
    """P = 8 SeedLinear (mat-vec and GEMM) and SeedEmbedding against the scalar decode."""
    rng = np.random.default_rng(23)
    parts = random_seeds8((1, 40, 64), rng, -20)
    w = ref_decode8(*parts)[0]
    assert np.array_equal(np.array(ns.decode(*map(mx.array, parts))[0].view(mx.uint16)), bf16_bits(w))
    layer = ns.SeedLinear(64, 40, p=8)
    layer.seeds, layer.coefs, layer.codes, layer.exp_bias = map(mx.array, parts)
    for tokens in (1, 3, 40):
        x = mx.array(rng.standard_normal((1, tokens, 64)).astype(np.float32)).astype(mx.bfloat16)
        want = x @ mx.array(w).astype(mx.bfloat16).T
        got = layer(x)
        assert got.shape == (1, tokens, 40) and rel_err(want, got) < 1e-2, tokens
    emb = ns.SeedEmbedding(40, 64, p=8)
    emb.seeds, emb.coefs, emb.codes, emb.exp_bias = map(mx.array, parts)
    ids = mx.array(rng.integers(0, 40, (2, 5)).astype(np.int32))
    assert np.array_equal(np.array(emb(ids).view(mx.uint16)), bf16_bits(w)[np.array(ids)])


def test_p3():
    """P = 3 (SEED4) decode bitwise, SeedLinear (mat-vec and GEMM), SeedEmbedding and SeedSwitchLinear against the scalar decode."""
    rng = np.random.default_rng(31)
    parts = random_seeds3((2, 40, 64), rng, -20)
    w = ref_decode3(*parts)
    assert np.array_equal(np.array(ns.decode(*p3_parts(parts)).view(mx.uint16)), bf16_bits(w))
    one = tuple(a[:1] for a in parts)
    layer = ns.SeedLinear(64, 40, p=3)
    layer.seeds, layer.coefs, layer.codes, layer.exp_bias = p3_parts(one)
    for tokens in (1, 3, 40):
        x = mx.array(rng.standard_normal((1, tokens, 64)).astype(np.float32)).astype(mx.bfloat16)
        got, want = layer(x), x @ mx.array(w[0]).astype(mx.bfloat16).T
        assert got.shape == (1, tokens, 40) and rel_err(want, got) < 1e-2, tokens
    emb = ns.SeedEmbedding(40, 64, p=3)
    emb.seeds, emb.coefs, emb.codes, emb.exp_bias = p3_parts(one)
    ids = mx.array(rng.integers(0, 40, (2, 5)).astype(np.int32))
    assert np.array_equal(np.array(emb(ids).view(mx.uint16)), bf16_bits(w[0])[np.array(ids)])
    sw = ns.SeedSwitchLinear(64, 40, 2, p=3)
    sw.seeds, sw.coefs, sw.codes, sw.exp_bias = p3_parts(parts)
    x = mx.array(rng.standard_normal((5, 1, 64)).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(rng.integers(0, 2, (5, 2)).astype(np.uint32))
    got = sw(mx.expand_dims(x, -3), idx)
    wb = mx.array(w).astype(mx.bfloat16)
    want = mx.stack([mx.stack([x[t, 0] @ wb[int(idx[t, j])].T for j in range(2)]) for t in range(5)])[:, :, None]
    assert rel_err(want, got) < 1e-2


def test_seed_heads():
    """Per-head maps (MLA) at P = 4 and 8: y[b, a, l] = W_a x[b, a, l], the mat-vec (decode) and the GEMM (prompt)."""
    rng = np.random.default_rng(41)
    for gen, ref, p in ((random_seeds, ref_decode, 4), (random_seeds8, ref_decode8, 8)):
        parts = gen((4, 24, 32), rng, -20)
        heads = ns.SeedHeads(4, 32, 24, p=p)
        heads.seeds, heads.coefs, heads.codes, heads.exp_bias = map(mx.array, parts)
        wb = mx.array(ref(*parts)).astype(mx.bfloat16)
        for b, l in ((1, 1), (2, 3), (1, 40)):
            x = mx.array(rng.standard_normal((b, 4, l, 32)).astype(np.float32)).astype(mx.bfloat16)
            got, want = heads(x), mx.einsum("aoi,bali->balo", wb, x)
            assert got.shape == (b, 4, l, 24) and rel_err(want, got) < 1e-2, (p, b, l)


DENSE_SEEDED = ("model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.o_proj", "model.layers.1.mlp.shared_experts.down_proj",
                "lm_head", "model.embed_tokens")


DENSE_P8 = ("model.layers.1.self_attn.o_proj", "lm_head")


def small_config():
    return dict(model_type="k2_horizon", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=128, rms_norm_eps=1e-5, layernorm_num_groups=2,
                mlp_only_layers=[0], num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64, num_shared_experts=1,
                moe_gate_bias=True, norm_topk_prob=True, router_score_func="sigmoid", router_scaling_factor=1,
                query_key_norm=False, rope_parameters={"rope_type": "default", "rope_theta": 10000},
                mova_num_experts=4, mova_num_experts_per_tok=2, attention_gate_func="softplus")


def test_load_through_mlx_lm():
    from mlx_lm.models import k2_horizon
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.utils import load_model

    config = small_config()
    mx.random.seed(3)
    ref = k2_horizon.Model(k2_horizon.ModelArgs.from_dict(config))
    ref.set_dtype(mx.bfloat16)
    rng = np.random.default_rng(11)
    weights, seeded, quant = {}, {}, {}
    for name, value in tree_flatten(ref.parameters()):
        path = name.rsplit(".", 1)[0]
        hf = name.replace("mlp.expert_bias", "mlp.gate.bias").replace("self_attn.v_expert_bias", "self_attn.v_router.bias")
        if "mlp.experts." in name or "v_experts" in name or path in DENSE_SEEDED:
            gen = random_seeds8 if path in DENSE_P8 else random_seeds
            parts = gen(value.shape if value.ndim == 3 else (1,) + value.shape, rng)
            seeded[path] = parts
            for k, v in zip(("seeds", "coefs", "codes", "exp_bias"), parts):
                weights[f"{path}.{k}"] = mx.array(v)
        elif name.endswith(".weight") and value.ndim >= 2 and not path.endswith(("mlp.gate", "v_router")):
            bits = 4 if "v_experts" in name else 8
            w, s, b = mx.quantize(value, 64, bits)
            weights.update({f"{path}.weight": w, f"{path}.scales": s, f"{path}.biases": b})
            quant[path] = bits
        else:
            weights[hf] = value
    with tempfile.TemporaryDirectory() as d:
        mx.save_safetensors(f"{d}/model.safetensors", weights, metadata={"format": "mlx"})
        q = {"group_size": 64, "bits": 8, "mode": "affine"}
        q.update({p: {"group_size": 64, "bits": 4} for p, b in quant.items() if b == 4})
        Path(d, "config.json").write_text(json.dumps(dict(config, quantization=q, model_file="nanoseedlm_k2.py")))
        shutil.copy(TOOLS / "nanoseedlm_k2.py", d)
        model, _ = load_model(Path(d), trust_remote_code=True)

    nn.quantize(ref, class_predicate=lambda p, m: {"group_size": 64, "bits": quant[p]} if p in quant else False)
    for path, parts in seeded.items():
        e, n, k = parts[0].shape
        w = mx.array(bf16_bits((ref_decode8 if path in DENSE_P8 else ref_decode)(*parts))).view(mx.bfloat16)
        if path in DENSE_SEEDED:
            layer = nn.Embedding(n, k * 8) if path.endswith("embed_tokens") else nn.Linear(k * 8, n, bias=False)
            layer.weight = w[0]
        else:
            layer = SwitchLinear(k * 8, n, e, bias=False)
            layer.weight = w
        ref.update_modules(tree_unflatten([(path, layer)]))
    assert type(model.model.layers[1].mlp.experts).__name__ == "SeedSwitchGLU"   # model_file is imported as its own module
    assert type(model.model.layers[1].self_attn.v_experts).__name__ == "SeedSwitchLinear"
    assert type(model.model.layers[1].self_attn.q_proj).__name__ == "SeedLinear"
    assert type(model.lm_head).__name__ == "SeedLinear" and model.lm_head.coefs.dtype == mx.uint32
    assert type(model.model.embed_tokens).__name__ == "SeedEmbedding"
    assert isinstance(model.model.layers[1].self_attn.k_proj, nn.QuantizedLinear)
    assert type(model.model.layers[1].mlp).__name__ == "SeedSparseMoeBlock"
    for ids in (mx.array([[5]]), mx.array([list(range(1, 41))])):
        err = rel_err(ref(ids), model(ids))
        assert err < 0.02, err


def _load_mla(affine=False):
    """Seeded MLA, optionally mixed with the packer's Q8 / Q4 arrays, against the decoded BF16 model."""
    from mlx_lm.utils import load_model

    config = dict(small_config(), mla_ranks=[64, 128] if affine else [16, 32], mla_rope_dim=16)
    mx.random.seed(5)
    ref = ns.Model(ns.ModelArgs.from_dict(config))
    rng = np.random.default_rng(13)
    gens = {3: (random_seeds3, ref_decode3), 4: (random_seeds, ref_decode), 8: (random_seeds8, ref_decode8)}
    weights, kinds, quantized = {}, {}, {}
    for i, layer in enumerate(ref.model.layers):
        m = layer.self_attn.mla
        for j, (name, value) in enumerate(sorted(m.items())):
            path = f"model.layers.{i}.self_attn.mla.{name}"
            if affine and name in ("v_up", "kv_a_x", "k_rope_proj"):
                bits = 4 if name == "kv_a_x" else 8
                w = mx.array((rng.standard_normal(value.shape) * 0.03).astype(np.float32)).astype(mx.bfloat16)
                packed, scales, biases = mx.quantize(w, 64, bits)
                # Match ns_write's affine MLA names, including the Q8 v_up kept by default with --mla p4.
                weights.update({path: packed, path + ".scales": scales, path + ".biases": biases})
                setattr(m, name, mx.dequantize(packed, scales, biases, group_size=64, bits=bits).astype(mx.bfloat16))
                quantized[path] = bits
                continue
            p = (4, 8, 3)[(i + j) % 3]
            gen, dec = gens[p]
            shape = value.shape if value.ndim == 3 else (1,) + value.shape
            parts = gen(shape, rng, -14)
            w = mx.array(bf16_bits(dec(*parts))).view(mx.bfloat16)
            setattr(m, name, w if value.ndim == 3 else w[0])
            keys = ("seeds", "nibbles", "exp_bias") if p == 3 else ("seeds", "coefs", "codes", "exp_bias")
            weights.update({f"{path}.{k}": mx.array(v) for k, v in zip(keys, parts)})
            kinds[path] = (value.ndim, p)
    ref.set_dtype(mx.bfloat16)
    for name, value in tree_flatten(ref.parameters()):
        if ".mla." not in name:
            weights[name.replace("mlp.expert_bias", "mlp.gate.bias").replace("self_attn.v_expert_bias", "self_attn.v_router.bias")] = value
    with tempfile.TemporaryDirectory() as d:
        mx.save_safetensors(f"{d}/model.safetensors", weights, metadata={"format": "mlx"})
        if affine:
            config["quantization"] = {"group_size": 64, "bits": 8, "mode": "affine"}
            config["quantization"].update({p: {"group_size": 64, "bits": b} for p, b in quantized.items() if b == 4})
        Path(d, "config.json").write_text(json.dumps(dict(config, model_file="nanoseedlm_k2.py")))
        shutil.copy(TOOLS / "nanoseedlm_k2.py", d)
        model, _ = load_model(Path(d), trust_remote_code=True)
    assert len(kinds) + len(quantized) == 11 and {p for _, p in kinds.values()} == {3, 4, 8}
    params = dict(tree_flatten(model.parameters()))
    ref_params = dict(tree_flatten(ref.parameters()))
    for path in quantized:
        assert params[path].dtype == mx.bfloat16 and params[path].shape == ref_params[path].shape
        assert np.array_equal(np.array(params[path].view(mx.uint16)), np.array(ref_params[path].view(mx.uint16)))
        assert path + ".scales" not in params and path + ".biases" not in params
    for path, (ndim, p) in kinds.items():
        layer, name = path.split(".self_attn.mla.")
        mod = getattr(model.model.layers[int(layer.rsplit(".", 1)[1])].self_attn.mla, name)
        assert type(mod).__name__ == ("SeedHeads" if ndim == 3 else "SeedLinear"), path
    assert type(model.model.layers[1].self_attn).__name__ == "MLAAttention" and "k_proj" not in model.model.layers[1].self_attn
    for ids in (mx.array([[5]]), mx.array([list(range(1, 41))])):
        err = rel_err(ref(ids), model(ids))
        assert err < 0.02, err


def test_load_mla():
    _load_mla()


def test_load_mla_affine():
    _load_mla(affine=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    print("test_nanoseedlm_k2: PASS")
