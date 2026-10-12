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


def ref_mla(q, qr, lat, kr, q_lat, v_up, scale, mask=None):
    """Absorbed MLA attention in float32: o_a = v_up_a softmax(scale ((q_lat_a q_a) . c + qr_a . kr)) c, causal from the
    bottom right (query l sees keys 0 .. N - L + l), or a boolean / additive mask [B, 1, L, N]."""
    f = lambda a: np.array(a.astype(mx.float32))
    q, qr, lat, kr, q_lat, v_up = map(f, (q, qr, lat, kr, q_lat, v_up))
    ql = np.einsum("are,bale->balr", q_lat, q)
    s = scale * (np.einsum("balr,bnr->baln", ql, lat[:, 0]) + np.einsum("bald,bnd->baln", qr, kr[:, 0]))
    L, N = s.shape[2], s.shape[3]
    if mask is None:
        s = np.where(np.arange(N)[None, :] > np.arange(L)[:, None] + N - L, -np.inf, s)
    else:
        m = np.array(mask) if mask.dtype == mx.bool_ else f(mask)
        s = np.where(m, s, -np.inf) if m.dtype == bool else s + m
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("aer,balr->bale", v_up, np.einsum("baln,bnr->balr", p, lat[:, 0]))


def _mla_inputs(rng, b, heads, L, n, r, hd, dr):
    g = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)
    return g(b, heads, L, hd), g(b, heads, L, dr), g(b, 1, n, r), g(b, 1, n, dr)


def test_mla_attention_paths():
    """Every attention path (absorbed MLX reference, blocked absorbed, blocked expanded) against the float32 reference,
    on the real model's head shapes, with BF16 and with seeded q_lat / v_up."""
    rng = np.random.default_rng(21)
    heads, r, hd, dr = 32, 768, 128, 128
    scale = hd ** -0.5
    w = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float32)
    bf = {"q_lat": mx.array(w(heads, r, hd)).astype(mx.bfloat16), "v_up": mx.array(w(heads, hd, r)).astype(mx.bfloat16)}
    seeded, decoded = {}, {}
    for name, (o, i) in {"q_lat": (r, hd), "v_up": (hd, r)}.items():
        parts = random_seeds((heads, o, i), rng, -21)   # weights ~0.035, as the BF16 case
        mod = ns.SeedHeads(heads, i, o)
        mod.seeds, mod.coefs, mod.codes, mod.exp_bias = map(mx.array, parts)
        seeded[name] = mod
        decoded[name] = mx.array(bf16_bits(ref_decode(*parts))).view(mx.bfloat16)
    cases = [(1, 1), (1, 37), (1, 1500), (3, 300), (100, 100), (200, 700)]
    for weights, ref_w in ((bf, bf), (seeded, decoded)):
        for L, n in cases:
            q, qr, lat, kr = _mla_inputs(rng, 1, heads, L, n, r, hd, dr)
            ref = ref_mla(q, qr, lat, kr, ref_w["q_lat"], ref_w["v_up"], scale)
            for path in ("absorbed", "blocked", "expanded"):
                got = ns.mla_attention(q, qr, lat, kr, weights["q_lat"], weights["v_up"], scale,
                                       mask="causal" if L > 1 else None, path=path)
                assert got.shape == (1, heads, L, hd) and got.dtype == mx.bfloat16, (path, got.shape, got.dtype)
                err = rel_err(mx.array(ref), got)
                # BF16 intermediates (absorbed queries, expanded keys / values, BF16 score products) cost ~1%
                assert err < 0.02, (path, L, n, type(weights["q_lat"]).__name__, err)


def test_mla_attention_small_dims_and_masks():
    """Dims that are not multiples of 32 (the loader tests' small models), batch 2, the automatic choice, and an array
    mask (batched left padding: sequence 1's first keys masked)."""
    rng = np.random.default_rng(22)
    heads, r, hd, dr = 4, 48, 16, 16
    q_lat = mx.array((rng.standard_normal((heads, r, hd)) * 0.2).astype(np.float32)).astype(mx.bfloat16)
    v_up = mx.array((rng.standard_normal((heads, hd, r)) * 0.2).astype(np.float32)).astype(mx.bfloat16)
    for L, n in [(1, 9), (5, 40), (90, 90), (130, 200)]:
        q, qr, lat, kr = _mla_inputs(rng, 2, heads, L, n, r, hd, dr)
        ref = ref_mla(q, qr, lat, kr, q_lat, v_up, 0.25)
        for path in (None, "absorbed", "blocked", "expanded"):
            got = ns.mla_attention(q, qr, lat, kr, q_lat, v_up, 0.25, mask="causal" if L > 1 else None, path=path)
            err = rel_err(mx.array(ref), got)
            assert err < 0.02, (path, L, n, err)
    q, qr, lat, kr = _mla_inputs(rng, 2, heads, 6, 30, r, hd, dr)
    mask = np.tril(np.ones((6, 30), bool), 24)[None, None].repeat(2, 0)
    mask[1, :, :, :7] = False                                                         # sequence 1 left-padded by 7
    mask = mx.array(mask)
    ref = ref_mla(q, qr, lat, kr, q_lat, v_up, 0.25, mask)
    for path in (None, "absorbed", "blocked", "expanded"):
        got = ns.mla_attention(q, qr, lat, kr, q_lat, v_up, 0.25, mask=mask, path=path)
        assert rel_err(mx.array(ref), got) < 0.02, path


def test_mla_attention_many_blocks():
    """A tiny score budget: many key blocks, the causal diagonal inside a block, a strided cache view, and a padded row
    whose first key blocks are all masked."""
    rng = np.random.default_rng(24)
    heads, r, hd, dr = 8, 64, 32, 32
    q_lat = mx.array((rng.standard_normal((heads, r, hd)) * 0.15).astype(np.float32)).astype(mx.bfloat16)
    v_up = mx.array((rng.standard_normal((heads, hd, r)) * 0.15).astype(np.float32)).astype(mx.bfloat16)
    saved = ns.SCORE_BUDGET
    ns.SCORE_BUDGET = 4 * 2 * heads * 70 * 128                                       # 128-key blocks at L = 70
    try:
        q, qr, lat_buf, kr_buf = _mla_inputs(rng, 2, heads, 70, 1100, r, hd, dr)
        lat, kr = lat_buf[:, :, :1000], kr_buf[:, :, :1000]
        ref = ref_mla(q, qr, lat, kr, q_lat, v_up, 0.2)
        for path in ("blocked", "expanded"):
            got = ns.mla_attention(q, qr, lat, kr, q_lat, v_up, 0.2, mask="causal", path=path)
            assert rel_err(mx.array(ref), got) < 0.02, path
        # a prompt chunk longer than a block: the diagonal crosses blocks, early rows see none of the later blocks
        q2, qr2, lat2, kr2 = _mla_inputs(rng, 1, heads, 300, 300, r, hd, dr)
        ref = ref_mla(q2, qr2, lat2, kr2, q_lat, v_up, 0.2)
        for path in ("blocked", "expanded"):
            got = ns.mla_attention(q2, qr2, lat2, kr2, q_lat, v_up, 0.2, mask="causal", path=path)
            assert not bool(mx.any(mx.isnan(got))) and rel_err(mx.array(ref), got) < 0.02, path
        mask = np.tril(np.ones((70, 1000), bool), 930)[None, None].repeat(2, 0)
        mask[1, :, :, :300] = False
        mask = mx.array(mask)
        ref = ref_mla(q, qr, lat, kr, q_lat, v_up, 0.2, mask)
        for path in ("blocked", "expanded"):
            got = ns.mla_attention(q, qr, lat, kr, q_lat, v_up, 0.2, mask=mask, path=path)
            assert not bool(mx.any(mx.isnan(got))) and rel_err(mx.array(ref), got) < 0.02, path
    finally:
        ns.SCORE_BUDGET = saved


def test_mla_attention_memory_bounded():
    """K-1's guard: a 2,048-token prompt chunk against 16K cached tokens keeps its transient near the score budget."""
    rng = np.random.default_rng(25)
    heads, r, hd, dr, L, n = 32, 768, 128, 128, 2048, 16384
    q_lat = mx.array((rng.standard_normal((heads, r, hd)) * 0.05).astype(np.float32)).astype(mx.bfloat16)
    v_up = mx.array((rng.standard_normal((heads, hd, r)) * 0.05).astype(np.float32)).astype(mx.bfloat16)
    q, qr, lat, kr = _mla_inputs(rng, 1, heads, L, n, r, hd, dr)
    mx.eval(q, qr, lat, kr, q_lat, v_up)
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    mx.eval(ns.mla_attention(q, qr, lat, kr, q_lat, v_up, hd ** -0.5, mask="causal"))
    extra = mx.get_peak_memory() - base
    assert extra < 6 * ns.SCORE_BUDGET, extra / 1e6                                  # unbounded: 4.3 GB of scores


def _mla_cache_contents(cache):
    """The latent and RoPE key a quantized MLA cache holds, dequantized and rotated back."""
    kq, lq = cache.keys_and_values()
    deq = lambda t, bits: mx.dequantize(*t, group_size=cache.group_size, bits=bits)
    kr, lat = deq(kq, cache.rope_bits), deq(lq, cache.bits)
    if cache.rotate:
        lat = ns._rotate(lat, inverse=True)
        kr = ns._rotate(kr, inverse=True)
    return kr, lat


def test_mla_quant_cache():
    """MLAQuantCache: chunked appends match quantizing everything at once; attention through it matches the reference
    on its dequantized contents (rotation transparent), decode and prompt chunks; trim and state round trip."""
    rng = np.random.default_rng(26)
    heads, r, hd, dr = 32, 768, 128, 128
    q_lat = mx.array((rng.standard_normal((heads, r, hd)) * 0.05).astype(np.float32)).astype(mx.bfloat16)
    v_up = mx.array((rng.standard_normal((heads, hd, r)) * 0.05).astype(np.float32)).astype(mx.bfloat16)
    for bits, rope_bits, rotate in ((8, 8, False), (4, 8, True), (4, 4, True)):
        cache = ns.MLAQuantCache(bits=bits, rope_bits=rope_bits, group_size=64, rotate=rotate)
        _, _, lat, kr = _mla_inputs(rng, 1, heads, 1, 700, r, hd, dr)
        for a, b in ((0, 300), (300, 301), (301, 700)):
            got_k, got_l = cache.update_and_fetch(kr[:, :, a:b], lat[:, :, a:b])
        assert cache.offset == 700 and got_l[0].shape[2] == 700
        ckr, clat = _mla_cache_contents(cache)
        whole = ns.MLAQuantCache(bits=bits, rope_bits=rope_bits, group_size=64, rotate=rotate)
        whole.update_and_fetch(kr, lat)
        wkr, wlat = _mla_cache_contents(whole)
        assert np.array_equal(np.array(clat.astype(mx.float32)), np.array(wlat.astype(mx.float32)))
        tol = 0.02 if bits == 8 else 0.12
        assert rel_err(lat, clat) < tol and rel_err(kr, ckr) < (0.02 if rope_bits == 8 else 0.12), (bits, rope_bits)
        for L in (1, 5, 120):
            q, qr, _, _ = _mla_inputs(rng, 1, heads, L, 1, r, hd, dr)
            kq, lq = cache.keys_and_values()
            ref = ref_mla(q, qr, clat, ckr, q_lat, v_up, hd ** -0.5)
            for pv in ("qmm", "dequant", "mma"):
                ns.QUANT_PV, ns.DEQ_KEYS = pv, 256
                try:
                    got = ns.mla_attention(q, qr, lq, kq, q_lat, v_up, hd ** -0.5, mask="causal" if L > 1 else None,
                                           cache=cache)
                finally:
                    ns.QUANT_PV, ns.DEQ_KEYS = "mma", 65536
                assert got.shape == (1, heads, L, hd) and rel_err(mx.array(ref), got) < 0.02, (bits, rope_bits, L, pv)
        st = cache.state
        cache2 = ns.MLAQuantCache(bits=bits, rope_bits=rope_bits, group_size=64, rotate=rotate)
        cache2.state = st
        cache2.meta_state = cache.meta_state
        assert cache2.offset == 700 and cache2.trim(100) == 100 and cache2.offset == 600
        assert cache2.keys_and_values()[1][0].shape[2] == 600


def test_mla_quant_cache_batch():
    """BatchMLAQuantCache.merge of two caches of different lengths: masked batched attention equals each cache alone;
    extract gives the caches back; filter keeps one."""
    rng = np.random.default_rng(27)
    heads, r, hd, dr = 8, 128, 32, 64
    q_lat = mx.array((rng.standard_normal((heads, r, hd)) * 0.1).astype(np.float32)).astype(mx.bfloat16)
    v_up = mx.array((rng.standard_normal((heads, hd, r)) * 0.1).astype(np.float32)).astype(mx.bfloat16)
    singles = []
    for n in (50, 90):
        c = ns.MLAQuantCache(bits=4, rope_bits=8, group_size=64, rotate=True)
        _, _, lat, kr = _mla_inputs(rng, 1, heads, 1, n, r, hd, dr)
        c.update_and_fetch(kr, lat)
        singles.append(c)
    batch = ns.MLAQuantCache.merge(singles)
    assert type(batch).__name__ == "BatchMLAQuantCache" and batch.size() == 90
    q, qr, lat1, kr1 = _mla_inputs(rng, 2, heads, 1, 1, r, hd, dr)
    mask = batch.make_mask(1, return_array=True)                                      # before the update, as the model
    kq, lq = batch.update_and_fetch(kr1, lat1)                                         # one decode step each
    got = ns.mla_attention(q, qr, lq, kq, q_lat, v_up, hd ** -0.5, mask=mask, cache=batch)
    for i, c in enumerate(singles):
        kq1, lq1 = c.update_and_fetch(kr1[i:i + 1], lat1[i:i + 1])
        alone = ns.mla_attention(q[i:i + 1], qr[i:i + 1], lq1, kq1, q_lat, v_up, hd ** -0.5, cache=c)
        assert rel_err(alone, got[i:i + 1]) < 0.01, i
        back = batch.extract(i)
        assert type(back).__name__ == "MLAQuantCache" and back.offset == c.offset
        bk, bl = _mla_cache_contents(back)
        ck, cl = _mla_cache_contents(c)
        assert np.array_equal(np.array(bl.astype(mx.float32)), np.array(cl.astype(mx.float32)))
    batch.filter([1])
    assert batch.size() == 91 and batch.keys_and_values()[1][0].shape[0] == 1


def test_mla_make_cache():
    """MLA models default to the q8 cache; mla_kv "bf16" keeps BF16 KVCaches; a dict is used as given; a prompt then
    decode through the q8 cache tracks the BF16 cache."""
    from mlx_lm.models.cache import KVCache
    cfg = dict(small_config(), hidden_size=128, head_dim=32, mla_ranks=[64, 128], mla_rope_dim=64)
    mx.random.seed(9)
    plain = ns.Model(ns.ModelArgs.from_dict(dict(cfg, mla_kv="bf16")))
    quant = ns.Model(ns.ModelArgs.from_dict(cfg))
    given = ns.Model(ns.ModelArgs.from_dict(dict(cfg, mla_kv={"bits": 4, "rope_bits": 8, "group_size": 64, "rotate": True})))
    plain.set_dtype(mx.bfloat16)
    quant.update(plain.parameters())
    assert all(type(c) is KVCache for c in plain.make_cache())
    caches = quant.make_cache()
    assert len(caches) == 2 and all(type(c) is ns.MLAQ8Cache for c in caches)
    assert (caches[0].bits, caches[0].rope_bits, caches[0].rotate) == (8, 8, False)
    assert given.make_cache()[0].bits == 4 and given.make_cache()[0].rotate
    ids = mx.array([list(range(3, 70))])
    cp, cq = plain.make_cache(), quant.make_cache()
    a = plain(ids[:, :66], cache=cp)
    b = quant(ids[:, :66], cache=cq)
    assert rel_err(a, b) < 0.05
    a, b = plain(ids[:, 66:], cache=cp), quant(ids[:, 66:], cache=cq)
    assert rel_err(a, b) < 0.05 and cq[0].offset == 67


def test_mla_kv_choice():
    """q8 unless the requested max_context needs more memory than is available, then rq4, saying why on stdout (once);
    the cache sizes per token are the measured ones (K2-Horizon MLA: 48 layers of rank 768, RoPE key 128)."""
    import contextlib
    import io
    import os
    real = ns.ModelArgs.from_dict(dict(small_config(), num_hidden_layers=48, mla_ranks=[768] * 48, mla_rope_dim=128))
    assert ns.kv_bytes(real, "q8") == 45696 and ns.kv_bytes(real, "rq4") == 27264
    gb = 1 << 30
    assert ns.choose_kv(real, None, 4 * gb) == ("q8", None)                      # no request: q8
    assert ns.choose_kv(real, 32768, 4 * gb) == ("q8", None)                     # 1.5 GB as q8: fits
    fmt, msg = ns.choose_kv(real, 262144, 10 * gb)                              # q8 12.0 GB > 10, rq4 7.1 GB fits
    assert fmt == "rq4" and "262,144" in msg and "q8" in msg and "kv-cache.md" in msg, msg
    fmt, msg = ns.choose_kv(real, 524288, 10 * gb)                              # even rq4 (14.3 GB) does not fit
    assert fmt == "rq4" and "about" in msg and "tokens fit" in msg, msg
    cfg = dict(small_config(), hidden_size=128, head_dim=32, mla_ranks=[64, 128], mla_rope_dim=64)
    model = ns.Model(ns.ModelArgs.from_dict(cfg))
    saved = os.environ.get("NSLM_MAX_CONTEXT")
    ns._KV_LOGGED.clear()
    try:
        os.environ["NSLM_MAX_CONTEXT"] = str(10 ** 12)                          # more than any Mac holds
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            caches = model.make_cache()
            model.make_cache()
        assert caches[0].bits == 4 and caches[0].rotate
        assert out.getvalue().count("rq4") >= 1 and out.getvalue().count("NanoSeedLM: KV cache") == 1, out.getvalue()
        del os.environ["NSLM_MAX_CONTEXT"]
        assert model.make_cache(max_context=1024)[0].bits == 8
    finally:
        if saved is None:
            os.environ.pop("NSLM_MAX_CONTEXT", None)
        else:
            os.environ["NSLM_MAX_CONTEXT"] = saved


def test_mla_tune():
    """The score budget sized to the GPU working set (6 x budget within 5% of it, 64 MB .. 512 MB)."""
    assert ns.mla_tune(118e9)["score_budget"] == 512 << 20     # M4 Max 128 GB
    assert ns.mla_tune(24e9)["score_budget"] == 128 << 20      # 32 GB Macs
    assert ns.mla_tune(10e9)["score_budget"] == 64 << 20       # 16 GB Macs


def test_mla_quant_pv_mma():
    """The MMA decode P.V kernel on quantized latents (4 and 8 bits, strided cache views, several splits, 32 and 64
    rows) against dequantize (f32 scales) -> matmul."""
    rng = np.random.default_rng(29)
    for bits, n, rows in ((4, 1500, 32), (8, 777, 32), (4, 9000, 64), (8, 70, 32)):
        lat = mx.array(rng.standard_normal((1, 1, n + 100, 768)).astype(np.float32)).astype(mx.bfloat16)
        w, sc, bi = mx.quantize(lat, group_size=64, bits=bits)
        w, sc, bi = w[:, :, :n], sc[:, :, :n], bi[:, :, :n]
        p = mx.softmax(mx.array((rng.standard_normal((1, 1, rows, n)) * 3).astype(np.float32)), axis=-1).astype(mx.bfloat16)
        deq = mx.dequantize(w, sc.astype(mx.float32), bi.astype(mx.float32), group_size=64, bits=bits)
        ref = p.astype(mx.float32) @ deq
        got = ns._quant_pv_mma(p, (w, sc, bi), group_size=64, bits=bits)
        assert got.shape == ref.shape, (got.shape, ref.shape)
        assert rel_err(ref, got) < 1e-3, (bits, n, rows, rel_err(ref, got))


def test_mla_omlx_prefix_reuse():
    """Under oMLX (importable here), the q8 / rq4 caches register prefix-reuse handlers: oMLX counts them as
    reconstructible and sliceable, and serialize -> two blocks -> concatenate -> reconstruct (and oMLX's flat SSD
    element round trip) gives back the same cache, which then keeps decoding."""
    try:
        import omlx.scheduler as sched
        from omlx.cache import paged_ssd_cache as ssd
        from omlx.cache.type_registry import CacheTypeRegistry
    except ImportError:
        print("  (oMLX not importable: skipped)")
        return
    assert ns.register_omlx_prefix_reuse(), "registration failed"
    rng = np.random.default_rng(30)
    for cls in (ns.MLAQ8Cache, ns.MLARQ4Cache):
        name = cls.__name__
        assert name in CacheTypeRegistry.list_known_class_names() and name in sched._KNOWN_SLICEABLE_CACHE_TYPES
        h = CacheTypeRegistry.get_handler_by_class_name(name)
        assert h.supports_block_slicing and h.cache_type.value == name
        c = cls()
        _, _, lat, kr = _mla_inputs(rng, 1, 4, 1, 300, 128, 32, 64)
        c.update_and_fetch(kr, lat)
        elements = h.serialize_state(c)
        assert len(elements) == 6 and all(e.shape[2] == 300 for e in elements)
        state = {"states": elements, "cache_type": name}
        blocks = [h.slice_state(state, 0, 256), h.slice_state(state, 256, 300)]
        back = h.reconstruct_cache(h.concatenate_states(blocks), None)
        assert type(back) is cls and back.offset == 300
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(h.serialize_state(back), elements))
        arrays, meta = {}, {}                                              # oMLX's SSD tensor layout, no disk
        ssd._store_nstate_elements_flat(arrays, meta, "layer_0", elements)
        loaded = ssd._load_nstate_flat(arrays, meta, "layer_0", name)
        assert loaded is not None and all(bool(mx.array_equal(a, b)) for a, b in zip(loaded[2], elements))
        _, _, lat2, kr2 = _mla_inputs(rng, 1, 4, 1, 3, 128, 32, 64)
        back.update_and_fetch(kr2, lat2)                                   # keeps going after a restore
        assert back.offset == 303
        merged = cls.merge([back])
        assert type(merged.extract(0)) is cls and merged.extract(0).offset == 303


def test_mla_fused_prefill():
    """The fused prompt kernel (_MLA_PREFILL) against the float32 reference and the unfused expanded path: causal and
    unmasked, several key blocks, batch 2, ragged query tiles, the real head shapes and small ones, both quantized
    caches."""
    rng = np.random.default_rng(25)
    g = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)
    saved = ns.FUSED_PREFILL, ns.FUSED_KEYS
    try:
        for heads, r, hd, dr in [(32, 768, 128, 128), (8, 64, 32, 32)]:
            q_lat, v_up = g(heads, r, hd) * 0.05, g(heads, hd, r) * 0.05
            for b, L, n, keys in [(1, 64, 64, None), (1, 100, 100, None), (2, 70, 333, 128), (1, 257, 1000, 256),
                                  (1, 64, 2000, 512)]:
                q, qr, lat, kr = _mla_inputs(rng, b, heads, L, n, r, hd, dr)
                for mask in ("causal", None):
                    full = None if mask else mx.array(np.ones((b, 1, L, n), bool))
                    ref = mx.array(ref_mla(q, qr, lat, kr, q_lat, v_up, hd ** -0.5, full))
                    ns.FUSED_PREFILL, ns.FUSED_KEYS = False, None
                    old = ns.mla_attention(q, qr, lat, kr, q_lat, v_up, hd ** -0.5, mask=mask, path="expanded")
                    ns._kernels.pop("mla_prefill", None)
                    ns.FUSED_PREFILL, ns.FUSED_KEYS = True, keys
                    got = ns.mla_attention(q, qr, lat, kr, q_lat, v_up, hd ** -0.5, mask=mask, path="expanded")
                    assert "mla_prefill" in ns._kernels, "fused kernel not used"
                    assert got.shape == (b, heads, L, hd) and got.dtype == mx.bfloat16, got.shape
                    err, err_old = rel_err(ref, got), rel_err(ref, old)
                    # f32 P.V: at most the unfused path's BF16 error
                    assert err < 0.01 and err <= err_old * 1.05 + 1e-3, (heads, b, L, n, keys, mask, err, err_old)
        heads, r, hd, dr = 32, 768, 128, 128
        q_lat, v_up = g(heads, r, hd) * 0.05, g(heads, hd, r) * 0.05
        for cls in (ns.MLAQ8Cache, ns.MLARQ4Cache):
            cache = cls()
            kq, lq = cache.update_and_fetch(g(1, 1, 600, dr), g(1, 1, 600, r))
            deq = lambda t, bits: mx.dequantize(*t, group_size=64, bits=bits)
            kr, lat = deq(kq, cache.rope_bits), deq(lq, cache.bits)
            if cache.rotate:
                kr, lat = ns._rotate(kr, inverse=True), ns._rotate(lat, inverse=True)
            q, qr = g(1, heads, 100, hd), g(1, heads, 100, dr)
            ref = mx.array(ref_mla(q, qr, lat, kr, q_lat, v_up, hd ** -0.5))
            ns._kernels.pop("mla_prefill", None)
            ns.FUSED_PREFILL, ns.FUSED_KEYS = True, 256
            got = ns.mla_attention(q, qr, lq, kq, q_lat, v_up, hd ** -0.5, mask="causal", cache=cache)
            assert "mla_prefill" in ns._kernels, cls.__name__
            assert rel_err(ref, got) < 0.01, cls.__name__
    finally:
        ns.FUSED_PREFILL, ns.FUSED_KEYS = saved

def test_load_hybrid():
    """A hybrid model (mla_ranks 0: the layer keeps GQA attention): the loader keeps k_proj / v_proj there, the cache
    is a KVCache there and an MLA cache elsewhere, kv_bytes counts BF16 K and V for it, and prompt + decode through
    the cache match the full forward."""
    from mlx_lm.utils import load_model

    config = dict(small_config(), mla_ranks=[16, 0], mla_rope_dim=16, mla_kv="bf16")
    mx.random.seed(6)
    ref = ns.Model(ns.ModelArgs.from_dict(config))
    rng = np.random.default_rng(14)
    m = ref.model.layers[0].self_attn.mla
    for name, value in m.items():
        setattr(m, name, mx.array((rng.standard_normal(value.shape) * 0.05).astype(np.float32)))
    ref.set_dtype(mx.bfloat16)
    weights = {name.replace("mlp.expert_bias", "mlp.gate.bias").replace("self_attn.v_expert_bias", "self_attn.v_router.bias"):
               value for name, value in tree_flatten(ref.parameters())}
    with tempfile.TemporaryDirectory() as d:
        mx.save_safetensors(f"{d}/model.safetensors", weights, metadata={"format": "mlx"})
        Path(d, "config.json").write_text(json.dumps(dict(config, model_file="nanoseedlm_k2.py")))
        shutil.copy(TOOLS / "nanoseedlm_k2.py", d)
        model, _ = load_model(Path(d), trust_remote_code=True)
    att0, att1 = model.model.layers[0].self_attn, model.model.layers[1].self_attn
    assert type(att0).__name__ == "MLAAttention" and type(att1).__name__ != "MLAAttention" and "k_proj" in att1
    args = ns.ModelArgs.from_dict(config)
    gqa = 2 * args.num_key_value_heads * args.head_dim * 2
    assert ns.kv_bytes(args, "q8") == ns.kv_bytes(ns.ModelArgs.from_dict(dict(config, mla_ranks=[16])), "q8") + gqa
    q8 = ns.mla_caches(ns.ModelArgs.from_dict(dict(config, mla_ranks=[768, 0], mla_kv=dict(ns.KV_FORMATS["q8"]))), 2)
    assert [type(c).__name__ for c in q8] == ["MLAQ8Cache", "KVCache"]
    cache = model.make_cache()
    ids = mx.array([list(range(1, 41))])
    full = model(ids)
    assert rel_err(ref(ids), full) < 0.02
    first = model(ids[:, :39], cache=cache)
    last = model(ids[:, 39:], cache=cache)
    assert rel_err(full[:, :39], first) < 0.02 and rel_err(full[:, 39:], last) < 0.02

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
