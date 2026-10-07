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
        if "mlp.experts." in name:
            parts = random_seeds(value.shape, rng)
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
        layer = SwitchLinear(k * 8, n, e, bias=False)
        layer.weight = mx.array(bf16_bits(ref_decode(*parts))).view(mx.bfloat16)
        ref.update_modules(tree_unflatten([(path, layer)]))
    assert type(model.model.layers[1].mlp.experts).__name__ == "SeedSwitchGLU"   # model_file is imported as its own module
    assert type(model.model.layers[1].mlp).__name__ == "SeedSparseMoeBlock"
    assert model.model.layers[1].self_attn.v_experts.bits == 4
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    for ids in (mx.array([[5]]), mx.array([list(range(1, 41))])):
        err = rel_err(ref(ids), model(ids))
        assert err < 0.02, err


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    print("test_nanoseedlm_k2: PASS")
