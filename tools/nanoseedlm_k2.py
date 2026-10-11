# SPDX-License-Identifier: MIT
"""NanoSeedLM for MLX: K2-Horizon with SeedLM weights, decoded on the GPU; GQA or calibrated TransMLA attention.

mlx-lm loads this file through config.json "model_file" (trust_remote_code). It needs the K2-Horizon model
(mlx_lm.models.k2_horizon, provided by oMLX).

Block: 8 weights from a 16-bit LFSR seed (K=16, taps 0,1,3,12), P int4 coefficients and a 4-bit exponent code (P = 3,
4 or 8). With S[c][p] = state(Pc+p+1) - 32768: w_c = bf16(R32 * 2^(exp_bias+code) * sum_p S[c][p] * q_p), R32 = fl(1/32767).

MLA (config mla_ranks): per layer the cache holds the latent c (rank r_l) and one 128-dim RoPE key per token instead of
8 x 128 keys and values:
  c        = kv_a_x x (+ kv_a_v v, v from the value experts; layers 0-2 fold v_proj into kv_a_x)
  k_rope   = RoPE(k_rope_proj x)
  scores_a = (q_lat_a q_a) . c + RoPE(q_rope_mix_a q_a) . k_rope,  scale 128^-0.5 as the original
  o_a      = v_up_a (sum_t p_t c_t), then the softplus gate and o_proj.
A layer with mla_ranks[i] == 0 keeps its original (GQA) attention and a BF16 KVCache (a hybrid model: MLA where the
latent holds the layer, GQA where it does not).
"""

from dataclasses import dataclass, field
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map, tree_unflatten
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, _BaseCache, create_attention_mask, dynamic_roll
from mlx_lm.models.switch_layers import SwitchGLU

try:
    from mlx_lm.models import k2_horizon as _k2
except ImportError as error:
    raise ImportError("NanoSeedLM K2-Horizon models need mlx_lm.models.k2_horizon (oMLX provides it)") from error

# (token, expert) pairs at or below this use the fused mat-vec; above it, the pairs are sorted by expert and go
# through the seeded tile GEMM (weight tiles rebuilt in threadgroup memory; no BF16 expert tensor).
MATVEC_MAX_PAIRS = 32
# GEMM tile: BM weight rows x BN pairs, BK inputs per step.
BM, BN, BK = 64, 64, 16

_HEADER = """
constant float R32 = as_type<float>(0x38000100u);
inline float block_dot(uint g, const device uint16_t* seeds, const device uint16_t* coefs, const device uint8_t* codes,
                       const device uint32_t* table, thread const float* xv, int bias, thread float* w) {
    // isum_c / 2^16 exactly in f32: states as 1 + V / 2^16 from 32-bit stream words (two ops each),
    // sum_p q_p v_p - 1.5 sum_p q_p (every term and partial sum is a multiple of 2^-16 below 2^6)
    uint s = seeds[g];
    uint q = coefs[g];
    int code = (codes[g >> 1] >> ((g & 1) * 4)) & 15;
    float sc = ldexp(R32, bias + code + 16);
    uint t = table[s], lo = s | (t << 16);
    float q0 = float(int(q << 28) >> 28), q1 = float(int(q << 24) >> 28), q2 = float(int(q << 20) >> 28), q3 = float(int(q << 16) >> 28);
    float qs = 1.5f * (((q0 + q1) + q2) + q3);
    float d = 0.0f;
    for (uint c = 0; c < 8; ++c) {
        float v[4];
        for (uint p = 0; p < 4; ++p) {
            uint k = 4 * c + p + 1;
            v[p] = as_type<float>(insert_bits(0x3F800000u, k <= 16 ? lo >> k : t >> (k - 16), 7u, 16u));
        }
        float isum = (((v[0] * q0 + v[1] * q1) + v[2] * q2) + v[3] * q3) - qs;
        if (w) w[c] = isum * sc;
        if (xv) d += isum * xv[c];
    }
    return d * sc;
}
// P = 8 (32-bit coefficient words, 64 states; states 33..64 are states 1..32 of s2 = state 32 = table[s] >> 16):
// isum_c / 2^16 = sum_p q_p v_p - 1.5 sum_p q_p, exact (multiples of 2^-16 below 2^7).
inline float block_dot(uint g, const device uint16_t* seeds, const device uint32_t* coefs, const device uint8_t* codes,
                       const device uint32_t* table, thread const float* xv, int bias, thread float* w) {
    uint s = seeds[g], cw = coefs[g];
    int code = (codes[g >> 1] >> ((g & 1) * 4)) & 15;
    float sc = ldexp(R32, bias + code + 16);
    uint t = table[s], lo = s | (t << 16), s2 = t >> 16, t2 = table[s2], lo2 = s2 | (t2 << 16);
    float q[8], qs = 0.0f;
    for (uint p = 0; p < 8; ++p) { q[p] = float(int(cw << (28 - 4 * p)) >> 28); qs += q[p]; }
    qs *= 1.5f;
    float d = 0.0f;
    for (uint c = 0; c < 8; ++c) {
        float isum = 0.0f;
        for (uint p = 0; p < 8; ++p) {
            uint k = 8 * c + p + 1;
            uint src = k <= 16 ? lo >> k : k <= 32 ? t >> (k - 16) : k <= 48 ? lo2 >> (k - 32) : t2 >> (k - 48);
            isum += as_type<float>(insert_bits(0x3F800000u, src, 7u, 16u)) * q[p];
        }
        isum -= qs;
        if (w) w[c] = isum * sc;
        if (xv) d += isum * xv[c];
    }
    return d * sc;
}
// P = 3 (SEED4: 16-bit nibble words, the exponent code in bits 0-3, q_p in bits 4p+4 .. 4p+7; read as int16 to pick
// this overload, codes unused; MLX may pass the 1-element placeholder as constant): states 1..24, exact as above.
template <typename C>
inline float block_dot(uint g, const device uint16_t* seeds, const device int16_t* coefs, C codes,
                       const device uint32_t* table, thread const float* xv, int bias, thread float* w) {
    uint s = seeds[g], nw = uint(ushort(coefs[g]));
    float sc = ldexp(R32, bias + int(nw & 15u) + 16);
    uint t = table[s], lo = s | (t << 16);
    float q0 = float(int(nw << 24) >> 28), q1 = float(int(nw << 20) >> 28), q2 = float(int(nw << 16) >> 28);
    float qs = 1.5f * ((q0 + q1) + q2);
    float d = 0.0f;
    for (uint c = 0; c < 8; ++c) {
        float v[3];
        for (uint p = 0; p < 3; ++p) {
            uint k = 3 * c + p + 1;
            v[p] = as_type<float>(insert_bits(0x3F800000u, k <= 16 ? lo >> k : t >> (k - 16), 7u, 16u));
        }
        float isum = ((v[0] * q0 + v[1] * q1) + v[2] * q2) - qs;
        if (w) w[c] = isum * sc;
        if (xv) d += isum * xv[c];
    }
    return d * sc;
}
template <typename U> inline float round_to(float x) { return float(static_cast<U>(x)); }
template <> inline float round_to<bfloat16_t>(float x) {   // nearest even, in integer ops (kept by the compiler)
    uint u = as_type<uint>(x);
    return as_type<float>((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}
"""

_DECODE = """
uint g = thread_position_in_grid.x;
if (g >= BLOCKS) return;
float w[8];
block_dot(g, seeds, coefs, codes, table, nullptr, exp_bias[g / (N * KB)], w);
for (uint c = 0; c < 8; ++c) out[g * 8 + c] = static_cast<bfloat16_t>(w[c]);
"""

_MATVEC = """
uint lane = thread_position_in_threadgroup.x;
uint r0 = thread_position_in_grid.y * 4;
uint m = thread_position_in_grid.z;
uint e = indices[m];
int bias = exp_bias[e];
const device T* xm = x + m * (KB * 8);
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint b = lane; b < KB; b += 32) {
    float xv[8];
    for (uint c = 0; c < 8; ++c) xv[c] = float(xm[b * 8 + c]);
    for (uint r = 0; r < 4; ++r)
        if (r0 + r < N) acc[r] += block_dot((e * N + r0 + r) * KB + b, seeds, coefs, codes, table, xv, bias, nullptr);
}
for (uint r = 0; r < 4; ++r) {
    float v = simd_sum(acc[r]);
    if (lane == 0 && r0 + r < N) out[m * N + r0 + r] = static_cast<T>(v);
}
"""

# One threadgroup per (BM-row block, tile of one expert's sorted pairs). tiles[e] / starts[e]: the expert's first
# tile / pair (cumulative; entry E is the total). The weight tile is decode's BF16, the multiply f32 simdgroup MMA
# (4 simdgroups of 32 rows x 32 pairs), stored from the accumulators.
_GEMM = """
const uint tid = thread_index_in_threadgroup, sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
const int t = int(threadgroup_position_in_grid.y);
if (t >= tiles[E]) return;
int lo = 0, hi = E;
while (hi - lo > 1) { int mid = (lo + hi) / 2; if (tiles[mid] <= t) lo = mid; else hi = mid; }
const uint e = lo;
const int p0 = starts[e] + (t - tiles[e]) * BN, count = min(BN, starts[e + 1] - p0);
const int r0 = int(threadgroup_position_in_grid.x) * BM;
const int bias = exp_bias[e];
constexpr int TH = 128, KP = BK + 4;
threadgroup float Wt[BM * KP];   // [row][k]
threadgroup float Xt[BN * KP];   // [pair][k]
simdgroup_float8x8 acc[4][4];
_Pragma("clang loop unroll(full)") for (int i = 0; i < 4; ++i)
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
const int sr = int(sgi / 2) * 32, sn = int(sgi % 2) * 32;
for (int k0 = 0; k0 < KB * 8; k0 += BK) {
    for (int b = int(tid); b < BM * BK / 8; b += TH) {   // weights: one block per thread
        const int rr = b / (BK / 8), jb = b % (BK / 8);
        float w[8];
        if (r0 + rr < N) {
            block_dot((e * N + r0 + rr) * KB + k0 / 8 + jb, seeds, coefs, codes, table, nullptr, bias, w);
            for (int c = 0; c < 8; ++c) w[c] = float(static_cast<bfloat16_t>(w[c]));
        } else for (int c = 0; c < 8; ++c) w[c] = 0.0f;
        for (int c = 0; c < 8; ++c) Wt[rr * KP + jb * 8 + c] = w[c];
    }
    for (int b = int(tid); b < BN * BK / 8; b += TH) {   // inputs: 8 per thread
        const int n = b / (BK / 8), kk = (b % (BK / 8)) * 8;
        const device T* xr = x + (p0 + n) * (KB * 8) + k0 + kk;
        for (int c = 0; c < 8; ++c) Xt[n * KP + kk + c] = n < count ? float(xr[c]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)") for (int kb = 0; kb < BK; kb += 8) {
        simdgroup_float8x8 wa[4], xb[4];
        _Pragma("clang loop unroll(full)") for (int i = 0; i < 4; ++i) simdgroup_load(wa[i], Wt + (sr + 8 * i) * KP + kb, KP);
        _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) simdgroup_load(xb[j], Xt + (sn + 8 * j) * KP + kb, KP, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)") for (int i = 0; i < 4; ++i)
            _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], wa[i], xb[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const short qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
_Pragma("clang loop unroll(full)") for (int i = 0; i < 4; ++i) {
    const int r = r0 + sr + 8 * i + fm;
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) {
        thread auto& v = acc[i][j].thread_elements();
        const int n = sn + 8 * j + fn;
        if (r < N && n < count) out[(p0 + n) * N + r] = static_cast<T>(v[0]);
        if (r < N && n + 1 < count) out[(p0 + n + 1) * N + r] = static_cast<T>(v[1]);
    }
}
"""

# out[t] = sum_j T(y[inverse[t K + j]] * T(w[t][j])): K2's routed (y * w).sum(-2) on the expert-sorted rows, rounded as
# mlx does it (products in T, then a sequential sum in T), 4 columns per thread. The rounding to T is explicit
# (round_to): the compiler may keep a loop-carried bfloat in f32.
_COMBINE = """
const uint d = thread_position_in_grid.x * 4, t = thread_position_in_grid.y;
if (d >= D) return;
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint j = 0; j < K; ++j) {
    const device T* yr = y + inverse[t * K + j] * D + d;
    const float w = round_to<T>(float(weights[t * K + j]));
    for (uint c = 0; c < 4; ++c) acc[c] = round_to<T>(round_to<T>(float(yr[c]) * w) + acc[c]);
}
for (uint c = 0; c < 4; ++c) out[t * D + d + c] = static_cast<T>(acc[c]);
"""

# One threadgroup: starts[e] = first pair of expert e in the sorted indices (binary search), tiles = the cumulative
# BN-pair tile counts.
_TILES = """
const uint e = thread_position_in_grid.x;
threadgroup int st[E + 1];
if (e <= E) {
    int lo = 0, hi = P;
    while (lo < hi) { const int mid = (lo + hi) / 2; if (indices[mid] < e) lo = mid + 1; else hi = mid; }
    st[e] = lo;
    starts[e] = lo;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (e == 0) {
    int t = 0;
    tiles[0] = 0;
    for (int i = 0; i < E; ++i) { t += (st[i + 1] - st[i] + BN - 1) / BN; tiles[i + 1] = t; }
}
"""

# Embedding rows: one thread per (token, block), decode's BF16.
_EMBED = """
uint b = thread_position_in_grid.x, t = thread_position_in_grid.y;
if (b >= KB) return;
float w[8];
block_dot(uint(ids[t]) * KB + b, seeds, coefs, codes, table, nullptr, exp_bias[0], w);
for (uint c = 0; c < 8; ++c) out[(t * KB + b) * 8 + c] = static_cast<bfloat16_t>(w[c]);
"""

_SEED_INPUTS = ["seeds", "coefs", "codes", "exp_bias", "table"]
_kernels: dict = {}
_table = None


def lfsr_table() -> mx.array:
    """The 32 stream bits after each seed: bit k = bit 15 of state k+1."""
    global _table
    if _table is None:
        s = np.arange(65536, dtype=np.uint32)
        g = np.zeros_like(s)
        for k in range(32):
            b = (s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1
            s = (s >> 1) | (b << 15)
            g |= b << k
        _table = mx.array(g)
    return _table


def _kernel(name: str, inputs: list[str], source: str):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(
            name=f"nanoseedlm_p4_{name}", input_names=inputs, output_names=["out"], source=source, header=_HEADER
        )
    return _kernels[name]


def decode(seeds, coefs, codes, exp_bias) -> mx.array:
    """All blocks to BF16, shape (experts, rows, cols) (the reference for the tests; inference never builds it)."""
    e, n, kb = seeds.shape
    kernel = _kernel("decode", _SEED_INPUTS, _DECODE)
    return kernel(
        inputs=[seeds, coefs, codes, exp_bias, lfsr_table()],
        template=[("BLOCKS", e * n * kb), ("N", n), ("KB", kb)],
        grid=(e * n * kb, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(e, n, kb * 8)],
        output_dtypes=[mx.bfloat16],
    )[0]


def gather_matvec(x, indices, seeds, coefs, codes, exp_bias) -> mx.array:
    """out[m] = W[indices[m]] @ x[m] for x (M, cols), indices (M,)."""
    _, n, kb = seeds.shape
    m = x.shape[0]
    kernel = _kernel("matvec", ["x", "indices"] + _SEED_INPUTS, _MATVEC)
    return kernel(
        inputs=[x, indices, seeds, coefs, codes, exp_bias, lfsr_table()],
        template=[("T", x.dtype), ("N", n), ("KB", kb)],
        grid=(32, (n + 3) // 4, m),
        threadgroup=(32, 8, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def expert_tiles(sorted_indices, experts: int):
    """(tiles, starts): each expert's first BN-pair tile and first pair in the sorted pairs, cumulative (E + 1,)."""
    if experts > 1023:
        raise ValueError("expert_tiles: one threadgroup, at most 1023 experts")
    if "tiles" not in _kernels:
        _kernels["tiles"] = mx.fast.metal_kernel(name="nanoseedlm_p4_tiles", input_names=["indices"],
                                                 output_names=["tiles", "starts"], source=_TILES)
    width = (experts + 32) // 32 * 32
    return _kernels["tiles"](
        inputs=[sorted_indices],
        template=[("E", experts), ("P", sorted_indices.size), ("BN", BN)],
        grid=(width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(experts + 1,), (experts + 1,)],
        output_dtypes=[mx.int32, mx.int32],
    )


def gather_gemm(x, sorted_indices, seeds, coefs, codes, exp_bias, tiles=None) -> mx.array:
    """out[m] = W[sorted_indices[m]] @ x[m] for x (M, cols) and expert indices (M,) in ascending order."""
    e, n, kb = seeds.shape
    m = x.shape[0]
    if kb * 8 % BK:
        raise ValueError(f"the seeded GEMM needs input_dims divisible by {BK}")
    tiles, starts = expert_tiles(sorted_indices, e) if tiles is None else tiles
    kernel = _kernel("gemm", ["x", "tiles", "starts"] + _SEED_INPUTS, _GEMM)
    return kernel(
        inputs=[x, tiles, starts, seeds, coefs, codes, exp_bias, lfsr_table()],
        template=[("T", x.dtype), ("E", e), ("N", n), ("KB", kb), ("BM", BM), ("BN", BN), ("BK", BK)],
        grid=((n + BM - 1) // BM * 128, (m + BN - 1) // BN + e, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def combine_sorted(ys, order, weights) -> mx.array:
    """(y * weights[..., None]).sum(-2) for y = the expert-sorted rows ys (P, D) back in token order, without
    building y: weights (..., k), order = the sort permutation of the flattened (token, k) pairs."""
    k, (p, d) = weights.shape[-1], ys.shape
    if d % 4:
        raise ValueError("combine_sorted needs a hidden size divisible by 4")
    inverse = mx.zeros((p,), mx.uint32)
    inverse[order] = mx.arange(p, dtype=mx.uint32)
    kernel = _kernel("combine", ["y", "inverse", "weights"], _COMBINE)
    return kernel(
        inputs=[ys, inverse, weights.reshape(-1, k)],
        template=[("T", ys.dtype), ("D", d), ("K", k)],
        grid=(d // 4, p // k, 1),
        threadgroup=(min(256, d // 4), 1, 1),
        output_shapes=[(p // k, d)],
        output_dtypes=[ys.dtype],
    )[0].reshape(weights.shape[:-1] + (d,))


def embed_rows(ids, seeds, coefs, codes, exp_bias) -> mx.array:
    """Rows ids of a seeded [1, vocab, dims] table as BF16, shape ids.shape + (dims,)."""
    kb = seeds.shape[-1]
    flat = ids.reshape(-1).astype(mx.int32)
    kernel = _kernel("embed", ["ids"] + _SEED_INPUTS, _EMBED)
    out = kernel(
        inputs=[flat, seeds, coefs, codes, exp_bias, lfsr_table()],
        template=[("KB", kb)],
        grid=(kb, flat.size, 1),
        threadgroup=(min(kb, 256), 1, 1),
        output_shapes=[(flat.size, kb * 8)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out.reshape(ids.shape + (kb * 8,))


def _sort_pairs(x, indices):
    """x (..., cols) per token and indices (..., k) -> x per (token, expert) pair sorted by expert, the sorted indices
    and the order."""
    idx = indices.reshape(-1).astype(mx.uint32)
    order = mx.argsort(idx)
    return x.reshape(-1, x.shape[-1])[order // indices.shape[-1]], idx[order], order


def _unsort(y, order, shape):
    inverse = mx.zeros(order.shape, mx.uint32)
    inverse[order] = mx.arange(order.size, dtype=mx.uint32)
    return y[inverse].reshape(shape + y.shape[-1:])


def _seed_params(m: nn.Module, slices: int, rows: int, cols: int, p: int):
    """A module's seed streams: P = 4 16-bit coefficient words, P = 8 32-bit, P = 3 the SEED4 nibble words as int16
    (exponent codes inside; codes is a placeholder)."""
    if cols % BK:
        raise ValueError(f"SeedLM in MLX needs input_dims divisible by {BK}")
    m.seeds = mx.zeros((slices, rows, cols // 8), mx.uint16)
    m.coefs = mx.zeros((slices, rows, cols // 8), {3: mx.int16, 4: mx.uint16, 8: mx.uint32}[p])
    m.codes = mx.zeros((1,) if p == 3 else (slices, rows, cols // 16), mx.uint8)
    m.exp_bias = mx.zeros((slices,), mx.int32)


class SeedSwitchLinear(nn.Module):
    """mlx-lm's SwitchLinear with SeedLM P=4 (or P=3) weights."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int, p: int = 4):
        super().__init__()
        _seed_params(self, num_experts, output_dims, input_dims, p)
        self.input_dims, self.output_dims = input_dims, output_dims

    def _seeds(self):
        return self.seeds, self.coefs, self.codes, self.exp_bias

    def gemm(self, xs, sorted_indices, tiles=None):
        return gather_gemm(xs, sorted_indices, *self._seeds(), tiles)

    def matvec(self, x, indices):
        rows = x.shape[-2]
        batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
        xs = mx.broadcast_to(x, batch + x.shape[-2:]).reshape(-1, self.input_dims)
        idx = mx.broadcast_to(indices, batch).reshape(-1).astype(mx.uint32)
        if rows > 1:
            idx = mx.repeat(idx, rows)
        return gather_matvec(xs, idx, *self._seeds()).reshape(batch + (rows, self.output_dims))

    def __call__(self, x, indices):
        if indices.size * x.shape[-2] <= MATVEC_MAX_PAIRS:
            return self.matvec(x, indices)
        if x.shape[-2] != 1:
            raise ValueError("SeedSwitchLinear: one row per token above the mat-vec limit")
        xs, idx, order = _sort_pairs(x, indices)
        return _unsort(self.gemm(xs, idx), order, indices.shape).reshape(indices.shape + (1, self.output_dims))


class SeedLinear(nn.Module):
    """nn.Linear (no bias) with SeedLM P=3 / 4 / 8 weights: a one-expert SeedSwitchLinear."""

    def __init__(self, input_dims: int, output_dims: int, p: int = 4):
        super().__init__()
        _seed_params(self, 1, output_dims, input_dims, p)
        self.input_dims, self.output_dims = input_dims, output_dims

    def __call__(self, x):
        xs = x.reshape(-1, self.input_dims)
        idx = mx.zeros((xs.shape[0],), mx.uint32)
        parts = (self.seeds, self.coefs, self.codes, self.exp_bias)
        y = gather_matvec(xs, idx, *parts) if xs.shape[0] <= MATVEC_MAX_PAIRS else gather_gemm(xs, idx, *parts)
        return y.reshape(x.shape[:-1] + (self.output_dims,))


class SeedEmbedding(nn.Module):
    """nn.Embedding with SeedLM P=3 / 4 / 8 rows, decoded per token."""

    def __init__(self, num_embeddings: int, dims: int, p: int = 4):
        super().__init__()
        _seed_params(self, 1, num_embeddings, dims, p)

    def __call__(self, ids):
        return embed_rows(ids, self.seeds, self.coefs, self.codes, self.exp_bias)


class SeedHeads(nn.Module):
    """Per-head maps y[b, a, l] = W_a x[b, a, l] (MLA's q_lat, q_rope_mix, v_up) with seed weights: the heads are the
    experts, the (head, token) pairs head-major (sorted)."""

    def __init__(self, heads: int, input_dims: int, output_dims: int, p: int = 4):
        super().__init__()
        _seed_params(self, heads, output_dims, input_dims, p)
        self.input_dims, self.output_dims = input_dims, output_dims

    def __call__(self, x):
        b, a, l, k = x.shape
        xs = x.transpose(1, 0, 2, 3).reshape(-1, k)
        idx = mx.repeat(mx.arange(a, dtype=mx.uint32), b * l)
        parts = (self.seeds, self.coefs, self.codes, self.exp_bias)
        y = gather_matvec(xs, idx, *parts) if xs.shape[0] <= MATVEC_MAX_PAIRS else gather_gemm(xs, idx, *parts)
        return y.reshape(a, b, l, self.output_dims).transpose(1, 0, 2, 3)


class SeedSwitchGLU(SwitchGLU):
    """SwitchGLU on seed projections: the fused mat-vec for few pairs, else expert-sorted pairs through the seeded GEMM."""

    def _matvec(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        x_up = self.up_proj.matvec(x, indices)
        x_gate = self.gate_proj.matvec(x, indices)
        return self.down_proj.matvec(self.activation(x_up, x_gate), indices).squeeze(-2)

    def _sorted(self, x, indices):
        """The down projection's output per (token, expert) pair, sorted by expert, and the sort order."""
        xs, idx, order = _sort_pairs(x, indices)
        tiles = expert_tiles(idx, self.up_proj.seeds.shape[0])
        h = self.activation(self.up_proj.gemm(xs, idx, tiles), self.gate_proj.gemm(xs, idx, tiles))
        return self.down_proj.gemm(h, idx, tiles), order

    def __call__(self, x, indices) -> mx.array:
        if indices.size <= MATVEC_MAX_PAIRS:
            return self._matvec(x, indices)
        ys, order = self._sorted(x, indices)
        return _unsort(ys, order, indices.shape)

    def weighted(self, x, indices, weights) -> mx.array:
        """(self(x, indices) * weights[..., None]).sum(-2); above the mat-vec limit without the [..., k, hidden] tensor."""
        if indices.size <= MATVEC_MAX_PAIRS:
            return (self._matvec(x, indices) * weights.astype(x.dtype)[..., None]).sum(axis=-2)
        return combine_sorted(*self._sorted(x, indices), weights)


class SeedSparseMoeBlock(getattr(_k2, "SparseMoeBlock", nn.Module)):
    """K2's SparseMoeBlock whose routed sum comes from SeedSwitchGLU.weighted. __call__ mirrors oMLX's
    (k2_horizon_model.py, main 2026-10-07) with only the routed line changed; keep the two in step."""

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = _k2.route(x, self.gate.weight, self.expert_bias, self.top_k, self.scaling_factor,
                                  partitions=self.router_partitions)
        ane = getattr(self.shared_experts, "_omlx_ane_prefill", None)
        prepared = ane.prepare(x) if ane is not None and ane.active else None
        routed = self.experts.weighted(x, inds, weights)
        shared = self.shared_experts(x) if prepared is None else ane.finish(prepared, alongside=routed)
        return routed + shared


@dataclass
class ModelArgs(_k2.ModelArgs):
    mla_ranks: list = field(default_factory=list)   # MLA: the latent rank per layer (empty: GQA)
    mla_rope_dim: int = 128
    mla_kv: object = None                            # MLA cache: None / "auto" (q8, or rq4 for a long max_context),
                                                     # "bf16", or MLAQuantCache settings (bits, rope_bits, group_size, rotate)
    mla_max_context: int = 0                         # MLA: the longest context to plan the cache for (0: not set)


def _lin(w, x):
    """x W^T for an MLA projection: a BF16 array [out, in] or a SeedLinear."""
    return w(x) if isinstance(w, nn.Module) else x @ w.T


def _heads(w, x):
    """W_a x[b, a, l] for a per-head map: a BF16 array [heads, out, in] or SeedHeads."""
    return w(x) if isinstance(w, nn.Module) else mx.einsum("aoi,bali->balo", w, x)


def _dense_heads(w):
    """A per-head map as a BF16 array [heads, out, in]; SeedHeads decoded (transient, once per prompt chunk)."""
    return decode(w.seeds, w.coefs, w.codes, w.exp_bias) if isinstance(w, nn.Module) else w


# MLA attention, in key blocks with an online softmax (running max, sum and output in f32), so the transient is one
# block's scores (SCORE_BUDGET bytes), whatever the context. Decode and short chunks: the absorbed form, the queries
# q_lat_a q_a against the cached latent c and RoPE key in place (two score products, no concatenated keys). Prompt chunks
# of EXPAND_MIN_L tokens or more: the expanded form, per block k_a = q_lat_a^T c (128) beside the shared RoPE key and
# v_a = v_up_a c: 256-dim instead of 896-dim dot products, ~4x fewer FLOPs.
EXPAND_MIN_L = 64
SCORE_BUDGET = 128 << 20        # bytes of f32 scores per key block (transient ~5.3x: scores, softmax, BF16 copy)


def _block_keys(b, heads, L, n):
    nb = SCORE_BUDGET // (4 * b * heads * L)
    return n if nb >= n else max(128, nb // 128 * 128)


def _mla_blocked(q, qr, lat, kr, q_lat, v_up, scale, mask, expanded, cache=None):
    """softmax over key blocks merged by log-sum-exp; mask None (one query or none), "causal" (bottom right) or an array
    broadcastable to [B, H, L, N] (bool or additive). Returns the per-head outputs [B, H, L, hd] (v_up applied)."""
    b, heads, L, hd = q.shape
    quant = isinstance(lat, (tuple, list))
    n = lat[0].shape[2] if quant else lat.shape[2]
    nb = _block_keys(b, heads, L, n)
    if quant:
        return _mla_blocked_quant(q, qr, lat, kr, q_lat, v_up, scale, mask, expanded, cache, nb)
    if not expanded and nb >= n:
        # one block (decode): as few kernels as MLX's own attention, scale and sum inside one fused mat-mul
        qa = _heads(q_lat, q)
        s = mx.addmm(qr.astype(q.dtype) @ kr.swapaxes(-1, -2), qa, lat.swapaxes(-1, -2), alpha=scale, beta=scale)
        if isinstance(mask, str):
            if L > 1:
                s = mx.where(mx.arange(n)[None] <= mx.arange(n - L, n)[:, None], s, -mx.inf)
        elif mask is not None:
            s = mx.where(mask, s, -mx.inf) if mask.dtype == mx.bool_ else s + mask.astype(s.dtype)
        return _heads(v_up, mx.softmax(s, axis=-1, precise=True) @ lat)
    if (expanded and FUSED_PREFILL and (mask is None or isinstance(mask, str)) and q.dtype in (mx.bfloat16, mx.float16)
            and hd % 8 == 0 and qr.shape[-1] % 8 == 0):
        return _mla_fused(q, qr, lat, kr, q_lat, v_up, scale, isinstance(mask, str))
    qr = qr.astype(q.dtype) * scale
    if expanded:
        wq, wv = _dense_heads(q_lat), _dense_heads(v_up).swapaxes(1, 2)                # [H, r, hd] each
        qa = q * scale
    else:
        qa = _heads(q_lat, q) * scale                                                   # [B, H, L, r]
    causal = isinstance(mask, str)
    o = lse = None
    for k0 in range(0, n, nb):
        k1 = min(n, k0 + nb)
        lb, kb = lat[:, :, k0:k1], kr[:, :, k0:k1]
        keys = lb @ wq if expanded else lb                                              # [B, H or 1, nb, hd or r]
        # the two BF16 score products summed in f32 (a BF16 sum costs ~2x the error)
        s = (qa @ keys.swapaxes(-1, -2)).astype(mx.float32) + (qr @ kb.swapaxes(-1, -2)).astype(mx.float32)
        if causal and k1 > n - L:                                                       # the block meets the diagonal
            s = mx.where(mx.arange(k0, k1)[None] <= mx.arange(n - L, n)[:, None], s, -mx.inf)
        elif mask is not None and not causal:
            mb = mask[..., k0:k1]
            s = mx.where(mb, s, -mx.inf) if mb.dtype == mx.bool_ else s + mb
        vals = lb @ wv if expanded else lb
        ob = mx.softmax(s, axis=-1).astype(q.dtype) @ vals                              # normalized within the block
        if (mask is not None and not causal) or (causal and k0 > n - L):
            ob = mx.where(mx.isnan(ob), 0.0, ob)                                        # rows that see none of the block
        if nb >= n:
            o = ob
            break
        lb_ = mx.logsumexp(s, axis=-1, keepdims=True)
        if o is None:
            o, lse = ob.astype(mx.float32), lb_
        else:
            new = mx.logaddexp(lse, lb_)
            wa = mx.exp(lse - new)
            wb = mx.exp(lb_ - new)
            o = o * mx.where(mx.isnan(wa), 0.0, wa) + ob.astype(mx.float32) * mx.where(mx.isnan(wb), 0.0, wb)
            lse = new
        mx.eval(o, lse)               # one block in flight: MLX frees buffers only when a command buffer completes
    out = o.astype(q.dtype)
    return out if expanded else _heads(v_up, out)


@partial(mx.compile, shapeless=True)
def _score_sum(a, b, scale):
    """(a + b) * scale in f32: one kernel per decode step instead of four."""
    return (a.astype(mx.float32) + b.astype(mx.float32)) * scale


def _mla_blocked_quant(q, qr, lat, kr, q_lat, v_up, scale, mask, expanded, cache, nb):
    """_mla_blocked on a quantized cache (lat, kr = (codes, scales, biases) from MLAQuantCache)."""
    g, rot = cache.group_size, cache.rotate
    n = lat[0].shape[2]
    if not expanded and nb >= n:
        # decode: the codes straight through MLX's quantized mat-muls, queries rotated into the cache's basis
        qa, qr = _heads(q_lat, q), qr.astype(q.dtype)
        if rot:
            qa, qr = _rotate(qa), _rotate(qr)
        # heads as rows of one product against the shared cache (a head axis against its size-1 axis costs 2x)
        b, heads, L = q.shape[:3]
        rows = lambda x: x.reshape(b, 1, heads * L, x.shape[-1])
        s = _score_sum(mx.quantized_matmul(rows(qa), *lat, transpose=True, group_size=g, bits=cache.bits),
                       mx.quantized_matmul(rows(qr), *kr, transpose=True, group_size=g, bits=cache.rope_bits),
                       mx.array(scale, mx.float32)).reshape(b, heads, L, n)
        if isinstance(mask, str):
            if L > 1:
                s = mx.where(mx.arange(n)[None] <= mx.arange(n - L, n)[:, None], s, -mx.inf)
        elif mask is not None:
            s = mx.where(mask, s, -mx.inf) if mask.dtype == mx.bool_ else s + mask
        r = lat[0].shape[-1] * 32 // cache.bits
        mma_ok = cache.bits in (4, 8) and g == 64 and r % 128 == 0 and (heads * L) % 8 == 0 and heads * L <= 64
        if QUANT_PV == "mma" and mma_ok:
            o = _quant_pv_mma(rows(mx.softmax(s, axis=-1)), lat, g, cache.bits).astype(q.dtype)
        elif QUANT_PV in ("mma", "dequant"):   # one block of keys at a time to BF16, then the BF16 product
            pr = rows(mx.softmax(s, axis=-1).astype(q.dtype))
            o = None
            for k0 in range(0, n, DEQ_KEYS):
                blk = mx.dequantize(*(x[:, :, k0:k0 + DEQ_KEYS] for x in lat), group_size=g, bits=cache.bits)
                part = pr[..., k0:k0 + DEQ_KEYS] @ blk.astype(q.dtype)
                o = part.astype(mx.float32) if o is None else o + part.astype(mx.float32)
            o = o.astype(q.dtype)
        else:
            p = rows(mx.softmax(s, axis=-1).astype(q.dtype))
            o = mx.quantized_matmul(p, *lat, transpose=False, group_size=g, bits=cache.bits)
        o = o.reshape(b, heads, L, -1)
        return _heads(v_up, _rotate(o, inverse=True) if rot else o)
    # prompt chunks: dequantize (and rotate back) one key block at a time
    deq = lambda t, a, b_, bits: mx.dequantize(*(x[:, :, a:b_] for x in t), group_size=g, bits=bits).astype(q.dtype)

    class _Blocks:
        def __init__(self, t, bits):
            self.t, self.bits, self.shape = t, bits, (*t[0].shape[:3], None)

        def __getitem__(self, idx):
            a, b_ = idx[2].start, idx[2].stop
            x = deq(self.t, a, b_, self.bits)
            return _rotate(x, inverse=True) if rot else x

    return _mla_blocked(q, qr, _Blocks(lat, cache.bits), _Blocks(kr, cache.rope_bits), q_lat, v_up, scale, mask,
                        expanded, cache)


_QUANT_PV_MMA = """
// One threadgroup: output columns [cb * 128, +128) of all M rows, keys [k0, k1) of split sp. Per block of 32 keys the
// codes are dequantized into threadgroup memory (f32) and multiplied there with 8 x 8 simdgroup matrices: the codes
// are the only reads of the cache (no BF16 copy). Simdgroup sg: columns sg * 16 .. + 15, every row.
uint tid = thread_position_in_threadgroup.x;
uint sg = tid / 32;
uint cb = threadgroup_position_in_grid.x, sp = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
uint S = threadgroups_per_grid.y;
int N = p_shape[3];
int per = (N + S - 1) / S, k0 = sp * per, k1 = min(N, k0 + per);
constexpr int VPW = 32 / BITS, WPB = 128 / VPW, VLD = 132;
threadgroup float Vt[32 * VLD];
threadgroup float Pt[M * 32];
simdgroup_float8x8 acc[M / 8][2];
for (int r = 0; r < M / 8; ++r) { acc[r][0] = simdgroup_float8x8(0.0f); acc[r][1] = simdgroup_float8x8(0.0f); }
const device uint32_t* wb = w + b * w_strides[0];
const device T* sb = scales + b * scales_strides[0];
const device T* bb = biases + b * biases_strides[0];
auto pb = p + b * p_strides[0];
for (int base = k0; base < k1; base += 32) {
    for (int e = tid; e < 32 * WPB; e += 256) {
        int j = e / WPB, wi = e % WPB, d0 = wi * VPW, key = base + j;
        if (key < k1) {
            uint word = wb[key * w_strides[2] + (cb * WPB + wi) * w_strides[3]];
            int g = (cb * 128 + d0) / 64;
            float sc = float(sb[key * scales_strides[2] + g * scales_strides[3]]);
            float bi = float(bb[key * biases_strides[2] + g * biases_strides[3]]);
            for (int i = 0; i < VPW; ++i) Vt[j * VLD + d0 + i] = float((word >> (BITS * i)) & ((1u << BITS) - 1u)) * sc + bi;
        } else {
            for (int i = 0; i < VPW; ++i) Vt[j * VLD + d0 + i] = 0.0f;
        }
    }
    for (int e = tid; e < M * 32; e += 256) {
        int m = e / 32, j = e % 32;
        Pt[e] = base + j < k1 ? float(pb[m * p_strides[2] + (base + j) * p_strides[3]]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 4; ++kk) {
        simdgroup_float8x8 B0, B1, A;
        simdgroup_load(B0, Vt + kk * 8 * VLD + sg * 16, VLD);
        simdgroup_load(B1, Vt + kk * 8 * VLD + sg * 16 + 8, VLD);
        for (int r = 0; r < M / 8; ++r) {
            simdgroup_load(A, Pt + r * 8 * 32 + kk * 8, 32);
            simdgroup_multiply_accumulate(acc[r][0], A, B0, acc[r][0]);
            simdgroup_multiply_accumulate(acc[r][1], A, B1, acc[r][1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
device float* ob = part + ((size_t) b * S + sp) * M * R + cb * 128 + sg * 16;
for (int r = 0; r < M / 8; ++r) {
    simdgroup_store(acc[r][0], ob + r * 8 * R, R);
    simdgroup_store(acc[r][1], ob + r * 8 * R + 8, R);
}
"""
PV_SPLIT_KEYS = 256   # MMA P.V: keys per threadgroup (split-K), at most 256 splits (short contexts need the splits
                      # to fill the GPU: 2,048 keys per split halved P.V speed at 4K)


def _quant_pv_mma(p, lat, group_size, bits):
    """p @ dequantize(lat) for decode rows: p [B, 1, M, N] (M a multiple of 8, <= 64), lat (codes [B, 1, N, words],
    scales, biases); the latent dim a multiple of 128, group_size 64, 4 or 8 bits. Returns [B, 1, M, r] f32."""
    w, sc, bi = lat
    b, _, m, n = p.shape
    r = w.shape[-1] * 32 // bits
    splits = max(1, min(-(-n // PV_SPLIT_KEYS), 256))
    if "quant_pv_mma" not in _kernels:
        _kernels["quant_pv_mma"] = mx.fast.metal_kernel(
            name="nanoseedlm_mla_quant_pv_mma", input_names=["p", "w", "scales", "biases"], output_names=["part"],
            source=_QUANT_PV_MMA, ensure_row_contiguous=False)
    part = _kernels["quant_pv_mma"](
        inputs=[p, w, sc, bi], template=[("T", sc.dtype), ("M", m), ("R", r), ("BITS", bits)],
        grid=(256 * (r // 128), splits, b), threadgroup=(256, 1, 1),
        output_shapes=[(b, splits, m, r)], output_dtypes=[mx.float32])[0]
    return part[:, :1] if splits == 1 else part.sum(axis=1)[:, None]


QUANT_PV = "mma"      # decode P.V on a quantized cache: "mma" (_quant_pv_mma: reads only the codes; faster than the BF16
                      # product), "dequant" (BF16 blocks, then the BF16 product) or "qmm" (MLX's quantized vector-matrix)
DEQ_KEYS = 65536      # "dequant": keys per block (100 MB of BF16; 16K blocks: 3-8% slower decode at 128K-512K)


# Prompt chunks (expanded form), fused: per block of keys the latent is expanded by two MLX mat-muls (k_a = q_lat_a^T c,
# v_a = v_up_a c, all heads) and one kernel does scores, causal mask, online softmax and P.V in simdgroup registers,
# carrying the running output, max and sum to the next block: no score is written to memory, and the blocks are as
# large as SCORE_BUDGET allows for the expanded keys and values (32K keys at 512 MB) instead of for f32 scores.
_MLA_PREFILL = """
// One threadgroup: BQ query rows of head h (BQ / 8 simdgroups of 8 rows). Per tile of BK keys the keys [k_a ; k_rope]
// and the values v_a are staged in threadgroup memory; scores and P.V are 8 x 8 simdgroup products in f32. Fragment
// layout as MLX's steel kernels: lane holds row fm, columns fn and fn + 1 of each 8 x 8 matrix.
constexpr int BK = 32, DQ = HD + DR, KLD = DQ + 8, VLD = HD + 8, NT = BQ * 4;
threadgroup T Ks[BK * KLD];
threadgroup T Vs[BK * VLD];
const uint tid = thread_position_in_threadgroup.x, lane = tid % 32, sg = tid / 32;
const uint qt = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
const int H = q_shape[1], L = q_shape[2], NB = kn_shape[2];
const int k0g = prm[0], n = prm[1], causal = prm[2], first = prm[3];
const short qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
const int row = qt * BQ + sg * 8 + fm;
const bool rv = row < L;
const int qpos = n - L + row;
const size_t qrow = (size_t) (b * H + h) * L + (rv ? row : 0);
float2 Qf[DQ / 8];
for (int d = 0; d < HD / 8; ++d)
    Qf[d] = rv ? float2(float(q[qrow * HD + d * 8 + fn]), float(q[qrow * HD + d * 8 + fn + 1])) : float2(0.0f);
for (int d = 0; d < DR / 8; ++d)
    Qf[HD / 8 + d] = rv ? float2(float(qr[qrow * DR + d * 8 + fn]), float(qr[qrow * DR + d * 8 + fn + 1])) : float2(0.0f);
float2 Of[HD / 8];
float m_i = -INFINITY, l_i = 0.0f;
for (int d = 0; d < HD / 8; ++d)
    Of[d] = (first || !rv) ? float2(0.0f) : float2(o_in[qrow * HD + d * 8 + fn], o_in[qrow * HD + d * 8 + fn + 1]);
if (!first && rv) {
    m_i = ml_in[qrow * 2];
    l_i = ml_in[qrow * 2 + 1];
}
// causal: keys past the tile's last query position are skipped
const int qmax = n - L + min(L - 1, (int) (qt * BQ + BQ - 1));
const int kend = causal ? min(NB, qmax - k0g + 1) : NB;
const device T* knb = kn + (size_t) (b * H + h) * NB * HD;
const device T* krb = kr + (size_t) b * NB * DR;
const device T* vb = v + (size_t) (b * H + h) * NB * HD;
for (int j0 = 0; j0 < kend; j0 += BK) {
    for (int e = tid; e < BK * DQ / 4; e += NT) {
        int j = e / (DQ / 4), c = (e % (DQ / 4)) * 4, key = j0 + j;
        uint2 w = uint2(0);
        if (key < NB)
            w = c < HD ? *(const device uint2*) (knb + (size_t) key * HD + c)
                       : *(const device uint2*) (krb + (size_t) key * DR + c - HD);
        *(threadgroup uint2*) (Ks + j * KLD + c) = w;
    }
    for (int e = tid; e < BK * HD / 4; e += NT) {
        int j = e / (HD / 4), c = (e % (HD / 4)) * 4, key = j0 + j;
        *(threadgroup uint2*) (Vs + j * VLD + c) = key < NB ? *(const device uint2*) (vb + (size_t) key * HD + c) : uint2(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 S[BK / 8];
    float rmax = -INFINITY;
    for (int kb = 0; kb < BK / 8; ++kb) {
        simdgroup_float8x8 acc = simdgroup_float8x8(0.0f), A, B;
        for (int d = 0; d < DQ / 8; ++d) {
            reinterpret_cast<thread float2&>(A.thread_elements()) = Qf[d];
            reinterpret_cast<thread float2&>(B.thread_elements()) =
                float2(float(Ks[(kb * 8 + fn) * KLD + d * 8 + fm]), float(Ks[(kb * 8 + fn + 1) * KLD + d * 8 + fm]));
            simdgroup_multiply_accumulate(acc, A, B, acc);
        }
        float2 s = reinterpret_cast<thread float2&>(acc.thread_elements());
        for (int e = 0; e < 2; ++e) {
            int key = j0 + kb * 8 + fn + e;
            if (!rv || key >= NB || (causal && k0g + key > qpos))
                s[e] = -INFINITY;
        }
        rmax = max(rmax, max(s.x, s.y));
        S[kb] = s;
    }
    rmax = max(rmax, simd_shuffle_xor(rmax, 1));
    rmax = max(rmax, simd_shuffle_xor(rmax, 8));
    const float mnew = max(m_i, rmax);
    const float alpha = mnew == -INFINITY ? 1.0f : metal::precise::exp(m_i - mnew);
    float rsum = 0.0f;
    for (int kb = 0; kb < BK / 8; ++kb) {
        for (int e = 0; e < 2; ++e)
            S[kb][e] = S[kb][e] == -INFINITY ? 0.0f : metal::precise::exp(S[kb][e] - mnew);
        rsum += S[kb].x + S[kb].y;
    }
    rsum += simd_shuffle_xor(rsum, 1);
    rsum += simd_shuffle_xor(rsum, 8);
    l_i = l_i * alpha + rsum;
    m_i = mnew;
    for (int db = 0; db < HD / 8; ++db) {
        simdgroup_float8x8 acc, A, B;
        reinterpret_cast<thread float2&>(acc.thread_elements()) = Of[db] * alpha;
        for (int kb = 0; kb < BK / 8; ++kb) {
            reinterpret_cast<thread float2&>(A.thread_elements()) = S[kb];
            reinterpret_cast<thread float2&>(B.thread_elements()) =
                float2(float(Vs[(kb * 8 + fm) * VLD + db * 8 + fn]), float(Vs[(kb * 8 + fm) * VLD + db * 8 + fn + 1]));
            simdgroup_multiply_accumulate(acc, A, B, acc);
        }
        Of[db] = reinterpret_cast<thread float2&>(acc.thread_elements());
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (rv) {
    for (int d = 0; d < HD / 8; ++d) {
        o_out[qrow * HD + d * 8 + fn] = Of[d].x;
        o_out[qrow * HD + d * 8 + fn + 1] = Of[d].y;
    }
    if (fn == 0) {
        ml_out[qrow * 2] = m_i;
        ml_out[qrow * 2 + 1] = l_i;
    }
}
"""
FUSED_PREFILL = True  # prompt chunks with a causal mask or none: _mla_fused (False: the MLX-ops blocks above)
FUSED_KEYS = None     # keys per expanded block (None: SCORE_BUDGET / expanded bytes per key, at least 256)
FUSED_BQ = 32         # query rows per threadgroup (a multiple of 8)


def _mla_fused(q, qr, lat, kr, q_lat, v_up, scale, causal):
    """Expanded MLA attention for a prompt chunk through _MLA_PREFILL; lat / kr arrays or dequantizing views (_Blocks).
    Returns [B, H, L, hd] in q's dtype."""
    b, heads, L, hd = q.shape
    dr, n = qr.shape[-1], lat.shape[2]
    wq, wv = _dense_heads(q_lat), _dense_heads(v_up).swapaxes(1, 2)                    # [H, r, hd] each
    qa, qr = q * scale, qr.astype(q.dtype) * scale
    nk = FUSED_KEYS or max(256, SCORE_BUDGET // (4 * b * heads * hd) // 256 * 256)
    if "mla_prefill" not in _kernels:
        _kernels["mla_prefill"] = mx.fast.metal_kernel(
            name="nanoseedlm_mla_prefill", input_names=["q", "qr", "kn", "kr", "v", "o_in", "ml_in", "prm"],
            output_names=["o_out", "ml_out"], source=_MLA_PREFILL)
    o, ml = mx.zeros((b, heads, L, hd), mx.float32), mx.zeros((b, heads, L, 2), mx.float32)
    nt = FUSED_BQ * 4
    for k0 in range(0, n, nk):
        k1 = min(n, k0 + nk)
        lb = lat[:, :, k0:k1]
        o, ml = _kernels["mla_prefill"](
            inputs=[qa, qr, lb @ wq, kr[:, :, k0:k1], lb @ wv, o, ml, mx.array([k0, n, int(causal), int(k0 == 0)], mx.int32)],
            template=[("T", q.dtype), ("HD", hd), ("DR", dr), ("BQ", FUSED_BQ)],
            grid=(-(-L // FUSED_BQ) * nt, heads, b), threadgroup=(nt, 1, 1),
            output_shapes=[o.shape, ml.shape], output_dtypes=[mx.float32, mx.float32])
        if k1 < n:
            mx.eval(o, ml)            # one block of expanded keys in flight
    return (o / ml[..., 1:]).astype(q.dtype)


# A quantized MLA cache: per token the RoPE key (rope_bits) and the latent (bits), MLX affine groups of group_size,
# optionally rotated first by a Walsh-Hadamard transform with +-1 signs from the 16-bit LFSR (the rotation is a seed,
# nothing stored). Attention reads the codes through MLX's fused quantized mat-muls (decode) or dequantizes one key
# block at a time (prompt chunks), so no full-size copy of the cache is ever made.
_ROT = {}


def _rotation(n):
    """R = H_n D / sqrt(n), D = diag(+-1) from the LFSR (seed 0xACE1, taps 0, 1, 3, 12)."""
    if n not in _ROT:
        h = np.array([[1.0]], np.float32)
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        st, signs = 0xACE1, np.empty(n, np.float32)
        for i in range(n):
            st = (st >> 1) | (((st ^ (st >> 1) ^ (st >> 3) ^ (st >> 12)) & 1) << 15)
            signs[i] = 1.0 if st & 1 else -1.0
        _ROT[n] = mx.array(h * signs[None, :] / np.sqrt(n))
    return _ROT[n]


def _rotate(x, inverse=False):
    """x R^T (or x R) in blocks of the largest power of two <= 256 dividing the last dim."""
    d = x.shape[-1]
    n = min(256, d & -d)
    rot = _rotation(n).astype(x.dtype)
    return (x.reshape(*x.shape[:-1], d // n, n) @ (rot if inverse else rot.T)).reshape(x.shape)


class MLAQuantCache(_BaseCache):
    """update_and_fetch(kr, lat) stores both quantized and returns their (codes, scales, biases) up to the offset."""

    step = 256

    def __init__(self, bits=4, rope_bits=8, group_size=64, rotate=True):
        self.keys = self.values = None
        self.offset = 0
        self.bits, self.rope_bits, self.group_size, self.rotate = bits, rope_bits, group_size, rotate

    def _quant(self, kr, lat):
        if self.rotate:
            kr, lat = _rotate(kr), _rotate(lat)
        return (mx.quantize(kr, group_size=self.group_size, bits=self.rope_bits),
                mx.quantize(lat, group_size=self.group_size, bits=self.bits))

    def _grow(self, new, used, n):
        steps = (self.step + n - 1) // self.step * self.step
        blank = lambda x: mx.zeros((*x.shape[:2], steps, x.shape[3]), x.dtype)
        if self.keys is None:
            self.keys, self.values = tree_map(blank, new)
            return
        if used % self.step:
            self.keys, self.values = tree_map(lambda x: x[..., :used, :], (self.keys, self.values))
        self.keys, self.values = tree_map(lambda x: mx.concatenate([x, blank(x)], axis=2), (self.keys, self.values))

    def _write(self, new, a, b):
        for buf, part in zip((self.keys, self.values), new):
            for i in range(3):
                buf[i][..., a:b, :] = part[i]

    def update_and_fetch(self, kr, lat):
        new, n, prev = self._quant(kr, lat), kr.shape[2], self.offset
        if self.keys is None or prev + n > self.keys[0].shape[2]:
            self._grow(new, prev, n)
        self.offset += n
        self._write(new, prev, self.offset)
        return self.keys_and_values()

    def keys_and_values(self):
        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(self):
        return self.keys_and_values() if self.keys is not None else []

    @state.setter
    def state(self, v):
        if v:
            self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.bits, self.rope_bits, self.group_size, int(self.rotate))))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.bits, self.rope_bits, self.group_size, rot = map(int, v)
        self.rotate = bool(rot)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def size(self):
        return self.offset

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return 0 if self.keys is None else sum(x.nbytes for t in (self.keys, self.values) for x in t)

    @classmethod
    def merge(cls, caches):
        return BatchMLAQuantCache.merge(caches)


class BatchMLAQuantCache(MLAQuantCache):
    """MLAQuantCache for continuous batching (mlx-lm's BatchKVCache protocol: left padding, filter, extend, extract)."""

    def __init__(self, left_padding, bits=4, rope_bits=8, group_size=64, rotate=True):
        super().__init__(bits, rope_bits, group_size, rotate)
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-p for p in left_padding])
        self._idx = 0
        self._right_padding = None

    def _params(self):
        return dict(bits=self.bits, rope_bits=self.rope_bits, group_size=self.group_size, rotate=self.rotate)

    def update_and_fetch(self, kr, lat):
        new, n, prev = self._quant(kr, lat), kr.shape[2], self._idx
        if self.keys is None or prev + n > self.keys[0].shape[2]:
            self._grow(new, prev, n)
        self.offset = self.offset + n
        self._idx += n
        self._write(new, prev, self._idx)
        return self.keys_and_values()

    def keys_and_values(self):
        return tree_map(lambda x: x[..., : self._idx, :], (self.keys, self.values))

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError("Left padding can only be added to an empty BatchMLAQuantCache")
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset = self.offset - left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            pad = self._right_padding
            self.keys, self.values = tree_map(lambda x: dynamic_roll(x, pad[:, None], axis=2), (self.keys, self.values))
            self.offset = self.offset - pad
            self.left_padding += pad
            self._right_padding = None

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v

    @property
    def meta_state(self):
        return tuple(map(str, (self._idx, self.bits, self.rope_bits, self.group_size, int(self.rotate))))

    @meta_state.setter
    def meta_state(self, v):
        self._idx, self.bits, self.rope_bits, self.group_size, rot = map(int, v)
        self.rotate = bool(rot)

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    def make_mask(self, N, return_array=False, **kwargs):
        return create_causal_mask(N, offset=self._idx, left_padding=self.left_padding, **kwargs)

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys, self.values = tree_map(lambda x: x[batch_indices], (self.keys, self.values))
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        min_left = min(self.left_padding.tolist())
        if min_left > 0:
            if self.keys is not None:
                self.keys, self.values = tree_map(lambda x: x[..., min_left:, :], (self.keys, self.values))
            self._idx -= min_left
            self.left_padding -= min_left

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        ref = self if self.keys is not None else other
        max_idx = max(self._idx, other._idx)
        max_size = max(c.keys[0].shape[2] for c in (self, other) if c.keys is not None)

        def pad(c):
            kv = (c.keys, c.values)
            if c.keys is None:
                b = c.offset.shape[0]
                kv = tree_map(lambda x: mx.zeros((b, x.shape[1], 0, x.shape[3]), x.dtype), (ref.keys, ref.values))
            left = max_idx - c._idx
            right = max_size - kv[0][0].shape[2] - left
            if right < 0:
                kv = tree_map(lambda x: x[..., :right, :], kv)
                right = 0
            if left or right:
                kv = tree_map(lambda x: mx.pad(x, [(0, 0), (0, 0), (left, right), (0, 0)]), kv)
            return kv, c.offset, c.left_padding + left

        (ka, oa, la), (kb, ob, lb) = pad(self), pad(other)
        self.keys, self.values = tree_map(lambda x, y: mx.concatenate([x, y]), ka, kb)
        self.offset = mx.concatenate([oa, ob])
        self.left_padding = mx.concatenate([la, lb])
        self._idx = max_idx

    def extract(self, idx):
        cache = getattr(self, "_single", MLAQuantCache)(**self._params())
        pad = self.left_padding.tolist()[idx]
        cache.keys, cache.values = tree_map(lambda x: mx.contiguous(x[idx:idx + 1, :, pad:self._idx]),
                                            (self.keys, self.values))
        cache.offset = self._idx - pad
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        top = max(lengths)
        params = next((c for c in caches), None)
        params = dict(bits=params.bits, rope_bits=params.rope_bits, group_size=params.group_size, rotate=params.rotate)
        if top == 0:
            empty = cls([0] * len(caches), **params)
            empty._single = type(caches[0])
            return empty
        padding = [top - n for n in lengths]
        full = next(c for c in caches if c.keys is not None)
        blank = lambda x: mx.zeros((len(caches), x.shape[1], top, x.shape[3]), x.dtype)
        keys, values = tree_map(blank, (full.keys, full.values))
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            for dst, src in zip(keys + values, c.keys + c.values):
                dst[i:i + 1, :, p:p + c.offset] = src[..., : c.offset, :]
        cache = cls(padding, **params)
        cache.keys, cache.values = keys, values
        cache.offset += top
        cache._idx = top
        cache._single = type(caches[0])
        return cache

    def size(self):
        return self._idx


# The cache format and the score budget, sized to this Mac. q8 (8-bit latent and RoPE key: half the BF16 cache, no
# measurable quality loss) unless the requested max_context needs more memory than is available; then rq4 (the latent
# rotated and in 4 bits: a third of BF16, a small quality cost). docs/kv-cache.md in NanoSeedLM has the trade-offs.
KV_FORMATS = {
    "q8": dict(bits=8, rope_bits=8, group_size=64, rotate=False),
    "rq4": dict(bits=4, rope_bits=8, group_size=64, rotate=True),
}
KV_DOC = "https://github.com/mbarnson/NanoSeedLM/blob/main/docs/kv-cache.md"
_KV_LOGGED: set = set()


def kv_bytes(args, fmt):
    """Cache bytes per token over all layers: codes plus a BF16 scale and bias per group, latent and RoPE key."""
    f = KV_FORMATS[fmt]
    per = lambda d, bits: d * bits // 8 + d // f["group_size"] * 4
    gqa = 2 * args.num_key_value_heads * args.head_dim * 2                    # a GQA layer (rank 0): BF16 K and V
    return sum(per(r, f["bits"]) + per(args.mla_rope_dim, f["rope_bits"]) if r else gqa for r in args.mla_ranks)


def choose_kv(args, max_context, available):
    """(format, message): q8 unless max_context tokens need more than `available` bytes as q8."""
    if not max_context or max_context * kv_bytes(args, "q8") <= available:
        return "q8", None
    q8, rq4 = (max_context * kv_bytes(args, f) / 1e9 for f in ("q8", "rq4"))
    msg = (f"NanoSeedLM: KV cache rq4 (4-bit latent) instead of q8: max_context {max_context:,} tokens needs "
           f"{q8:.1f} GB as q8, more than the {available / 1e9:.1f} GB available after the weights; rq4 needs {rq4:.1f} GB "
           f"at a small quality cost ({KV_DOC})")
    if rq4 * 1e9 > available:
        fit = int(max(available, 0) // kv_bytes(args, "rq4"))
        msg += f". Even rq4 does not fit: about {fit:,} tokens fit"
    return "rq4", msg


def mla_tune(working_set=None):
    """The prompt-attention score budget (transient ~5.3x) within 5% of the GPU working set, 64 MB .. 512 MB in powers
    of two."""
    if working_set is None:
        working_set = _device()["max_recommended_working_set_size"]
    budget = 64 << 20
    while budget < 512 << 20 and 6 * 2 * budget <= 0.05 * working_set:
        budget *= 2
    return {"score_budget": budget}


def _device():
    return mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()


def mla_kv_auto(args, max_context=None):
    """The cache settings for this Mac: sets the score budget, then picks q8 / rq4 for the requested max_context
    (make_cache(max_context=...), else config mla_max_context, else env NSLM_MAX_CONTEXT, else none: q8)."""
    import os
    global SCORE_BUDGET
    working_set = _device()["max_recommended_working_set_size"]
    SCORE_BUDGET = mla_tune(working_set)["score_budget"]
    max_context = max_context or args.mla_max_context or int(os.environ.get("NSLM_MAX_CONTEXT", "0") or 0)
    available = working_set - mx.get_active_memory() - 6 * SCORE_BUDGET - (2 << 30)   # weights, prompt transients
    fmt, msg = choose_kv(args, max_context, available)
    if msg and msg not in _KV_LOGGED:
        _KV_LOGGED.add(msg)
        print(msg, flush=True)
    return KV_FORMATS[fmt]


class MLAQ8Cache(MLAQuantCache):
    """The q8 cache (its own class name: oMLX restores prefix-cache blocks by class name, without metadata)."""

    FORMAT = "q8"

    def __init__(self, **kv):
        super().__init__(**{**KV_FORMATS["q8"], **kv})


class MLARQ4Cache(MLAQuantCache):
    """The rq4 cache."""

    FORMAT = "rq4"

    def __init__(self, **kv):
        super().__init__(**{**KV_FORMATS["rq4"], **kv})


_OMLX_REUSE = False
_FALLBACK_LOGGED = False


def mla_caches(args, layers, max_context=None):
    """make_cache for an MLA model: config mla_kv "bf16" -> KVCaches; a dict -> that MLAQuantCache; None / "auto" ->
    q8 or rq4 (mla_kv_auto). Under oMLX without prefix reuse for the quantized caches (register_omlx_prefix_reuse
    failed: an oMLX whose internals changed), "auto" falls back to KVCaches rather than lose prefix reuse."""
    import sys
    global _FALLBACK_LOGGED
    kv = args.mla_kv
    if kv in (None, "auto") and "omlx" in sys.modules and not _OMLX_REUSE:
        if not _FALLBACK_LOGGED:
            _FALLBACK_LOGGED = True
            print("NanoSeedLM: KV cache BF16: this oMLX version cannot reuse prompt prefixes with the q8 / rq4 caches, "
                  f"and prefix reuse matters more ({KV_DOC})", flush=True)
        kv = "bf16"
    if kv == "bf16":
        return [KVCache() for _ in range(layers)]
    if kv in (None, "auto"):
        kv = mla_kv_auto(args, max_context)
    cls = next((c for c in (MLAQ8Cache, MLARQ4Cache) if KV_FORMATS[c.FORMAT] == dict(kv)), MLAQuantCache)
    return [cls(**kv) if args.mla_ranks[i] else KVCache() for i in range(layers)]


_STATE_NAMES = ("k_codes", "k_scales", "k_biases", "v_codes", "v_scales", "v_biases")


def _omlx_handler(base, axis_info, cls):
    """An oMLX cache-type handler for cls: its state is the six quantized arrays, each with the sequence on axis 2
    (sliced, concatenated and restored like a KV cache's keys and values)."""

    class CacheType:
        value = cls.__name__

    class Handler(base):
        @property
        def cache_type(self):
            return CacheType

        @property
        def supports_block_slicing(self):
            return True

        def get_state_axis_info(self):
            return tuple(axis_info(n, 2, True) for n in _STATE_NAMES)

        def _get_state_keys(self):
            return _STATE_NAMES

        def serialize_state(self, cache_obj):
            if cache_obj.keys is None:
                return (None,) * 6
            k, v = cache_obj.keys_and_values()
            return tuple(k) + tuple(v)

        def serialize_meta_state(self, cache_obj):
            return tuple(cache_obj.meta_state)

        def _elements(self, state):
            e = state.get("states")
            return tuple(e) if e is not None else tuple(state.get(n) for n in _STATE_NAMES)

        def extract_state(self, cache_obj):
            e = self.serialize_state(cache_obj)
            return {**dict(zip(_STATE_NAMES, e)), "states": e, "offset": cache_obj.offset, "cache_type": CacheType.value}

        def get_seq_len(self, state):
            e = self._elements(state)
            return int(e[0].shape[2]) if e[0] is not None else 0

        def slice_state(self, state, start_idx, end_idx):
            e = self._elements(state)
            end = min(end_idx, self.get_seq_len(state))
            if start_idx >= end:
                return None
            part = tuple(x[:, :, start_idx:end] for x in e)
            return {**dict(zip(_STATE_NAMES, part)), "states": part, "cache_type": CacheType.value}

        def concatenate_states(self, states):
            parts = [self._elements(st) for st in states if st]
            parts = [p for p in parts if p[0] is not None]
            if not parts:
                return {}
            cat = tuple(mx.concatenate([p[i] for p in parts], axis=2) for i in range(6))
            return {**dict(zip(_STATE_NAMES, cat)), "states": cat, "offset": int(cat[0].shape[2]),
                    "cache_type": CacheType.value}

        def deserialize_state(self, elements, meta_state=None):
            elements = tuple(elements)
            cache = cls()
            if elements and elements[0] is not None:
                cache.keys, cache.values = tuple(elements[:3]), tuple(elements[3:])
                cache.offset = int(elements[0].shape[2])
            return cache

        def reconstruct_cache(self, state, meta_state=None):
            return self.deserialize_state(self._elements(state), meta_state)

    return Handler()


def register_omlx_prefix_reuse():
    """Prompt-prefix reuse in oMLX (its paged prefix cache and SSD cache) for the q8 / rq4 caches, from this model file:
    a cache-type handler per class, the class names added to oMLX's sliceable sets, then a round trip through each
    handler. All or nothing: on any failure oMLX is put back as it was (and mla_caches falls back to BF16).
    Returns whether reuse is on."""
    global _OMLX_REUSE
    try:
        import omlx.scheduler as sched
        from omlx.cache.type_handlers import CacheStateAxisInfo, CacheTypeHandler
        from omlx.cache.type_registry import CacheTypeRegistry as reg
    except Exception:
        return False
    try:
        import omlx.specprefill.draft as draft
    except Exception:
        draft = None
    classes = (MLAQ8Cache, MLARQ4Cache)
    added = frozenset(c.__name__ for c in classes)
    saved = (dict(reg._handlers), dict(reg._class_name_map), sched._KNOWN_SLICEABLE_CACHE_TYPES,
             draft._SLICEABLE_CACHE_TYPES if draft is not None else None)
    try:
        for cls in classes:
            h = _omlx_handler(CacheTypeHandler, CacheStateAxisInfo, cls)
            reg.register(h)
            reg._class_name_map[cls.__name__] = h.cache_type
        sched._KNOWN_SLICEABLE_CACHE_TYPES = frozenset(sched._KNOWN_SLICEABLE_CACHE_TYPES) | added
        if draft is not None:
            draft._SLICEABLE_CACHE_TYPES = frozenset(draft._SLICEABLE_CACHE_TYPES) | added
        for cls in classes:   # serialize, two blocks, concatenate, reconstruct: the same arrays back
            h = reg.get_handler_by_class_name(cls.__name__)
            c = cls()
            c.update_and_fetch(mx.ones((1, 1, 70, 128), mx.bfloat16), mx.ones((1, 1, 70, 256), mx.bfloat16))
            e = h.serialize_state(c)
            back = h.reconstruct_cache(h.concatenate_states([h.slice_state({"states": e}, 0, 64),
                                                             h.slice_state({"states": e}, 64, 70)]))
            if not (type(back) is cls and back.offset == 70 and cls.__name__ in reg.list_known_class_names()
                    and all(bool(mx.array_equal(a, b)) for a, b in zip(h.serialize_state(back), e))):
                raise RuntimeError(f"{cls.__name__}: prefix-cache round trip failed")
        _OMLX_REUSE = True
    except Exception:
        reg._handlers.clear()
        reg._handlers.update(saved[0])
        reg._class_name_map.clear()
        reg._class_name_map.update(saved[1])
        sched._KNOWN_SLICEABLE_CACHE_TYPES = saved[2]
        if draft is not None:
            draft._SLICEABLE_CACHE_TYPES = saved[3]
        _OMLX_REUSE = False
    return _OMLX_REUSE


def mla_attention(q, qr, lat, kr, q_lat, v_up, scale, mask=None, cache=None, path=None):
    """MLA attention per head: v_up_a softmax(scale ((q_lat_a q_a) . c + qr_a . kr)) c, shape [B, H, L, hd].

    q [B, H, L, hd] (q_proj heads), qr [B, H, L, dr] (RoPE applied), lat [B, 1, N, r] and kr [B, 1, N, dr] (cache
    contents, or (codes, scales, biases) from MLAQuantCache), q_lat / v_up BF16 arrays or SeedHeads. path: None chooses; "blocked" (absorbed, key blocks), "expanded"
    (expanded, key blocks), "absorbed" (MLX's attention on the concatenated keys: the reference, unbounded memory)."""
    if path is None or (path == "absorbed" and isinstance(lat, (tuple, list))):
        path = "expanded" if q.shape[2] >= EXPAND_MIN_L else "blocked"
    if path == "absorbed":
        keys = mx.concatenate([lat, kr], axis=-1)
        queries = mx.concatenate([_heads(q_lat, q), qr], axis=-1)
        out = scaled_dot_product_attention(queries, keys, lat, cache=cache, scale=scale, mask=mask)
        return _heads(v_up, out)
    return _mla_blocked(q, qr, lat, kr, q_lat, v_up, scale, mask, expanded=path == "expanded", cache=cache)


class MLA(nn.Module):
    """One layer's MLA tensors (BF16 arrays, including affine tensors decoded at load, or seed modules)."""

    def __init__(self, args: ModelArgs, rank: int, mova: bool):
        super().__init__()
        nh, hd, d = args.num_attention_heads, args.head_dim, args.hidden_size
        self.kv_a_x = mx.zeros((rank, d))
        if mova:
            self.kv_a_v = mx.zeros((rank, args.num_key_value_heads * hd))
        self.k_rope_proj = mx.zeros((args.mla_rope_dim, d))
        self.q_rope_mix = mx.zeros((nh, args.mla_rope_dim, hd))
        self.q_lat = mx.zeros((nh, rank, hd))
        self.v_up = mx.zeros((nh, hd, rank))


class MLAAttention(_k2.Attention):
    def __init__(self, args: ModelArgs, mova: bool, rank: int):
        super().__init__(args, mova)
        del self.k_proj
        if not mova:
            del self.v_proj
        self.mla = MLA(args, rank, mova)

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache=None) -> mx.array:
        b, L, _ = x.shape
        m = self.mla
        q = self.q_proj(x).reshape(b, L, self.n_heads, -1).transpose(0, 2, 1, 3)       # [B, heads, L, 128]
        c = _lin(m.kv_a_x, x)                                                           # [B, L, r]
        if self.mova:
            c = c + _lin(m.kv_a_v, self._values(x))
        kr = _lin(m.k_rope_proj, x)[:, None]                                            # [B, 1, L, 128]
        offset = cache.offset if cache is not None else 0
        qr = self.rope(_heads(m.q_rope_mix, q), offset=offset)
        kr = self.rope(kr, offset=offset)
        lat = c[:, None].astype(x.dtype)                                                # [B, 1, L, r]
        if cache is not None:
            kr, lat = cache.update_and_fetch(kr, lat)                                   # cache: 128 + r per token
        out = mla_attention(q, qr, lat, kr, m.q_lat, m.v_up, self.scale, mask, cache)
        out = out.transpose(0, 2, 1, 3)                                                 # [B, L, heads, 128]
        if "gate_proj" in self:
            gate = _k2.softplus_beta_ln2(self.gate_proj(x)).reshape(b, L, self.n_heads, -1)
            out = out * gate
        return self.o_proj(out.reshape(b, L, -1))


def _module(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


class Model(_k2.Model):
    """K2-Horizon (GQA, or MLA with config mla_ranks) whose tensors with NAME.seeds become SeedSwitchLinear (expert
    stacks), SeedLinear, SeedEmbedding or SeedHeads (MLA per-head maps)."""

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        if args.mla_ranks:
            if len(args.mla_ranks) != args.num_hidden_layers:
                raise ValueError(f"mla_ranks has {len(args.mla_ranks)} entries for {args.num_hidden_layers} layers")
            for i, layer in enumerate(self.model.layers):
                if args.mla_ranks[i]:                     # 0: the layer keeps its original (GQA) attention
                    layer.self_attn = MLAAttention(args, getattr(layer.self_attn, "mova", False), args.mla_ranks[i])

    def make_cache(self, max_context=None):
        """MLA: q8 or rq4 caches for max_context, or as config mla_kv gives (mla_caches); GQA: KVCaches."""
        if not self.args.mla_ranks:
            return [KVCache() for _ in self.model.layers]
        return mla_caches(self.args, len(self.model.layers), max_context)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        seeded = sorted({k[: -len(".seeds")] for k in weights if k.endswith(".seeds")})
        for p in seeded:
            if p + ".nibbles" in weights:   # SEED4 (P = 3): the nibble words carry the exponent codes
                weights[p + ".coefs"] = weights.pop(p + ".nibbles").view(mx.int16)
                weights[p + ".codes"] = mx.zeros((1,), mx.uint8)
            if p + ".codes" not in weights:
                raise ValueError(f"{p}: only SeedLM P=3 / P=4 / P=8 tensors load in MLX")
            weights[p + ".weight"] = mx.zeros((0,))   # the stack exists: K2's sanitize must not rebuild it
        weights = super().sanitize(weights)
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer.self_attn, MLAAttention):   # TransMLA drops k_proj (and v_proj where values are not experts)
                weights.pop(f"model.layers.{i}.self_attn.k_proj.weight", None)
                if not layer.self_attn.mova:
                    weights.pop(f"model.layers.{i}.self_attn.v_proj.weight", None)
                # The packer stores affine MLA arrays as NAME, NAME.scales, NAME.biases (no .weight).
                # They are arrays rather than quantizable Linear modules; restore their BF16 values here.
                for name, value in layer.self_attn.mla.items():
                    path = f"model.layers.{i}.self_attn.mla.{name}"
                    if path + ".scales" not in weights:
                        continue
                    packed = weights[path]
                    bits, rem = divmod(32 * packed.shape[-1], value.shape[-1])
                    if rem or bits not in (4, 8):
                        raise ValueError(f"{path}: expected affine Q4 or Q8 MLA weights")
                    weights[path] = mx.dequantize(
                        packed, weights.pop(path + ".scales"), weights.pop(path + ".biases"), group_size=64, bits=bits
                    ).astype(mx.bfloat16)
        for p in seeded:
            weights.pop(p + ".weight")
            e, n, kb = weights[p + ".seeds"].shape
            pp = {mx.int16: 3, mx.uint16: 4, mx.uint32: 8}[weights[p + ".coefs"].dtype]
            parent, _, attr = p.rpartition(".")
            if parent and isinstance(_module(self, parent), MLA):
                setattr(_module(self, parent), attr, SeedLinear(kb * 8, n, pp) if e == 1 else SeedHeads(e, kb * 8, n, pp))
                continue
            m = _module(self, p)
            if hasattr(m, "num_experts") and pp != 8:
                new = SeedSwitchLinear(kb * 8, n, e, pp)
            elif isinstance(m, nn.Embedding) and e == 1:
                new = SeedEmbedding(n, kb * 8, pp)
            elif isinstance(m, nn.Linear) and e == 1 and "bias" not in m:
                new = SeedLinear(kb * 8, n, pp)
            else:
                raise ValueError(f"{p}: seed weights need a SwitchLinear, bias-free Linear or Embedding module")
            self.update_modules(tree_unflatten([(p, new)]))
        for parent in {p.rsplit(".", 1)[0] for p in seeded}:
            m = _module(self, parent)
            if type(m) is SwitchGLU and all(isinstance(getattr(m, k), SeedSwitchLinear) for k in ("gate_proj", "up_proj", "down_proj")):
                m.__class__ = SeedSwitchGLU
                block = _module(self, parent.rsplit(".", 1)[0])
                if type(block) is getattr(_k2, "SparseMoeBlock", None) and hasattr(_k2, "route"):
                    block.__class__ = SeedSparseMoeBlock
        return weights


if "omlx" in __import__("sys").modules:   # loaded by oMLX (trust_remote_code): prompt-prefix reuse for q8 / rq4
    register_omlx_prefix_reuse()
