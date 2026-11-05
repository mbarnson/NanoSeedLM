# SPDX-License-Identifier: MIT
"""NanoSeedLM for MLX: the SeedLM weight kernels (Metal) and the seed modules that the model loaders share.

Block: 8 weights from a 16-bit LFSR seed (K=16, taps 0,1,3,12), P int4 coefficients and a 4-bit exponent code (P = 3,
4 or 8). With S[c][p] = state(Pc+p+1) - 32768: w_c = bf16(R32 * 2^(exp_bias+code) * sum_p S[c][p] * q_p), R32 = fl(1/32767).
"""

import mlx.core as mx
import mlx.nn as nn
try:
    from mlx_lm.models.switch_layers import SwitchGLU
except ImportError:   # mlx-vlm without mlx-lm
    from mlx_vlm.models.switch_layers import SwitchGLU

# (token, expert) pairs at or below this use the fused mat-vec; above it, the pairs are sorted by expert and go
# through the seeded tile GEMM (weight tiles rebuilt in threadgroup memory; no BF16 expert tensor).
MATVEC_MAX_PAIRS = 64
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

# One threadgroup per (BM-row block, tile of one expert's sorted pairs; tiles / starts cumulative per expert): BF16
# weight tiles, f32 simdgroup MMA (4 simdgroups of 32 rows x 32 pairs).
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

# out[t] = sum_j T(y[inverse[t K + j]] * T(w[t][j])): (y * w).sum(-2) on expert-sorted rows, rounded to T as mlx does
# (explicitly: the compiler may keep a loop-carried bfloat in f32); 4 columns per thread.
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
        s = mx.arange(65536, dtype=mx.uint32)
        g = mx.zeros_like(s)
        for k in range(32):   # one Fibonacci LFSR step per bit, all 65536 seeds at once
            b = (s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1
            s = (s >> 1) | (b << 15)
            g = g | (b << k)
        _table = g
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

    def __call__(self, x, indices, sorted_indices=False):
        if sorted_indices:   # rows already in expert order, one per index (mlx-lm's _gather_sort; oMLX prefill)
            xs, idx = x.reshape(-1, self.input_dims), indices.reshape(-1).astype(mx.uint32)
            y = gather_matvec(xs, idx, *self._seeds()) if idx.size <= MATVEC_MAX_PAIRS else self.gemm(xs, idx)
            return y.reshape(x.shape[:-1] + (self.output_dims,))
        if indices.size * x.shape[-2] <= MATVEC_MAX_PAIRS:
            return self.matvec(x, indices)
        if x.shape[-2] != 1:
            raise ValueError("SeedSwitchLinear: one row per token or pair above the mat-vec limit")
        # one row per (token, expert) pair: x may hold one per token (..., 1, 1, cols) or one per pair (..., k, 1, cols)
        xs = mx.broadcast_to(x, indices.shape + x.shape[-2:]).reshape(-1, self.input_dims)
        idx = indices.reshape(-1).astype(mx.uint32)
        order = mx.argsort(idx)
        return _unsort(self.gemm(xs[order], idx[order]), order, indices.shape).reshape(indices.shape + (1, self.output_dims))


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


def seed_p(weights: dict, prefix: str) -> int:
    """P of the seed tensors prefix.*; SEED4 (P = 3) nibble words become int16 coefs with a placeholder codes."""
    if prefix + ".nibbles" in weights:
        weights[prefix + ".coefs"] = weights.pop(prefix + ".nibbles").view(mx.int16)
        weights[prefix + ".codes"] = mx.zeros((1,), mx.uint8)
    return {mx.int16: 3, mx.uint16: 4, mx.uint32: 8}[weights[prefix + ".coefs"].dtype]


def _module(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m
