# SPDX-License-Identifier: MIT
"""NanoSeedLM for MLX: K2-Horizon with SeedLM P=4 expert weights, decoded on the GPU.

mlx-lm loads this file through config.json "model_file" (trust_remote_code). It needs the K2-Horizon model
(mlx_lm.models.k2_horizon, provided by oMLX).

Block: 8 weights from a 16-bit LFSR seed (K=16, taps 0,1,3,12), 4 int4 coefficients and a 4-bit exponent code.
With S[c][p] = state(4c+p+1) - 32768: w_c = bf16(R32 * 2^(exp_bias+code) * sum_p S[c][p] * q_p), R32 = fl(1/32767).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
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
    uint s = seeds[g];
    uint q = coefs[g];
    int code = (codes[g >> 1] >> ((g & 1) * 4)) & 15;
    float sc = ldexp(R32, bias + code);
    ulong st = ulong(s) | (ulong(table[s]) << 16);
    int q0 = int(q << 28) >> 28, q1 = int(q << 24) >> 28, q2 = int(q << 20) >> 28, q3 = int(q << 16) >> 28;
    float d = 0.0f;
    for (uint c = 0; c < 8; ++c) {
        uint k = 4 * c + 1;
        int isum = (int((st >> k) & 0xFFFF) - 32768) * q0 + (int((st >> (k + 1)) & 0xFFFF) - 32768) * q1
                 + (int((st >> (k + 2)) & 0xFFFF) - 32768) * q2 + (int((st >> (k + 3)) & 0xFFFF) - 32768) * q3;
        if (w) w[c] = float(isum) * sc;
        if (xv) d += float(isum) * xv[c];
    }
    return d * sc;
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


class SeedSwitchLinear(nn.Module):
    """mlx-lm's SwitchLinear with SeedLM P=4 weights."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__()
        if input_dims % BK:
            raise ValueError(f"SeedLM P=4 in MLX needs input_dims divisible by {BK}")
        self.seeds = mx.zeros((num_experts, output_dims, input_dims // 8), mx.uint16)
        self.coefs = mx.zeros((num_experts, output_dims, input_dims // 8), mx.uint16)
        self.codes = mx.zeros((num_experts, output_dims, input_dims // 16), mx.uint8)
        self.exp_bias = mx.zeros((num_experts,), mx.int32)
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


class SeedSwitchGLU(SwitchGLU):
    """SwitchGLU on seed projections: the fused mat-vec for few pairs, else expert-sorted pairs through the seeded GEMM."""

    def _matvec(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        x_up = self.up_proj.matvec(x, indices)
        x_gate = self.gate_proj.matvec(x, indices)
        return self.down_proj.matvec(self.activation(x_up, x_gate), indices).squeeze(-2)

    def __call__(self, x, indices) -> mx.array:
        if indices.size <= MATVEC_MAX_PAIRS:
            return self._matvec(x, indices)
        xs, idx, order = _sort_pairs(x, indices)
        tiles = expert_tiles(idx, self.up_proj.seeds.shape[0])
        h = self.activation(self.up_proj.gemm(xs, idx, tiles), self.gate_proj.gemm(xs, idx, tiles))
        return _unsort(self.down_proj.gemm(h, idx, tiles), order, indices.shape)


@dataclass
class ModelArgs(_k2.ModelArgs):
    pass


def _module(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


class Model(_k2.Model):
    """K2-Horizon whose expert stacks with NAME.seeds tensors become SeedSwitchLinear."""

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        seeded = sorted({k[: -len(".seeds")] for k in weights if k.endswith(".seeds")})
        for p in seeded:
            if p + ".codes" not in weights:
                raise ValueError(f"{p}: only SeedLM P=4 tensors load in MLX")
            weights[p + ".weight"] = mx.zeros((0,))   # the stack exists: K2's sanitize must not rebuild it
        weights = super().sanitize(weights)
        for p in seeded:
            weights.pop(p + ".weight")
            e, n, kb = weights[p + ".seeds"].shape
            if not hasattr(_module(self, p), "num_experts"):
                raise ValueError(f"{p}: seed weights need a SwitchLinear module")
            self.update_modules(tree_unflatten([(p, SeedSwitchLinear(kb * 8, n, e))]))
        for parent in {p.rsplit(".", 1)[0] for p in seeded}:
            m = _module(self, parent)
            if type(m) is SwitchGLU and all(isinstance(getattr(m, k), SeedSwitchLinear) for k in ("gate_proj", "up_proj", "down_proj")):
                m.__class__ = SeedSwitchGLU
        return weights
