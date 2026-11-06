# SPDX-License-Identifier: MIT
"""NanoSeedLM for MLX: trellis-coded routed experts (nslm/trellis.h format v0) from nslm-moe's .blkt banks.

A bank L{l}_{gate,up,down}_proj.blkt holds one projection of a layer's experts: E slices of rows x cols weights. A row is
cut into blocks of T weights; block b is a stream of n = K T + L - K bits from bit b n of the row's 32-bit words
(little-endian, plus one spare word). Weight t of a block reads the L-bit window s at bit K t, and its value is
scale * q(s), q(s) = (sum of the 4 bytes of 34038481 s + 76625530) - 510, with one BF16 scale a block.

The loader (nanoseedlm_qwen4.py) puts a TrellisSwitchLinear in place of each switch_mlp projection that has a bank beside
it (add_banks), as ns_add_banks does in the C engines: up to MATVEC_MAX_PAIRS (token, expert) pairs a fused gather
mat-vec, above it the pairs sorted by expert through a tile GEMM whose weight tiles are decoded in threadgroup memory.
No dequantized expert tensor is built.
"""
import os
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from nanoseedlm_seeds import BK, BM, BN, MATVEC_MAX_PAIRS, SeedSwitchGLU, _module, _sort_pairs, _unsort, expert_tiles

# MoeBlkHeader (nslm/moe.h): magic, rows, cols, n_experts, n_seeds = L, n_exp = K, refit = T, exp_delta[3], n0, seconds
HEADER = struct.Struct("<8s3I3I3ifd")
PROJ = ("gate_proj", "up_proj", "down_proj")


def row_words(cols: int, T: int, L: int, K: int) -> int:
    """32-bit words of one row's stream, the spare word included (nslm_tq_row_words)."""
    return ((cols + T - 1) // T * (K * T + L - K) + 31) // 32 + 1


def read_bank(path: str) -> dict:
    """A .blkt file: its header fields, words (E, rows, nw) uint32 and scales (E, rows, nb) uint16 (BF16 bits)."""
    with open(path, "rb") as f:
        raw = f.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError(f"{path}: not a trellis bank")
        magic, rows, cols, e, L, K, T = HEADER.unpack(raw)[:7]
        if magic != b"NSLMBLKT" or not (T and 0 < K < L <= 16):
            raise ValueError(f"{path}: not a trellis bank")
        nw, nb = row_words(cols, T, L, K), (cols + T - 1) // T
        if os.fstat(f.fileno()).st_size != HEADER.size + 8 * e + (4 * nw + 2 * nb) * rows * e:
            raise ValueError(f"{path}: not a trellis bank (size)")
        f.seek(HEADER.size + 8 * e)   # past rel[E] and wrel[E]
        words = np.fromfile(f, np.uint32, e * rows * nw).reshape(e, rows, nw)
        scales = np.fromfile(f, np.uint16, e * rows * nb).reshape(e, rows, nb)
    return dict(rows=rows, cols=cols, experts=e, L=L, K=K, T=T, words=words, scales=scales)


_HEADER = """
inline float tq_bf16(uint16_t h) { return as_type<float>(uint(h) << 16); }
inline int tq_code(uint s) {   // q(s): the 1MAD code
    const uint x = s * 34038481u + 76625530u;
    return int((x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24)) - 510;
}
inline float tq_unit(uint s) {   // (q(s) + 510) / 255, within ~1e-7 of the exact code
    const float4 u = unpack_unorm4x8_to_float(s * 34038481u + 76625530u);
    return (u.x + u.y) + (u.z + u.w);
}
// The 64 stream bits from bit o of a row (u0 the low 32), reading the third word only when [o, o + span) reaches it.
inline void tq_bits(const device uint* row, uint o, uint span, thread uint& u0, thread uint& u1) {
    const device uint* p = row + (o >> 5);
    const uint sh = o & 31u, w0 = p[0], w1 = p[1], w2 = sh + span > 64u ? p[2] : 0u;
    u0 = uint(((ulong(w1) << 32) | ulong(w0)) >> sh);
    u1 = uint(((ulong(w2) << 32) | ulong(w1)) >> sh);
}
template <int L> inline uint tq_state(uint u0, uint u1, int off) {   // the L-bit window at bit off (off + L <= 64)
    return (off + L <= 32 ? u0 >> uint(off) : off >= 32 ? u1 >> uint(off - 32) : (u0 >> uint(off)) | (u1 << uint(32 - off))) &
           ((1u << L) - 1u);
}
"""

# One thread a weight, float32: scale * q, as nslm_tq_decode_row (the tests' reference; inference never builds it).
_DECODE = """
const uint g = thread_position_in_grid.x;
if (g >= TOTAL) return;
const uint r = g / C, j = g % C, b = j / TB;
uint u0, u1;
tq_bits(words + ulong(r) * NW, b * NBITS + KQ * (j % TB), L, u0, u1);
out[g] = tq_bf16(scales[ulong(r) * NB + b]) * float(tq_code(u0 & ((1u << L) - 1u)));
"""

# out[m] = W[indices[m]] @ x[m]: a simdgroup takes 4 rows of pair m; lane l the W = T / 32 weights from l W of each
# block (one 64-bit window read), each at q = 255 (sum of unpack_unorm4x8) - 510 with the 255 and the 510 applied once a
# block, as the Metal engine's MF_TCQ gathers.
_MATVEC = """
constexpr int W = TB / 32;
const uint lane = thread_position_in_threadgroup.x;
const uint r0 = thread_position_in_grid.y * 4;
const uint m = thread_position_in_grid.z;
const uint e = indices[m];
const device T* xm = x + ulong(m) * C;
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (int b = 0; b < NB; ++b) {
    float xv[W];
    float xs = 0.0f;
    for (int i = 0; i < W; ++i) { xv[i] = float(xm[b * TB + int(lane) * W + i]); xs += xv[i]; }
    for (uint r = 0; r < 4; ++r) {
        if (r0 + r >= uint(N)) break;
        const ulong row = ulong(e) * N + r0 + r;
        uint u0, u1;
        tq_bits(words + row * NW, uint(b) * NBITS + uint(KQ * W) * lane, KQ * (W - 1) + L, u0, u1);
        float s = 0.0f;
        for (int i = 0; i < W; ++i) s += tq_unit(tq_state<L>(u0, u1, KQ * i)) * xv[i];
        acc[r] += tq_bf16(scales[row * NB + b]) * (255.0f * s - 510.0f * xs);
    }
}
for (uint r = 0; r < 4; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0 && r0 + r < uint(N)) out[ulong(m) * N + r0 + r] = static_cast<T>(v);
}
"""

# One threadgroup per (BM-row block, tile of one expert's sorted pairs; tiles / starts cumulative per expert, from
# nanoseedlm_seeds.expert_tiles): float32 weight tiles decoded exactly (8 weights of one block a thread), f32 simdgroup
# MMA (4 simdgroups of 32 rows x 32 pairs), as nanoseedlm_seeds' seeded GEMM.
_GEMM = """
const uint tid = thread_index_in_threadgroup, sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
const int t = int(threadgroup_position_in_grid.y);
if (t >= tiles[E]) return;
int lo = 0, hi = E;
while (hi - lo > 1) { int mid = (lo + hi) / 2; if (tiles[mid] <= t) lo = mid; else hi = mid; }
const uint e = lo;
const int p0 = starts[e] + (t - tiles[e]) * BN, count = min(BN, starts[e + 1] - p0);
const int r0 = int(threadgroup_position_in_grid.x) * BM;
constexpr int TH = 128, KP = BK + 4;
threadgroup float Wt[BM * KP];   // [row][k]
threadgroup float Xt[BN * KP];   // [pair][k]
simdgroup_float8x8 acc[4][4];
_Pragma("clang loop unroll(full)") for (int i = 0; i < 4; ++i)
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
const int sr = int(sgi / 2) * 32, sn = int(sgi % 2) * 32;
for (int k0 = 0; k0 < C; k0 += BK) {
    for (int b = int(tid); b < BM * BK / 8; b += TH) {   // weights: 8 of one block per thread
        const int rr = b / (BK / 8), c0 = k0 + (b % (BK / 8)) * 8;
        threadgroup float* dst = Wt + rr * KP + (c0 - k0);
        if (r0 + rr < N) {
            const ulong row = ulong(e) * N + r0 + rr;
            const int blk = c0 / TB;
            uint u0, u1;
            tq_bits(words + row * NW, uint(blk) * NBITS + uint(KQ * (c0 % TB)), KQ * 7 + L, u0, u1);
            const float sc = tq_bf16(scales[row * NB + blk]);
            for (int c = 0; c < 8; ++c) dst[c] = sc * float(tq_code(tq_state<L>(u0, u1, KQ * c)));
        } else for (int c = 0; c < 8; ++c) dst[c] = 0.0f;
    }
    for (int b = int(tid); b < BN * BK / 8; b += TH) {   // inputs: 8 per thread
        const int n = b / (BK / 8), kk = (b % (BK / 8)) * 8;
        const device T* xr = x + ulong(p0 + n) * C + k0 + kk;
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
        if (r < N && n < count) out[ulong(p0 + n) * N + r] = static_cast<T>(v[0]);
        if (r < N && n + 1 < count) out[ulong(p0 + n + 1) * N + r] = static_cast<T>(v[1]);
    }
}
"""

_kernels: dict = {}


def _kernel(name: str, inputs: list[str], source: str):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(
            name=f"nanoseedlm_tq_{name}", input_names=inputs, output_names=["out"], source=source, header=_HEADER
        )
    return _kernels[name]


def _consts(words, scales, cols: int, L: int, K: int, T: int) -> list:
    return [("N", words.shape[1]), ("C", cols), ("NW", words.shape[2]), ("NB", scales.shape[2]), ("TB", T), ("KQ", K),
            ("L", L), ("NBITS", K * T + L - K)]


def decode(words, scales, cols: int, L: int, K: int, T: int) -> mx.array:
    """All weights as float32, shape (experts, rows, cols), as nslm_tq_decode_row (for the tests)."""
    e, n, _ = words.shape
    total = e * n * cols
    return _kernel("decode", ["words", "scales"], _DECODE)(
        inputs=[words, scales],
        template=[("TOTAL", total)] + _consts(words, scales, cols, L, K, T),
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(e, n, cols)],
        output_dtypes=[mx.float32],
    )[0]


def gather_matvec(x, indices, words, scales, cols: int, L: int, K: int, T: int) -> mx.array:
    """out[m] = W[indices[m]] @ x[m] for x (M, cols), indices (M,)."""
    n, m = words.shape[1], x.shape[0]
    return _kernel("matvec", ["x", "indices", "words", "scales"], _MATVEC)(
        inputs=[x, indices.astype(mx.uint32), words, scales],
        template=[("T", x.dtype)] + _consts(words, scales, cols, L, K, T),
        grid=(32, (n + 3) // 4, m),
        threadgroup=(32, 8, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def gather_gemm(x, sorted_indices, words, scales, cols: int, L: int, K: int, T: int, tiles=None) -> mx.array:
    """out[m] = W[sorted_indices[m]] @ x[m] for x (M, cols) and expert indices (M,) in ascending order."""
    e, n, _ = words.shape
    m = x.shape[0]
    sorted_indices = sorted_indices.astype(mx.uint32)
    tiles, starts = expert_tiles(sorted_indices, e) if tiles is None else tiles
    return _kernel("gemm", ["x", "tiles", "starts", "words", "scales"], _GEMM)(
        inputs=[x, tiles, starts, words, scales],
        template=[("T", x.dtype), ("E", e), ("BM", BM), ("BN", BN), ("BK", BK)] + _consts(words, scales, cols, L, K, T),
        grid=((n + BM - 1) // BM * 128, (m + BN - 1) // BN + e, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


class TrellisSwitchLinear(nn.Module):
    """mlx-lm's SwitchLinear with the trellis-coded weights of one bank (read_bank's dict)."""

    def __init__(self, bank: dict):
        super().__init__()
        L, K, T, cols = bank["L"], bank["K"], bank["T"], bank["cols"]
        if T % 32 or T > 320 or cols % T or cols % BK or K * (T // 32 - 1) + L > 64 or 7 * K + L > 64:
            raise ValueError(f"trellis bank with L {L}, K {K}, T {T}, {cols} columns: not supported by the MLX kernels")
        self.words = mx.array(bank["words"])
        self.block_scales = mx.array(bank["scales"])
        self.input_dims, self.output_dims, self.num_experts = cols, bank["rows"], bank["experts"]
        self.L, self.K, self.T = L, K, T

    def _tq(self):
        return self.words, self.block_scales, self.input_dims, self.L, self.K, self.T

    def to_quantized(self, *args, **kwargs):   # mlx-vlm's quantize pass may name this path (the MTP head's config entry)
        return self

    def gemm(self, xs, sorted_indices, tiles=None):
        return gather_gemm(xs, sorted_indices, *self._tq(), tiles)

    def matvec(self, x, indices):
        rows = x.shape[-2]
        batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
        xs = mx.broadcast_to(x, batch + x.shape[-2:]).reshape(-1, self.input_dims)
        idx = mx.broadcast_to(indices, batch).reshape(-1).astype(mx.uint32)
        if rows > 1:
            idx = mx.repeat(idx, rows)
        return gather_matvec(xs, idx, *self._tq()).reshape(batch + (rows, self.output_dims))

    def __call__(self, x, indices, sorted_indices=False):
        if sorted_indices:   # rows already in expert order, one per index (mlx-lm's _gather_sort; oMLX prefill)
            xs, idx = x.reshape(-1, self.input_dims), indices.reshape(-1).astype(mx.uint32)
            y = gather_matvec(xs, idx, *self._tq()) if idx.size <= MATVEC_MAX_PAIRS else self.gemm(xs, idx)
            return y.reshape(x.shape[:-1] + (self.output_dims,))
        if indices.size * x.shape[-2] <= MATVEC_MAX_PAIRS:
            return self.matvec(x, indices)
        if x.shape[-2] != 1:
            raise ValueError("TrellisSwitchLinear: one row per token or pair above the mat-vec limit")
        # one row per (token, expert) pair: x may hold one per token (..., 1, 1, cols) or one per pair (..., k, 1, cols)
        xs = mx.broadcast_to(x, indices.shape + x.shape[-2:]).reshape(-1, self.input_dims)
        idx = indices.reshape(-1).astype(mx.uint32)
        order = mx.argsort(idx)
        return _unsort(self.gemm(xs[order], idx[order]), order, indices.shape).reshape(indices.shape + (1, self.output_dims))


class TrellisSwitchGLU(SeedSwitchGLU):
    """SeedSwitchGLU's paths (the mat-vec for few pairs, else expert-sorted pairs through the tile GEMM) on trellis
    projections (a projection may also be a SeedSwitchLinear)."""

    def _sorted(self, x, indices):
        xs, idx, order = _sort_pairs(x, indices)
        up = self.up_proj
        tiles = expert_tiles(idx, up.num_experts if isinstance(up, TrellisSwitchLinear) else up.seeds.shape[0])
        h = self.activation(self.up_proj.gemm(xs, idx, tiles), self.gate_proj.gemm(xs, idx, tiles))
        return self.down_proj.gemm(h, idx, tiles), order


def add_banks(model: nn.Module, weights: dict, folder: str, n_layers: int) -> dict:
    """Each L{l}_{proj}.blkt in folder replaces layer l's switch_mlp projection (whatever weights the layer has, as
    ns_add_banks does); l = n_layers is the MTP head's (mtp.layers.0), used when the model has an MTP module. The
    projection's tensors leave weights and its streams join them (for the strict load). Returns weights."""
    glus = set()
    for layer in range(n_layers + 1):
        if layer < n_layers:
            glu = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        elif "mtp" in model:
            glu = "mtp.layers.0.mlp.switch_mlp"
        else:
            continue
        for p in PROJ:
            path = os.path.join(folder, f"L{layer}_{p}.blkt")
            if not os.path.exists(path):
                continue
            prefix = f"{glu}.{p}"
            for k in [k for k in weights if k.startswith(prefix + ".")]:
                del weights[k]
            lin = TrellisSwitchLinear(read_bank(path))
            model.update_modules(tree_unflatten([(prefix, lin)]))
            weights[prefix + ".words"], weights[prefix + ".block_scales"] = lin.words, lin.block_scales
            glus.add(glu)
    for glu in glus:
        _module(model, glu).__class__ = TrellisSwitchGLU
    return weights
