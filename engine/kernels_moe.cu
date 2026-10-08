// engine/kernels_moe.cu - CUDA kernels of the MoVA engine (engine/mova_cuda.c), ports of engine/kernels_moe.metal with
// the same arithmetic and the same BF16 rounding points (activations are f32 buffers of BF16-rounded values, rounded
// wherever the MLX reference implementation produces a BF16 tensor).  A Metal simdgroup is a warp; a threadgroup is a
// block.  Written in C style: one __device__ body per kernel, specialised per weight format by the FMT_KERNELS macro.
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "kernels_cuda.h"

#define FULL 0xFFFFFFFFu
#undef INFINITY   // MSVC defines it as an overflowing constant expression
#define INFINITY __int_as_float(0x7f800000)

static __device__ __forceinline__ float bfr(float x) {   // round to BF16 (nearest even), as f32
    uint32_t u = __float_as_uint(x);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}
static __device__ __forceinline__ float bf(uint16_t h) { return __uint_as_float((uint32_t) h << 16); }
static __device__ __forceinline__ uint16_t tobf(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static __device__ __forceinline__ float silu_bf(float g) {   // nn.silu on a BF16 tensor: each op rounded to BF16
    const float s = bfr(1.0f / (1.0f + expf(-g)));
    return bfr(g * s);
}
static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
static __device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
static __device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static __device__ __forceinline__ float sum8(float4 a, float4 b) { return (a.x + b.x) + (a.y + b.y) + (a.z + b.z) + (a.w + b.w); }
// the exact value of a 4-bit field as f32, from the 2^23 trick (no convert)
static __device__ __forceinline__ float u4f(uint32_t w, int sh) { return __uint_as_float(0x4B000000u | ((w >> sh) & 15u)) - 8388608.0f; }
static __device__ __forceinline__ float s4f(uint32_t w, int sh) {   // signed 4-bit field: (u ^ 8) - 8
    return __uint_as_float(0x4B000000u | (((w >> sh) & 15u) ^ 8u)) - 8388616.0f;
}

// ---- seed blocks ------------------------------------------------------------------------------------------------------
// SEED4 (P = 3, 24 states) and SEED4P4 (P = 4, 32 states).  State k of seed s = bits k .. k + 15 of the stream
// s | G[s] << 16 (G: lfsr_stream24 / lfsr_stream32), as v_k = 1 + V_k / 2^16 (exact: the state under the exponent of 1).
static __device__ __forceinline__ float st1(uint64_t st, int k) {
    return __uint_as_float(0x3F800000u | (((uint32_t) (st >> k) & 0xFFFFu) << 7));
}
// One P = 4 block without R32: 2^(e+16) (sum_p q_p d_p + (sum_p q_p) xm), d_p = x . v_p, xm = -1.5 sum(x).
static __device__ __forceinline__ float seed4_dot(uint32_t s, uint32_t g, uint32_t cw, int e, float4 x0, float4 x1, float xm) {
    const uint64_t st = (uint64_t) s | ((uint64_t) g << 16);
    const float q0 = s4f(cw, 0), q1 = s4f(cw, 4), q2 = s4f(cw, 8), q3 = s4f(cw, 12);
    const float e2 = __uint_as_float((uint32_t) (e + 127 + 16) << 23);
    const float d0 = dot4(x0, make_float4(st1(st, 1), st1(st, 5), st1(st, 9), st1(st, 13))) +
                     dot4(x1, make_float4(st1(st, 17), st1(st, 21), st1(st, 25), st1(st, 29)));
    const float d1 = dot4(x0, make_float4(st1(st, 2), st1(st, 6), st1(st, 10), st1(st, 14))) +
                     dot4(x1, make_float4(st1(st, 18), st1(st, 22), st1(st, 26), st1(st, 30)));
    const float d2 = dot4(x0, make_float4(st1(st, 3), st1(st, 7), st1(st, 11), st1(st, 15))) +
                     dot4(x1, make_float4(st1(st, 19), st1(st, 23), st1(st, 27), st1(st, 31)));
    const float d3 = dot4(x0, make_float4(st1(st, 4), st1(st, 8), st1(st, 12), st1(st, 16))) +
                     dot4(x1, make_float4(st1(st, 20), st1(st, 24), st1(st, 28), st1(st, 32)));
    return e2 * ((((q0 * d0 + q1 * d1) + q2 * d2) + q3 * d3) + (((q0 + q1) + q2) + q3) * xm);
}
// One P = 3 block (nibble word: bits 0-3 exponent code, then q0 q1 q2) without R32.
static __device__ __forceinline__ float seed3_dot(uint32_t s, uint32_t g, uint32_t nb, int ebias, float4 x0, float4 x1, float xm) {
    const uint64_t st = (uint64_t) s | ((uint64_t) g << 16);
    const float q0 = s4f(nb, 4), q1 = s4f(nb, 8), q2 = s4f(nb, 12);
    const float e2 = __uint_as_float((uint32_t) (ebias + (int) (nb & 15u) + 127 + 16) << 23);
    const float d0 = dot4(x0, make_float4(st1(st, 1), st1(st, 4), st1(st, 7), st1(st, 10))) +
                     dot4(x1, make_float4(st1(st, 13), st1(st, 16), st1(st, 19), st1(st, 22)));
    const float d1 = dot4(x0, make_float4(st1(st, 2), st1(st, 5), st1(st, 8), st1(st, 11))) +
                     dot4(x1, make_float4(st1(st, 14), st1(st, 17), st1(st, 20), st1(st, 23)));
    const float d2 = dot4(x0, make_float4(st1(st, 3), st1(st, 6), st1(st, 9), st1(st, 12))) +
                     dot4(x1, make_float4(st1(st, 15), st1(st, 18), st1(st, 21), st1(st, 24)));
    return e2 * (((q0 * d0 + q1 * d1) + q2 * d2) + ((q0 + q1) + q2) * xm);
}
static __device__ __forceinline__ int seed4_ecode(const uint8_t* EN, uint32_t k) { return (EN[k >> 1] >> ((k & 1) * 4)) & 15; }

// The 8 weights of block j of local row r, dequantized to BF16 values, for BF16 / Q8 / Q4 (as mx.dequantize).
static __device__ __forceinline__ void wblock(int fmt, const WSlice* w, int r, int K, int j, float* o) {
    if (fmt == MF_BF16) {
        const uint4 v = *(const uint4*) ((const uint16_t*) w->p[0] + (size_t) r * K + (size_t) j * 8);
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
        for (int i = 0; i < 4; ++i) { o[2 * i] = __uint_as_float(u[i] << 16); o[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u); }
        return;
    }
    const size_t gi = ((size_t) r * K + (size_t) j * 8) / 64;
    const float s = bf(((const uint16_t*) w->p[1])[gi]), b = bf(((const uint16_t*) w->p[2])[gi]);
    if (fmt == MF_Q8) {
        const uint2 q = *(const uint2*) ((const uint32_t*) w->p[0] + ((size_t) r * K) / 4 + (size_t) j * 2);
        for (int i = 0; i < 4; ++i) { o[i] = bfr(s * (float) ((q.x >> (8 * i)) & 255u) + b); o[4 + i] = bfr(s * (float) ((q.y >> (8 * i)) & 255u) + b); }
    } else {
        const uint32_t q = ((const uint32_t*) w->p[0])[((size_t) r * K) / 8 + (size_t) j];
        for (int i = 0; i < 8; ++i) o[i] = bfr(s * (float) ((q >> (4 * i)) & 15u) + b);
    }
}

// One lane's share of row r . x (one token), every format: the Metal mv_lane arithmetic.  The caller sums the lanes.
static __device__ __forceinline__ float mv_lane(int fmt, const WSlice* w, const uint32_t* G, int r, int K, const float* xb, int lane) {
    const int nbk = K / 8;
    float acc = 0;
    if (fmt == MF_SEED4P4) {
        const uint16_t* seeds = (const uint16_t*) w->p[0] + (size_t) r * nbk;
        const uint16_t* coefs = (const uint16_t*) w->p[1] + (size_t) r * nbk;
        const uint8_t* EN = (const uint8_t*) w->p[3];
        const uint32_t kb = (uint32_t) r * (uint32_t) nbk;
        if (nbk % 64 == 0) {   // two adjacent blocks per lane: 4-byte seed / coefficient loads, one exponent byte
            const uint32_t* s2 = (const uint32_t*) seeds;
            const uint32_t* c2 = (const uint32_t*) coefs;
            for (int jq = lane; jq < nbk / 2; jq += 32) {
                const uint32_t ss = s2[jq], cc = c2[jq], eb2 = EN[(kb + 2 * jq) >> 1];
                const uint32_t sa = ss & 0xFFFFu, sb = ss >> 16;
                const uint32_t ga = __ldg(G + sa), gb = __ldg(G + sb);
                const float4* xr = (const float4*) (xb + (size_t) (2 * jq) * 8);
                const float4 x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
                acc += seed4_dot(sa, ga, cc & 0xFFFFu, w->eb + (int) (eb2 & 15u), x0, x1, -1.5f * sum8(x0, x1));
                acc += seed4_dot(sb, gb, cc >> 16, w->eb + (int) ((eb2 >> 4) & 15u), x2, x3, -1.5f * sum8(x2, x3));
            }
            return acc * (1.0f / 32767.0f);
        }
        for (int j = lane; j < nbk; j += 32) {
            const uint32_t s = seeds[j];
            const float4* xr = (const float4*) (xb + (size_t) j * 8);
            const float4 x0 = xr[0], x1 = xr[1];
            acc += seed4_dot(s, __ldg(G + s), coefs[j], w->eb + seed4_ecode(EN, kb + (uint32_t) j), x0, x1, -1.5f * sum8(x0, x1));
        }
        return acc * (1.0f / 32767.0f);
    }
    if (fmt == MF_SEED4) {
        const uint16_t* seeds = (const uint16_t*) w->p[0] + (size_t) r * nbk;
        const uint16_t* nibs = (const uint16_t*) w->p[1] + (size_t) r * nbk;
        for (int j = lane; j < nbk; j += 32) {
            const uint32_t s = seeds[j];
            const float4* xr = (const float4*) (xb + (size_t) j * 8);
            const float4 x0 = xr[0], x1 = xr[1];
            acc += seed3_dot(s, __ldg(G + s), nibs[j], w->eb, x0, x1, -1.5f * sum8(x0, x1));
        }
        return acc * (1.0f / 32767.0f);
    }
    if (fmt == MF_Q4) {   // MLX's qmv form: per block s * sum(q x) + b * sum(x); two consecutive blocks per lane per step
        const uint16_t* S = (const uint16_t*) w->p[1];
        const uint16_t* B = (const uint16_t*) w->p[2];
        const uint32_t* qr = (const uint32_t*) w->p[0] + ((size_t) r * K) / 8;
        for (int j0 = lane * 2; j0 < nbk; j0 += 64) {
            const size_t gi = ((size_t) r * K + (size_t) j0 * 8) / 64;
            const float s = bf(S[gi]), b = bf(B[gi]);
            const uint2 qq = *(const uint2*) (qr + j0);
            const uint32_t qw[2] = {qq.x, qq.y};
            for (int k = 0; k < 2; ++k) {
                const float4 q0 = make_float4(u4f(qw[k], 0), u4f(qw[k], 4), u4f(qw[k], 8), u4f(qw[k], 12));
                const float4 q1 = make_float4(u4f(qw[k], 16), u4f(qw[k], 20), u4f(qw[k], 24), u4f(qw[k], 28));
                const float4* xr = (const float4*) (xb + (size_t) (j0 + k) * 8);
                const float4 x0 = xr[0], x1 = xr[1];
                acc += s * (dot4(q0, x0) + dot4(q1, x1)) + b * sum8(x0, x1);
            }
        }
        return acc;
    }
    if (fmt == MF_Q8 && nbk % 64 == 0) {   // two adjacent blocks per lane: one 16-byte load, one group's scale and bias.
        // The products add in a different order than in the generic path and in Metal (whose lane reductions differ from
        // CUDA's anyway): f32 noise within tests/test_mova_kernels.c's bound; summing per block cost 1% of decode.
        const uint16_t* S = (const uint16_t*) w->p[1];
        const uint16_t* B = (const uint16_t*) w->p[2];
        const uint4* qr = (const uint4*) ((const uint32_t*) w->p[0] + ((size_t) r * K) / 4);
#pragma unroll 2
        for (int jq = lane; jq < nbk / 2; jq += 32) {
            const size_t gi = ((size_t) r * K + (size_t) jq * 16) / 64;
            const float s = bf(S[gi]), b = bf(B[gi]);
            const uint4 q = qr[jq];
            const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
            const float4* xr = (const float4*) (xb + (size_t) jq * 16);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 xv = xr[i];
                acc += bfr(s * (float) (qw[i] & 255u) + b) * xv.x + bfr(s * (float) ((qw[i] >> 8) & 255u) + b) * xv.y +
                       bfr(s * (float) ((qw[i] >> 16) & 255u) + b) * xv.z + bfr(s * (float) (qw[i] >> 24) + b) * xv.w;
            }
        }
        return acc;
    }
    for (int j = lane; j < nbk; j += 32) {
        float wv[8];
        wblock(fmt, w, r, K, j, wv);
        const float4* xr = (const float4*) (xb + (size_t) j * 8);
        acc += dot4(make_float4(wv[0], wv[1], wv[2], wv[3]), xr[0]) + dot4(make_float4(wv[4], wv[5], wv[6], wv[7]), xr[1]);
    }
    return acc;
}

static __device__ __forceinline__ void store_y(float* yo, float s, int add) { *yo = add ? bfr(*yo + bfr(s)) : bfr(s); }

// ---- matvec: one warp per output row, lanes over the 8-weight blocks; MV_ROWS warps per block ------------------------
// Dense: T tokens (<= MV_MAXT) share each weight-row load (and each seed expansion).  Gather: grid y = pair p.
static __device__ __forceinline__ void mv_body(int fmt, MvArgs a, WSlice wd, const WSlice* ws, const float* x, float* y,
                                               const int32_t* sel, const uint32_t* G, int nt) {
    const int lane = threadIdx.x & 31, r = (int) (blockIdx.x * MV_ROWS + (threadIdx.x >> 5));
    if (r >= a.R) return;
    const int gather = sel != NULL;
    const int p0 = gather ? (int) blockIdx.y : 0;
    const WSlice w = gather ? ws[sel[p0]] : wd;
    const float* xb = x + (size_t) (gather ? p0 / a.xdiv : 0) * a.xs;
    if (nt == 1) {
        const float s = warp_sum(mv_lane(fmt, &w, G, r, a.K, xb, lane));
        if (lane == 0) store_y(y + (size_t) p0 * a.ys + r, s, a.add);
        return;
    }
    const int nbk = a.K / 8;
    float acc[MV_MAXT];
#pragma unroll
    for (int t = 0; t < MV_MAXT; ++t) acc[t] = 0;
    if (fmt == MF_SEED4P4 || fmt == MF_SEED4) {   // one state expansion per block for every token
        const uint16_t* seeds = (const uint16_t*) w.p[0] + (size_t) r * nbk;
        const uint16_t* cws = (const uint16_t*) w.p[1] + (size_t) r * nbk;
        const uint8_t* EN = (const uint8_t*) w.p[3];
        for (int j = lane; j < nbk; j += 32) {
            const uint32_t s = seeds[j], g = __ldg(G + s), cw = cws[j];
            const int e = fmt == MF_SEED4P4 ? w.eb + seed4_ecode(EN, (uint32_t) r * (uint32_t) nbk + (uint32_t) j) : w.eb;
#pragma unroll
            for (int t = 0; t < MV_MAXT; ++t) {
                if (t >= nt) break;
                const float4* xr = (const float4*) (xb + (size_t) t * a.xs + (size_t) j * 8);
                const float4 x0 = xr[0], x1 = xr[1];
                acc[t] += fmt == MF_SEED4P4 ? seed4_dot(s, g, cw, e, x0, x1, -1.5f * sum8(x0, x1))
                                            : seed3_dot(s, g, cw, e, x0, x1, -1.5f * sum8(x0, x1));
            }
        }
#pragma unroll
        for (int t = 0; t < MV_MAXT; ++t) acc[t] *= (1.0f / 32767.0f);
    } else if (fmt == MF_Q4) {
        const uint16_t* S = (const uint16_t*) w.p[1];
        const uint16_t* B = (const uint16_t*) w.p[2];
        const uint32_t* qr = (const uint32_t*) w.p[0] + ((size_t) r * a.K) / 8;
        for (int j0 = lane * 2; j0 < nbk; j0 += 64) {
            const size_t gi = ((size_t) r * a.K + (size_t) j0 * 8) / 64;
            const float s = bf(S[gi]), b = bf(B[gi]);
            const uint2 qq = *(const uint2*) (qr + j0);
            const uint32_t qw[2] = {qq.x, qq.y};
            for (int k = 0; k < 2; ++k) {
                const float4 q0 = make_float4(u4f(qw[k], 0), u4f(qw[k], 4), u4f(qw[k], 8), u4f(qw[k], 12));
                const float4 q1 = make_float4(u4f(qw[k], 16), u4f(qw[k], 20), u4f(qw[k], 24), u4f(qw[k], 28));
#pragma unroll
                for (int t = 0; t < MV_MAXT; ++t) {
                    if (t >= nt) break;
                    const float4* xr = (const float4*) (xb + (size_t) t * a.xs + (size_t) (j0 + k) * 8);
                    const float4 x0 = xr[0], x1 = xr[1];
                    acc[t] += s * (dot4(q0, x0) + dot4(q1, x1)) + b * sum8(x0, x1);
                }
            }
        }
    } else {
        for (int j = lane; j < nbk; j += 32) {
            float wv[8];
            wblock(fmt, &w, r, a.K, j, wv);
            const float4 w0 = make_float4(wv[0], wv[1], wv[2], wv[3]), w1 = make_float4(wv[4], wv[5], wv[6], wv[7]);
#pragma unroll
            for (int t = 0; t < MV_MAXT; ++t) {
                if (t >= nt) break;
                const float4* xr = (const float4*) (xb + (size_t) t * a.xs + (size_t) j * 8);
                acc[t] += dot4(w0, xr[0]) + dot4(w1, xr[1]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MV_MAXT; ++t) {
        if (t >= nt) break;
        const float s = warp_sum(acc[t]);
        if (lane == 0) store_y(y + (size_t) t * a.ys + r, s, a.add);
    }
}

// Routed experts' gate and up for one (token, expert) pair per grid row, and the SwiGLU:
// a[p][r] = bf16(silu(bf16(gate . x)) * bf16(up . x)).
static __device__ __forceinline__ void mv_gu_body(int fmt, MvArgs a, const WSlice* wg, const WSlice* wu, const float* x, float* y,
                                                  const int32_t* sel, const uint32_t* G) {
    const int lane = threadIdx.x & 31, r = (int) (blockIdx.x * MV_ROWS + (threadIdx.x >> 5));
    if (r >= a.R) return;
    const int p0 = (int) blockIdx.y, slice = sel[p0];
    const float* xb = x + (size_t) (p0 / a.xdiv) * a.xs;
    const WSlice g = wg[slice], u = wu[slice];
    const float gs = warp_sum(mv_lane(fmt, &g, G, r, a.K, xb, lane));
    const float us = warp_sum(mv_lane(fmt, &u, G, r, a.K, xb, lane));
    if (lane == 0) y[(size_t) p0 * a.ys + r] = bfr(silu_bf(bfr(gs)) * bfr(us));
}

// ---- GEMM (prefill): Y[n][r] = sum_k W[r][k] X[n][k] for a 32-row x 32-column tile, f32 products and sums -----------
// The weight tile is dequantized into shared memory, one 8-weight block per thread per K step (seeds: centred states,
// isum * fl(R32 2^e), as the Metal tile_weights).  Dense: tile (row block, token block).  Grouped: (row block, tile
// table entry); the entry's pairs are perm[start .. start + count), reading input row pair / xdiv, writing row pair.
static __device__ __forceinline__ float cstate(uint64_t st, int k) {   // state k - 32768, exact
    return __uint_as_float(0x4B000000u | ((uint32_t) (st >> k) & 0xFFFFu)) - 8421376.0f;
}
static __device__ __forceinline__ void tile_weights(int fmt, const WSlice* w, const uint32_t* G, int r, int K, int j, float* dst) {
    float o[8];
    const int nbk = K / 8;
    if (fmt == MF_SEED4P4) {
        const uint32_t k = (uint32_t) r * (uint32_t) nbk + (uint32_t) j;
        const uint32_t s = ((const uint16_t*) w->p[0])[k], cw = ((const uint16_t*) w->p[1])[k];
        const uint64_t st = (uint64_t) s | ((uint64_t) __ldg(G + s) << 16);
        const int ci = (int) cw;
        const float q0 = (float) ((ci << 28) >> 28), q1 = (float) ((ci << 24) >> 28), q2 = (float) ((ci << 20) >> 28),
                    q3 = (float) ((ci << 16) >> 28);
        const float sc = __uint_as_float((uint32_t) (w->eb + seed4_ecode((const uint8_t*) w->p[3], k) + 127) << 23) * (1.0f / 32767.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            o[i] = __fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(cstate(st, 4 * i + 1), q0), __fmul_rn(cstate(st, 4 * i + 2), q1)),
                                                 __fmul_rn(cstate(st, 4 * i + 3), q2)),
                                       __fmul_rn(cstate(st, 4 * i + 4), q3)),
                             sc);
    } else if (fmt == MF_SEED4) {
        const uint32_t k = (uint32_t) r * (uint32_t) nbk + (uint32_t) j;
        const uint32_t s = ((const uint16_t*) w->p[0])[k];
        const int nb = (int) ((const uint16_t*) w->p[1])[k];
        const uint64_t st = (uint64_t) s | ((uint64_t) __ldg(G + s) << 16);
        const float q0 = (float) ((nb << 24) >> 28), q1 = (float) ((nb << 20) >> 28), q2 = (float) ((nb << 16) >> 28);
        const float sc = __uint_as_float((uint32_t) (w->eb + (nb & 15) + 127) << 23) * (1.0f / 32767.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            o[i] = __fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(cstate(st, 3 * i + 1), q0), __fmul_rn(cstate(st, 3 * i + 2), q1)),
                                       __fmul_rn(cstate(st, 3 * i + 3), q2)),
                             sc);
    } else wblock(fmt, w, r, K, j, o);
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = o[i];
}

// Tensor-core GEMM (prefill): Y[n][r] = sum_k W[r][k] X[n][k] for a 64-row x 64-column tile.  BF16 mma.sync with f32
// accumulation: the activations are BF16 values (exact).  BF16 / Q8 / Q4 weights dequantize to BF16 values (exact).
// Seed weights (f32 in the matvec and in the Metal GEMM) are rounded to BF16 here (NT = 1), or, in the k_mm3 kernels
// (NSLM_SEED_GEMM_F32), carried as NT = 3 BF16 terms hi + mid + lo whose sum is the f32 value exactly.  Measured on the
// real model against BF16 reference log-probs, the exact weights change neither KLD (held-out 0.0261 vs 0.0260, standard
// error 0.0005) nor NLL, and cost 15% of 4k prefill, so BF16 is the default.  Four warps, each a 32 x 32 sub-tile.
// Dense: tile (row block, token block).  Grouped: (row block, tile table entry); the entry's pairs are
// perm[start .. start + count), reading input row pair / xdiv, writing row pair.  ntiles (device) bounds grouped grids.
#define SLD (MMT_BK + 8)   // shared row stride, BF16 elements (conflict-free fragment loads)
static __device__ __forceinline__ uint32_t pack_bf2(float a, float b) { return (uint32_t) tobf(a) | ((uint32_t) tobf(b) << 16); }
static __device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int NT>   // weight terms (shared memory and MMAs scale with it: a template, so NT = 1 keeps its occupancy)
static __device__ __forceinline__ void mm_body(int fmt, MmArgs a, WSlice wd, const WSlice* ws, const float* X, float* Y,
                                               const int32_t* perm, const MmTile* tiles, const int32_t* ntiles,
                                               const uint32_t* G) {
    __shared__ __align__(16) uint16_t Ws[NT][MMT_BM][SLD];
    __shared__ __align__(16) uint16_t Xs[MMT_BN][SLD];
    __shared__ int srow[MMT_BN], drow[MMT_BN];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int grouped = tiles != NULL;
    const int nterm = NT;
    if (grouped && ntiles && (int) blockIdx.y >= *ntiles) return;
    const int r0 = (int) blockIdx.x * MMT_BM;
    int start, count;
    WSlice wt;
    if (grouped) { const MmTile tl = tiles[blockIdx.y]; wt = ws[tl.slice]; start = tl.start; count = tl.count; }
    else { wt = wd; start = (int) blockIdx.y * MMT_BN; count = min(MMT_BN, a.T - start); }
    if (tid < MMT_BN) {
        const int ok = tid < count;
        srow[tid] = !ok ? -1 : grouped ? perm[start + tid] / a.xdiv : start + tid;
        drow[tid] = !ok ? -1 : grouped ? perm[start + tid] : start + tid;
    }
    __syncthreads();
    const int wm = (w >> 1) * 32, wn = (w & 1) * 32;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;
    for (int k0 = 0; k0 < a.K; k0 += MMT_BK) {
        // weights: 64 rows x 4 blocks of 8, two blocks per thread
        for (int b = tid; b < MMT_BM * 4; b += 128) {
            const int rr = b >> 2, jb = b & 3, row = r0 + rr;
            float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (row < a.R) tile_weights(fmt, &wt, G, row, a.K, k0 / 8 + jb, f);
            for (int q = 0; q < nterm; ++q) {   // f = hi + mid + lo: each term the BF16 value of what remains (exact)
                uint16_t h[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { h[i] = tobf(f[i]); f[i] -= __uint_as_float((uint32_t) h[i] << 16); }
                *(uint4*) &Ws[q][rr][jb * 8] = make_uint4(h[0] | (uint32_t) h[1] << 16, h[2] | (uint32_t) h[3] << 16,
                                                          h[4] | (uint32_t) h[5] << 16, h[6] | (uint32_t) h[7] << 16);
            }
        }
        // inputs: 64 columns x 32 k, 16 per thread
        {
            const int n = tid >> 1, h = (tid & 1) * 16, src = srow[n];
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
            if (src >= 0) {
                const float4* xr = (const float4*) (X + (size_t) src * a.xs + k0 + h);
                const float4 x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
                v0 = make_uint4(pack_bf2(x0.x, x0.y), pack_bf2(x0.z, x0.w), pack_bf2(x1.x, x1.y), pack_bf2(x1.z, x1.w));
                v1 = make_uint4(pack_bf2(x2.x, x2.y), pack_bf2(x2.z, x2.w), pack_bf2(x3.x, x3.y), pack_bf2(x3.z, x3.w));
            }
            *(uint4*) &Xs[n][h] = v0;
            *(uint4*) &Xs[n][h + 8] = v1;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < MMT_BK; kk += 16) {
            uint32_t af[2][4], bfr2[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int nn = wn + nt * 8 + g;
                bfr2[nt][0] = *(const uint32_t*) &Xs[nn][kk + t4 * 2];
                bfr2[nt][1] = *(const uint32_t*) &Xs[nn][kk + t4 * 2 + 8];
            }
            for (int q = 0; q < nterm; ++q) {   // the largest term first
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int rr = wm + mt * 16 + g;
                    af[mt][0] = *(const uint32_t*) &Ws[q][rr][kk + t4 * 2];
                    af[mt][1] = *(const uint32_t*) &Ws[q][rr + 8][kk + t4 * 2];
                    af[mt][2] = *(const uint32_t*) &Ws[q][rr][kk + t4 * 2 + 8];
                    af[mt][3] = *(const uint32_t*) &Ws[q][rr + 8][kk + t4 * 2 + 8];
                }
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) mma_bf16(acc[mt][nt], af[mt], bfr2[nt]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int rr = wm + mt * 16 + g + (q >> 1) * 8, cc = wn + nt * 8 + t4 * 2 + (q & 1), row = r0 + rr, dst = drow[cc];
                if (dst < 0 || row >= a.R) continue;
                store_y(Y + (size_t) dst * a.ys + row, acc[mt][nt][q], a.add);
            }
}

// ---- grouping (prefill): selections -> pairs by expert, tile table -----------------------------------------------------
// work = ntiles buffer: [0] tile count, [1 .. 129] pair offsets per expert, [130 .. 258] tile offsets per expert.
__global__ void __launch_bounds__(1024) k_bucket_count(const int32_t* inds, int count, int n, int32_t* work) {
    __shared__ int cnt[128];
    if (threadIdx.x < 128) cnt[threadIdx.x] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < count; i += 1024) atomicAdd(&cnt[inds[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, to = 0;
        for (int x = 0; x < n; ++x) {
            work[1 + x] = o;
            work[130 + x] = to;
            o += cnt[x];
            to += (cnt[x] + MMT_BN - 1) / MMT_BN;
        }
        work[1 + n] = o;
        work[130 + n] = to;
        work[0] = to;
    }
}
// One block per expert: its pairs in ascending order (a block-wide prefix of matches), and its tiles.
__global__ void __launch_bounds__(1024) k_bucket_place(const int32_t* inds, int count, const int32_t* work, int32_t* perm, MmTile* tiles) {
    __shared__ int wsum[32];
    __shared__ int base;
    const int x = (int) blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    if (tid == 0) base = work[1 + x];
    __syncthreads();
    for (int c0 = 0; c0 < count; c0 += 1024) {
        const int i = c0 + tid, f = i < count && inds[i] == x;
        const uint32_t m = __ballot_sync(FULL, f);
        if (lane == 0) wsum[w] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
        for (int k = 0; k < 32; ++k) { if (k < w) before += wsum[k]; total += wsum[k]; }
        if (f) perm[base + before + __popc(m & ((1u << lane) - 1u))] = i;
        __syncthreads();
        if (tid == 0) base += total;
        __syncthreads();
    }
    const int o = work[1 + x], cnt = work[2 + x] - o, t0 = work[130 + x];
    for (int t = tid; t * MMT_BN < cnt; t += 1024) {
        MmTile tl = {x, o + t * MMT_BN, min(MMT_BN, cnt - t * MMT_BN), 0};
        tiles[t0 + t] = tl;
    }
}

// One kernel per weight format for each format-generic body.
#define FMT_KERNELS(F, SUF)                                                                                              \
    __global__ void __launch_bounds__(32 * MV_ROWS) k_mv_##SUF(MvArgs a, WSlice wd, const WSlice* ws, const float* x,   \
                                                               float* y, const int32_t* sel, const uint32_t* G, int nt) { \
        mv_body(F, a, wd, ws, x, y, sel, G, nt);                                                                         \
    }                                                                                                                    \
    __global__ void __launch_bounds__(32 * MV_ROWS) k_mv_gu_##SUF(MvArgs a, const WSlice* wg, const WSlice* wu,         \
                                                                  const float* x, float* y, const int32_t* sel,          \
                                                                  const uint32_t* G) {                                   \
        mv_gu_body(F, a, wg, wu, x, y, sel, G);                                                                          \
    }                                                                                                                    \
    __global__ void __launch_bounds__(128) k_mm_##SUF(MmArgs a, WSlice wd, const WSlice* ws, const float* X, float* Y,  \
                                                      const int32_t* perm, const MmTile* tiles, const int32_t* ntiles,   \
                                                      const uint32_t* G) {                                               \
        mm_body<1>(F, a, wd, ws, X, Y, perm, tiles, ntiles, G);                                                          \
    }
FMT_KERNELS(MF_BF16, bf16)
FMT_KERNELS(MF_SEED4, seed4)
FMT_KERNELS(MF_Q8, q8)
FMT_KERNELS(MF_Q4, q4)
FMT_KERNELS(MF_SEED4P4, seed4p4)

typedef void (*MvKernel)(MvArgs, WSlice, const WSlice*, const float*, float*, const int32_t*, const uint32_t*, int);
typedef void (*MvGuKernel)(MvArgs, const WSlice*, const WSlice*, const float*, float*, const int32_t*, const uint32_t*);
typedef void (*MmKernel)(MmArgs, WSlice, const WSlice*, const float*, float*, const int32_t*, const MmTile*, const int32_t*,
                         const uint32_t*);
static const MvKernel MV_K[5] = {k_mv_bf16, k_mv_seed4, k_mv_q8, k_mv_q4, k_mv_seed4p4};
static const MvGuKernel MVGU_K[5] = {k_mv_gu_bf16, k_mv_gu_seed4, k_mv_gu_q8, k_mv_gu_q4, k_mv_gu_seed4p4};
static const MmKernel MM_K[5] = {k_mm_bf16, k_mm_seed4, k_mm_q8, k_mm_q4, k_mm_seed4p4};
template <int F>   // the seed formats with exact (three-term) weights
__global__ void __launch_bounds__(128) k_mm3(MmArgs a, WSlice wd, const WSlice* ws, const float* X, float* Y,
                                             const int32_t* perm, const MmTile* tiles, const int32_t* ntiles, const uint32_t* G) {
    mm_body<3>(F, a, wd, ws, X, Y, perm, tiles, ntiles, G);
}
static const MmKernel MM3_K[5] = {k_mm_bf16, k_mm3<MF_SEED4>, k_mm_q8, k_mm_q4, k_mm3<MF_SEED4P4>};

// ---- embedding, norm ---------------------------------------------------------------------------------------------------
__global__ void k_embed(int fmt, WSlice w, int d, const int32_t* ids, float* x) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x), t = (int) blockIdx.y;
    if (c >= d) return;
    const size_t row = (size_t) ids[t];
    float v;
    if (fmt == MF_Q8) {
        const uint32_t wq = ((const uint32_t*) w.p[0])[(row * (size_t) d + c) / 4];
        const float q = (float) ((wq >> (8 * (c % 4))) & 255u);
        const size_t gi = (row * (size_t) d + c) / 64;
        v = bfr(bf(((const uint16_t*) w.p[1])[gi]) * q + bf(((const uint16_t*) w.p[2])[gi]));
    } else v = bf(((const uint16_t*) w.p[0])[row * (size_t) d + c]);
    x[(size_t) t * d + c] = v;
}

// grouped RMSNorm (2 groups): y = bf16(w * (x * rsqrt(mean_group(x^2) + eps)))
__global__ void __launch_bounds__(256) k_gnorm(int d, float eps, const float* x, const uint16_t* w, float* y) {
    __shared__ float part[2][8];
    const int t = (int) blockIdx.x, tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    const float* xr = x + (size_t) t * d;
    const int hw = d / 2;
    float s0 = 0, s1 = 0;
    for (int c = tid; c < d; c += 256) {
        const float v = xr[c];
        if (c < hw) s0 += v * v; else s1 += v * v;
    }
    s0 = warp_sum(s0);
    s1 = warp_sum(s1);
    if (lane == 0) { part[0][sg] = s0; part[1][sg] = s1; }
    __syncthreads();
    float a = 0, b = 0;
    for (int i = 0; i < 8; ++i) { a += part[0][i]; b += part[1][i]; }
    const float r0 = rsqrtf(a / (float) hw + eps), r1 = rsqrtf(b / (float) hw + eps);
    for (int c = tid; c < d; c += 256) y[(size_t) t * d + c] = bfr(bf(w[c]) * (xr[c] * (c < hw ? r0 : r1)));
}

// ---- routers: logits with the two-partition BF16 contract, sigmoid, top-k by sigmoid + bias, weights = sigmoid / sum
// * scale.  k_router_logits: one warp per (expert, row).  k_router_topk: one warp per row.
__global__ void __launch_bounds__(256) k_router_logits(RouterArgs a, const uint16_t* W, const uint16_t* bias, const float* x,
                                                       float* score, float* selout) {
    const int lane = threadIdx.x & 31, e = (int) (blockIdx.x * 8 + (threadIdx.x >> 5)), t = (int) blockIdx.y;
    if (e >= a.n) return;
    const float* xr = x + (size_t) t * a.d;
    const uint16_t* wr = W + (size_t) e * a.d;
    float s0 = 0, s1 = 0;
    for (int c = lane * 8; c < a.d; c += 256) {
        const uint4 w = *(const uint4*) (wr + c);
        const float4 x0 = *(const float4*) (xr + c), x1 = *(const float4*) (xr + c + 4);
        const float v = dot4(make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xFFFF0000u), __uint_as_float(w.y << 16),
                                         __uint_as_float(w.y & 0xFFFF0000u)), x0) +
                        dot4(make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xFFFF0000u), __uint_as_float(w.w << 16),
                                         __uint_as_float(w.w & 0xFFFF0000u)), x1);
        if (c < a.hw) s0 += v; else s1 += v;
    }
    s0 = warp_sum(s0);
    s1 = warp_sum(s1);
    if (lane == 0) {
        const float sg = 1.0f / (1.0f + expf(-(bfr(s0) + bfr(s1))));
        score[t * a.n + e] = sg;
        selout[t * a.n + e] = sg + bf(bias[e]);
    }
}
// k rounds of a warp arg max over sel (ties to the lowest expert id), chosen in descending order.
__global__ void k_router_topk(RouterArgs a, const float* score, const float* sel, int32_t* inds, float* wts) {
    const int t = (int) blockIdx.x, lane = threadIdx.x;
    const float* sc = score + (size_t) t * a.n;
    const float* se = sel + (size_t) t * a.n;
    float v[4];   // lane holds experts lane, lane + 32, lane + 64, lane + 96 (n <= 128)
    for (int j = 0; j < 4; ++j) {
        const int e = lane + 32 * j;
        v[j] = e < a.n ? se[e] : -INFINITY;
    }
    int chosen[16];
    float sum = 0;
    for (int k = 0; k < a.top_k; ++k) {
        float bv = -INFINITY;
        int bi = 1 << 20;
        for (int j = 0; j < 4; ++j) {
            const int e = lane + 32 * j;
            if (e < a.n && (v[j] > bv || (v[j] == bv && e < bi))) { bv = v[j]; bi = e; }
        }
        const float m = warp_max(bv);
        int cand = bv == m ? bi : 1 << 20;
        for (int o = 16; o > 0; o >>= 1) cand = min(cand, __shfl_xor_sync(FULL, cand, o));
        chosen[k] = cand;
        sum += sc[cand];
        if (cand % 32 == lane) v[cand / 32] = -INFINITY;
    }
    if (lane == 0)
        for (int k = 0; k < a.top_k; ++k) {
            inds[t * a.top_k + k] = chosen[k];
            wts[t * a.top_k + k] = sc[chosen[k]] / sum * a.scale;
        }
}

// ---- elementwise -------------------------------------------------------------------------------------------------------
__global__ void k_swiglu(const float* g, const float* u, float* a, int n) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) a[i] = bfr(silu_bf(g[i]) * u[i]);
}
// y[t][c] = bf16(h[t][c] + bf16(bf16(sum_k bf16(D[t,k][c] * bf16(w[t,k]))) + shared[t][c]))
__global__ void k_moe_combine(const float* D, const float* w, const float* shared, float* x, int d, int k) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x), t = (int) blockIdx.y;
    if (c >= d) return;
    float s = 0;
    for (int j = 0; j < k; ++j) s += bfr(D[((size_t) t * k + j) * d + c] * bfr(w[t * k + j]));
    const float routed = bfr(s);
    const size_t i = (size_t) t * d + c;
    x[i] = bfr(x[i] + bfr(routed + shared[i]));
}
// v[t][c] = bf16(sum_k bf16(silu(V[t,k][c]) * bf16(w[t,k])))
__global__ void k_vcombine(const float* V, const float* w, float* v, int d, int k) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x), t = (int) blockIdx.y;
    if (c >= d) return;
    float s = 0;
    for (int j = 0; j < k; ++j) s += bfr(silu_bf(V[((size_t) t * k + j) * d + c]) * bfr(w[t * k + j]));
    v[(size_t) t * d + c] = bfr(s);
}
// ---- KV cache access (kernels_cuda.h KvView) ---------------------------------------------------------------------------
static __device__ __forceinline__ KvSeg kv_seg(const KvView& kv, int pos, int* row) {
    if (pos < kv.nv) { *row = pos; return kv.a; }
    *row = pos - kv.nv;
    return kv.b;
}
static __device__ __forceinline__ float4 i8x4(uint32_t u, float s) {
    return make_float4((float) (int8_t) (u & 255u) * s, (float) (int8_t) ((u >> 8) & 255u) * s, (float) (int8_t) ((u >> 16) & 255u) * s,
                       (float) (int8_t) (u >> 24) * s);
}
// 4 values (dims d0 .. d0 + 3) of K or V at (pos, head).  KVF = KV_Q8: the dequantized value rounded to BF16, as
// kv_load8_bf reads it for prefill (decode and prefill see the same K and V).  KVF = -1: the format read at run time and
// Q8 unrounded, the kernel BF16 caches run: its code is 1% faster in decode than a BF16-only instantiation (measured on
// an RTX 4080), and it never reads a Q8 cache (kc_attn launches KV_Q8 for those).
template <int KVF>
static __device__ __forceinline__ float4 kv_load4(const KvView& kv, int is_v, int pos, int kvh, int n_kv, int d0) {
    int row;
    const KvSeg sg = kv_seg(kv, pos, &row);
    const size_t off = ((size_t) row * n_kv + kvh) * ATT_HD + d0;
    if (KVF < 0 && kv.fmt == KV_BF16) {
        const uint2 u = *(const uint2*) ((const uint16_t*) (is_v ? sg.v : sg.k) + off);
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u), __uint_as_float(u.y << 16),
                           __uint_as_float(u.y & 0xFFFF0000u));
    }
    const float4 f = i8x4(*(const uint32_t*) ((const int8_t*) (is_v ? sg.v : sg.k) + off), (is_v ? sg.vs : sg.ks)[(size_t) row * n_kv + kvh]);
    if (KVF < 0) return f;
    return make_float4(bfr(f.x), bfr(f.y), bfr(f.z), bfr(f.w));
}
// 8 values (dims d8 .. d8 + 7) as BF16 bits (Q8: the dequantized value rounded to BF16)
static __device__ __forceinline__ uint4 kv_load8_bf(const KvView& kv, int is_v, int pos, int kvh, int n_kv, int d8) {
    int row;
    const KvSeg sg = kv_seg(kv, pos, &row);
    const size_t off = ((size_t) row * n_kv + kvh) * ATT_HD + d8;
    if (kv.fmt == KV_BF16) return *(const uint4*) ((const uint16_t*) (is_v ? sg.v : sg.k) + off);
    const uint2 u = *(const uint2*) ((const int8_t*) (is_v ? sg.v : sg.k) + off);
    const float s = (is_v ? sg.vs : sg.ks)[(size_t) row * n_kv + kvh];
    const float4 lo = i8x4(u.x, s), hi = i8x4(u.y, s);
    return make_uint4(pack_bf2(lo.x, lo.y), pack_bf2(lo.z, lo.w), pack_bf2(hi.x, hi.y), pack_bf2(hi.z, hi.w));
}

// RoPE (non-traditional halves) on q in place and k into the cache; v into the cache.  Block = one head (64 threads,
// thread p rotates the pair p, p + 64).  Q8: the BF16 K and V (the values the BF16 cache would hold) quantized with the
// row's scale from the block's max |value|.
__global__ void __launch_bounds__(64) k_rope_kv(float* q, const float* k, const float* v, KvView kv, const RowInfo* ri,
                                                const float* inv, int n_head, int n_kv) {
    __shared__ float red[2][2];
    const int head = (int) blockIdx.x, p = threadIdx.x, t = (int) blockIdx.y;
    const int pos = ri[t].pos;
    const float th = (float) pos * inv[p];
    float sn, cs;
    sincosf(th, &sn, &cs);
    if (head < n_head) {
        float* qh = q + (size_t) t * n_head * ATT_HD + head * ATT_HD;
        const float a = qh[p], b = qh[p + 64];
        qh[p] = bfr(a * cs - b * sn);
        qh[p + 64] = bfr(b * cs + a * sn);
    }
    if (head >= n_kv) return;   // uniform over the block
    const float* kh = k + (size_t) t * n_kv * ATT_HD + head * ATT_HD;
    const float* vh = v + (size_t) t * n_kv * ATT_HD + head * ATT_HD;
    const float k0 = kh[p] * cs - kh[p + 64] * sn, k1 = kh[p + 64] * cs + kh[p] * sn, v0 = vh[p], v1 = vh[p + 64];
    int row;
    const KvSeg sg = kv_seg(kv, pos, &row);
    const size_t off = ((size_t) row * n_kv + head) * ATT_HD;
    if (kv.fmt == KV_BF16) {
        uint16_t* kc = (uint16_t*) sg.k + off;
        uint16_t* vc = (uint16_t*) sg.v + off;
        kc[p] = tobf(k0);
        kc[p + 64] = tobf(k1);
        vc[p] = tobf(v0);
        vc[p + 64] = tobf(v1);
        return;
    }
    const float kb0 = bfr(k0), kb1 = bfr(k1), vb0 = bfr(v0), vb1 = bfr(v1);
    float mk = fmaxf(fabsf(kb0), fabsf(kb1)), mv = fmaxf(fabsf(vb0), fabsf(vb1));
    for (int o = 16; o > 0; o >>= 1) { mk = fmaxf(mk, __shfl_xor_sync(FULL, mk, o)); mv = fmaxf(mv, __shfl_xor_sync(FULL, mv, o)); }
    if ((p & 31) == 0) { red[p >> 5][0] = mk; red[p >> 5][1] = mv; }
    __syncthreads();
    mk = fmaxf(red[0][0], red[1][0]);
    mv = fmaxf(red[0][1], red[1][1]);
    const float sk = mk > 0 ? mk / 127.0f : 1.0f, sv = mv > 0 ? mv / 127.0f : 1.0f;
    int8_t* kc = (int8_t*) sg.k + off;
    int8_t* vc = (int8_t*) sg.v + off;
    kc[p] = (int8_t) __float2int_rn(kb0 / sk);
    kc[p + 64] = (int8_t) __float2int_rn(kb1 / sk);
    vc[p] = (int8_t) __float2int_rn(vb0 / sv);
    vc[p + 64] = (int8_t) __float2int_rn(vb1 / sv);
    if (p == 0) { sg.ks[(size_t) row * n_kv + head] = sk; sg.vs[(size_t) row * n_kv + head] = sv; }
}

// ---- attention ---------------------------------------------------------------------------------------------------------
// Split-key attention: block (split, kv head, row); warp = one of the ATT_SG query heads of the KV head; lanes hold 4
// dims each.  Partials [row][head][split] = (m, l, acc[128]).
template <int KVF>
__global__ void __launch_bounds__(32 * ATT_SG) k_attn(AttnArgs a, const float* q, KvView kv, const RowInfo* ri, float* part) {
    const int split = (int) blockIdx.x, kvh = (int) blockIdx.y, t = (int) blockIdx.z;
    const int lane = threadIdx.x & 31, sgi = threadIdx.x >> 5;
    const int qh = kvh * (a.n_head / a.n_kv) + sgi;
    const int pos = ri[t].pos;
    const int nsr = min(a.n_splits, (pos + a.chunk) / a.chunk);   // this row's split count, as a single AR step's
    const int chunk = (pos + nsr) / nsr;                              // ceil((pos + 1) / nsr)
    const int p0 = split * chunk, p1 = min(pos + 1, p0 + chunk);      // splits >= nsr are empty (m = -inf)
    const float* qr = q + (size_t) t * a.n_head * ATT_HD + qh * ATT_HD + lane * 4;
    const float4 qv = make_float4(qr[0] * a.scale, qr[1] * a.scale, qr[2] * a.scale, qr[3] * a.scale);
    float m = -INFINITY, l = 0;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int p = p0; p < p1; p += ATT_KU) {
        float s[ATT_KU];
        float4 vf[ATT_KU];
#pragma unroll
        for (int u = 0; u < ATT_KU; ++u) {
            const int pu = min(p + u, p1 - 1);
            const float4 kf = kv_load4<KVF>(kv, 0, pu, kvh, a.n_kv, lane * 4);
            vf[u] = kv_load4<KVF>(kv, 1, pu, kvh, a.n_kv, lane * 4);
            s[u] = dot4(qv, kf);
        }
        float mx = m;
#pragma unroll
        for (int u = 0; u < ATT_KU; ++u) {
            s[u] = warp_sum(s[u]);
            if (p + u >= p1) s[u] = -INFINITY;
            mx = fmaxf(mx, s[u]);
        }
        const float cor = expf(m - mx);
        acc.x *= cor; acc.y *= cor; acc.z *= cor; acc.w *= cor;
        l *= cor;
#pragma unroll
        for (int u = 0; u < ATT_KU; ++u) {
            const float e = expf(s[u] - mx);
            acc.x += vf[u].x * e; acc.y += vf[u].y * e; acc.z += vf[u].z * e; acc.w += vf[u].w * e;
            l += e;
        }
        m = mx;
    }
    float* pp = part + (((size_t) t * a.n_head + qh) * a.n_splits + split) * (ATT_HD + 2);
    if (lane == 0) { pp[0] = m; pp[1] = l; }
    pp[2 + lane * 4 + 0] = acc.x; pp[2 + lane * 4 + 1] = acc.y; pp[2 + lane * 4 + 2] = acc.z; pp[2 + lane * 4 + 3] = acc.w;
}
static __device__ __forceinline__ float softplus_gate(float g) {   // softplus(x, beta = ln 2) in f32, then BF16
    const float gx = g * 0.69314718055994531f;
    return bfr((fmaxf(gx, 0.0f) + logf(1.0f + expf(-fabsf(gx)))) / 0.69314718055994531f);
}
// Reduce the splits, then the softplus output gate: o[t][h*128+d] = bf16(bf16(attn) * bf16(softplus_ln2(g)))
__global__ void k_attn_reduce(AttnArgs a, const float* part, const float* g, float* o) {
    const int h = (int) blockIdx.x, t = (int) blockIdx.y, d = threadIdx.x;
    const float* pp = part + ((size_t) t * a.n_head + h) * a.n_splits * (ATT_HD + 2);
    float m = -INFINITY;
    for (int s = 0; s < a.n_splits; ++s) m = fmaxf(m, pp[s * (ATT_HD + 2)]);
    float l = 0, acc = 0;
    for (int s = 0; s < a.n_splits; ++s) {
        const float ms = pp[s * (ATT_HD + 2)];
        if (ms == -INFINITY) continue;
        const float c = expf(ms - m);
        l += pp[s * (ATT_HD + 2) + 1] * c;
        acc += pp[s * (ATT_HD + 2) + 2 + d] * c;
    }
    const float att = bfr(acc / l);
    const size_t i = (size_t) t * a.n_head * ATT_HD + h * ATT_HD + d;
    o[i] = bfr(att * softplus_gate(g[i]));
}

// Prefill attention on tensor cores: block (FA_RG x 16 rows, KV head), one warp per (query head of the group, 16 rows);
// keys in tiles of FA_BK through shared memory, shared by the block's heads and rows (FA_RG x ATTF_G warps).  MLX's prefill rounding points as mma operands:
// S = bf16(q * scale) K^T (BF16 products, f32 sums); online softmax per row in f32 (causal: key <= the row's
// position); P = bf16(exp(S - running max)) is the A operand of O += P V; the row sum from the unrounded P; then the
// softplus gate as k_attn_reduce.  Fragments follow mma.m16n8k16: the S accumulator of key tiles 2j, 2j + 1 is the A
// fragment of P for key step j.
#define FA_ROWS 16             // rows per warp (the mma M)
#ifndef FA_RG
#define FA_RG 2                // row groups per block: each K / V tile serves FA_RG x 16 rows of ATTF_G heads
#endif
#define FA_KLD (ATT_HD + 8)    // Ks [key][dim] row stride (BF16)
#define FA_VLD (FA_BK + 8)     // Vt [dim][key] row stride
__global__ void __launch_bounds__(32 * ATTF_G * FA_RG) k_attn_prefill_tc(AttnArgs a, const float* q, KvView kv, const RowInfo* ri,
                                                                         const float* g, float* o, int T) {
    __shared__ __align__(16) uint16_t Ks[FA_BK * FA_KLD];
    __shared__ __align__(16) uint16_t Vt[ATT_HD * FA_VLD];
    const int kvh = (int) blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5, h = kvh * ATTF_G + w % ATTF_G;
    const int gq = lane >> 2, t4 = lane & 3;
    const int b0 = (int) blockIdx.x * FA_ROWS * FA_RG, kend = ri[min(b0 + FA_ROWS * FA_RG, T) - 1].pos + 1;
    const int tw = b0 + (w / ATTF_G) * FA_ROWS, live = tw < T;   // this warp's rows (none past T: it only loads tiles)
    const int t0 = live ? tw : T - 1, nrows = live ? min(FA_ROWS, T - tw) : 1;
    // this thread's two rows (gq, gq + 8) and their positions; rows past T repeat the last
    const int ra = t0 + min(gq, nrows - 1), rb = t0 + min(gq + 8, nrows - 1);
    const int pa = ri[ra].pos, pb = ri[rb].pos;
    const int plast = ri[t0 + nrows - 1].pos;   // the warp's last row: tiles past it are skipped by the whole warp (mma.sync and
                                                // the shuffles need every lane; the mask handles each row)
    // Q fragments: 8 steps of 16 dims, bf16(q * scale)
    uint32_t qf[8][4];
    {
        const float* qa = q + ((size_t) ra * a.n_head + h) * ATT_HD;
        const float* qb = q + ((size_t) rb * a.n_head + h) * ATT_HD;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int c = s * 16 + t4 * 2;
            qf[s][0] = pack_bf2(qa[c] * a.scale, qa[c + 1] * a.scale);
            qf[s][1] = pack_bf2(qb[c] * a.scale, qb[c + 1] * a.scale);
            qf[s][2] = pack_bf2(qa[c + 8] * a.scale, qa[c + 9] * a.scale);
            qf[s][3] = pack_bf2(qb[c + 8] * a.scale, qb[c + 9] * a.scale);
        }
    }
    float O[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0;
    float ma = -INFINITY, mb = -INFINITY, la = 0, lb = 0;
    for (int k0 = 0; k0 < kend; k0 += FA_BK) {
        // K tile [key][dim] and V tile transposed [dim][key], 8 dims per load
        for (int e = threadIdx.x; e < FA_BK * ATT_HD / 8; e += 32 * ATTF_G * FA_RG) {
            const int key = e / (ATT_HD / 8), d8 = (e % (ATT_HD / 8)) * 8, p = k0 + key;
            uint4 kk = make_uint4(0, 0, 0, 0), vv = kk;
            if (p < kend) {
                kk = kv_load8_bf(kv, 0, p, kvh, a.n_kv, d8);
                vv = kv_load8_bf(kv, 1, p, kvh, a.n_kv, d8);
            }
            *(uint4*) &Ks[key * FA_KLD + d8] = kk;
            const uint32_t vw[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                Vt[(d8 + 2 * u) * FA_VLD + key] = (uint16_t) (vw[u] & 0xFFFFu);
                Vt[(d8 + 2 * u + 1) * FA_VLD + key] = (uint16_t) (vw[u] >> 16);
            }
        }
        __syncthreads();
        if (live && k0 <= plast) {
            // S = Q K^T: 8 key tiles of 8
            float S[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                S[j][0] = S[j][1] = S[j][2] = S[j][3] = 0;
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const uint16_t* kr = &Ks[(j * 8 + gq) * FA_KLD + s * 16 + t4 * 2];
                    const uint32_t b[2] = {*(const uint32_t*) kr, *(const uint32_t*) (kr + 8)};
                    mma_bf16(S[j], qf[s], b);
                }
            }
            // mask, row maxima (rows gq: elements 0, 1; gq + 8: 2, 3; a row lives on the 4 lanes of a quad)
            float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key = k0 + j * 8 + t4 * 2;
                if (key > pa) S[j][0] = -INFINITY;
                if (key + 1 > pa) S[j][1] = -INFINITY;
                if (key > pb) S[j][2] = -INFINITY;
                if (key + 1 > pb) S[j][3] = -INFINITY;
                mxa = fmaxf(mxa, fmaxf(S[j][0], S[j][1]));
                mxb = fmaxf(mxb, fmaxf(S[j][2], S[j][3]));
            }
            mxa = fmaxf(mxa, __shfl_xor_sync(FULL, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(FULL, mxa, 2));
            mxb = fmaxf(mxb, __shfl_xor_sync(FULL, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(FULL, mxb, 2));
            const float na = fmaxf(ma, mxa), nb = fmaxf(mb, mxb);
            const float ca = na == -INFINITY ? 1.0f : expf(ma - na), cb = nb == -INFINITY ? 1.0f : expf(mb - nb);
            float sa = 0, sb = 0;
            uint32_t pf[4][4];   // P as A fragments, 4 key steps of 16
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float p0 = S[j][0] == -INFINITY ? 0.0f : expf(S[j][0] - na);
                const float p1 = S[j][1] == -INFINITY ? 0.0f : expf(S[j][1] - na);
                const float p2 = S[j][2] == -INFINITY ? 0.0f : expf(S[j][2] - nb);
                const float p3 = S[j][3] == -INFINITY ? 0.0f : expf(S[j][3] - nb);
                sa += p0 + p1;
                sb += p2 + p3;
                pf[j >> 1][(j & 1) * 2] = pack_bf2(p0, p1);
                pf[j >> 1][(j & 1) * 2 + 1] = pack_bf2(p2, p3);
            }
            sa += __shfl_xor_sync(FULL, sa, 1); sa += __shfl_xor_sync(FULL, sa, 2);
            sb += __shfl_xor_sync(FULL, sb, 1); sb += __shfl_xor_sync(FULL, sb, 2);
            la = la * ca + sa;
            lb = lb * cb + sb;
            ma = na;
            mb = nb;
#pragma unroll
            for (int j = 0; j < 16; ++j) { O[j][0] *= ca; O[j][1] *= ca; O[j][2] *= cb; O[j][3] *= cb; }
            // O += P V: 16 dim tiles of 8, 4 key steps of 16
#pragma unroll
            for (int j = 0; j < 16; ++j)
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const uint16_t* vr = &Vt[(j * 8 + gq) * FA_VLD + s * 16 + t4 * 2];
                    const uint32_t b[2] = {*(const uint32_t*) vr, *(const uint32_t*) (vr + 8)};
                    mma_bf16(O[j], pf[s], b);
                }
        }
        __syncthreads();
    }
    // out: rows gq (elements 0, 1) and gq + 8 (2, 3), dims j * 8 + t4 * 2 + {0, 1}
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int q2 = 0; q2 < 4; ++q2) {
            const int r = (q2 >> 1) ? gq + 8 : gq;
            if (!live || r >= nrows) continue;
            const size_t i = ((size_t) (t0 + r) * a.n_head + h) * ATT_HD + j * 8 + t4 * 2 + (q2 & 1);
            o[i] = bfr(bfr(O[j][q2] / ((q2 >> 1) ? lb : la)) * softplus_gate(g[i]));
        }
}

// ---- expert cache ------------------------------------------------------------------------------------------------------
#define ADMIT_THREADS 1024
__global__ void __launch_bounds__(ADMIT_THREADS) k_cache_admit(CachePool p, int sl, const int32_t* inds, int count, int flags) {
    const int prev = flags & CACHE_PROTECT_PREV ? (int) p.tick[0] : -1;
    __shared__ int need[128];
    __shared__ int miss[128];
    __shared__ int nmiss;
    __shared__ uint32_t bv[32];
    __shared__ int bi[32];
    __shared__ int wmiss[4], wneed[4];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    if (tid < 128) need[tid] = 0;
    __syncthreads();
    for (int i = tid; i < count; i += ADMIT_THREADS) need[inds[i]] = 1;
    __syncthreads();
    // Ticks count up from CACHE_TICK_BASE; never-used slots (0) are the oldest.
    const uint32_t now = p.tick[0] + 1;
    // one thread per expert of the layer: hits are touched; misses are listed in expert order (a ballot per warp)
    const int isneed = tid < p.per_layer && need[tid];
    const int s0 = isneed ? p.unit_slot[sl * p.per_layer + tid] : -1, ismiss = isneed && s0 < 0;
    if (isneed && s0 >= 0) p.slot_last[s0] = now;
    const uint32_t mb = __ballot_sync(FULL, ismiss), nb = __ballot_sync(FULL, isneed);
    if (lane == 0 && w < 4) { wmiss[w] = __popc(mb); wneed[w] = __popc(nb); }
    __syncthreads();
    if (ismiss) {
        int before = __popc(mb & ((1u << lane) - 1u));
        for (int k = 0; k < w; ++k) before += wmiss[k];
        miss[before] = tid;
    }
    if (tid == 0) {
        const int nm = wmiss[0] + wmiss[1] + wmiss[2] + wmiss[3], nn = wneed[0] + wneed[1] + wneed[2] + wneed[3];
        nmiss = nm;
        p.tick[0] = now;
        p.stats[1] += (uint32_t) nm;
        p.stats[0] += (uint32_t) (nn - nm);
        *p.njobs = nm;
    }
    __syncthreads();
    for (int m = 0; m < nmiss; ++m) {
        // victim: the least recently used slot (ties: lowest index) not touched by this call
        uint32_t best = 0xFFFFFFFFu;
        int bs = 0x7FFFFFFF;
        for (int s = tid; s < p.slots; s += ADMIT_THREADS) {
            const uint32_t l = p.slot_last[s];
            if (l != now && (int) l != prev && l < best) { best = l; bs = s; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const uint32_t b2 = __shfl_xor_sync(FULL, best, o);
            const int s2 = __shfl_xor_sync(FULL, bs, o);
            if (b2 < best || (b2 == best && s2 < bs)) { best = b2; bs = s2; }
        }
        if (lane == 0) { bv[w] = best; bi[w] = bs; }
        __syncthreads();
        if (tid == 0) {
            for (int i = 1; i < ADMIT_THREADS / 32; ++i) if (bv[i] < best || (bv[i] == best && bi[i] < bs)) { best = bv[i]; bs = bi[i]; }
        }
        if (tid == 0 && bs >= p.slots) {   // no slot may be replaced (a pool smaller than two layers): stop admitting
            *p.njobs = m;
            nmiss = m;
        }
        __syncthreads();
        if (m >= nmiss) break;
        if (tid == 0) {
            const int x = miss[m], u = sl * p.per_layer + x, old = p.slot_unit[bs];
            if (old >= 0) p.unit_slot[old] = -1;
            p.slot_unit[bs] = u;
            p.unit_slot[u] = bs;
            p.slot_last[bs] = now;
            p.jobs[2 * m] = u;
            p.jobs[2 * m + 1] = bs;
            uint8_t* base = p.vram + (uint64_t) bs * p.unit_bytes;
            for (int i = 0; i < p.ntens; ++i) {
                WSlice* t = p.tab + ((size_t) sl * p.ntens + i) * p.per_layer + x;
                for (int st = 0; st < 4; ++st) if (p.off[i][st] != 0xFFFFFFFFu) t->p[st] = base + p.off[i][st];
            }
        }
        __syncthreads();
    }
}
// The queued units, host arena -> VRAM slots, 16 bytes per thread per step.
__global__ void __launch_bounds__(512) k_cache_copy(CachePool p) {
    const int nj = *p.njobs;
    const uint64_t words = p.unit_bytes / 16, stride = (uint64_t) gridDim.x * blockDim.x;
    for (int j = 0; j < nj; ++j) {
        const uint4* src = (const uint4*) (p.host + (uint64_t) p.jobs[2 * j] * p.unit_bytes);
        uint4* dst = (uint4*) (p.vram + (uint64_t) p.jobs[2 * j + 1] * p.unit_bytes);
        for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) dst[i] = src[i];
    }
}

// Greedy argmax over the vocabulary (ties: the lowest id); one block of 1024 per row.
__global__ void __launch_bounds__(1024) k_argmax(const float* logits, int32_t* out, int V) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int t = (int) blockIdx.x, tid = threadIdx.x, lane = tid & 31, sgi = tid >> 5;
    const float* l = logits + (size_t) t * V;
    float best = -INFINITY;
    int idx = 0;
    for (int i = tid; i < V; i += 1024) if (l[i] > best) { best = l[i]; idx = i; }
    for (int o = 16; o > 0; o /= 2) {
        const float b2 = __shfl_down_sync(FULL, best, o);
        const int i2 = __shfl_down_sync(FULL, idx, o);
        if (b2 > best || (b2 == best && i2 < idx)) { best = b2; idx = i2; }
    }
    if (lane == 0) { bv[sgi] = best; bi[sgi] = idx; }
    __syncthreads();
    if (tid == 0) {
        float b = bv[0];
        int ix = bi[0];
        for (int s = 1; s < 32; ++s) if (bv[s] > b || (bv[s] == b && bi[s] < ix)) { b = bv[s]; ix = bi[s]; }
        out[t] = ix;
    }
}

// ---- launchers ---------------------------------------------------------------------------------------------------------
extern "C" {

void kc_embed(cudaStream_t s, int fmt, WSlice w, int d, const int32_t* ids, float* x, int T) {
    k_embed<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(fmt, w, d, ids, x);
}
void kc_gnorm(cudaStream_t s, int d, float eps, const float* x, const uint16_t* w, float* y, int T) {
    k_gnorm<<<(unsigned) T, 256, 0, s>>>(d, eps, x, w, y);
}
void kc_mv(cudaStream_t s, int fmt, WSlice w, int K, int R, const float* x, int xs, float* y, int ys, int T, int add,
           const uint32_t* G) {
    MvArgs a = {K, R, T, 1, xs, ys, 0, add};
    MV_K[fmt]<<<dim3((unsigned) (R + MV_ROWS - 1) / MV_ROWS, 1), 32 * MV_ROWS, 0, s>>>(a, w, NULL, x, y, NULL, G, T);
}
void kc_mv_sel(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
               const int32_t* sel, int P, int xdiv, const uint32_t* G) {
    MvArgs a = {K, R, P, xdiv, xs, ys, 0, 0};
    WSlice none;
    memset(&none, 0, sizeof none);
    MV_K[fmt]<<<dim3((unsigned) (R + MV_ROWS - 1) / MV_ROWS, (unsigned) P), 32 * MV_ROWS, 0, s>>>(a, none, ws, x, y, sel, G, 1);
}
void kc_mv_gu(cudaStream_t s, int fmt, const WSlice* wg, const WSlice* wu, int K, int R, const float* x, int xs, float* a_,
              int ys, const int32_t* sel, int P, int xdiv, const uint32_t* G) {
    MvArgs a = {K, R, P, xdiv, xs, ys, 0, 0};
    MVGU_K[fmt]<<<dim3((unsigned) (R + MV_ROWS - 1) / MV_ROWS, (unsigned) P), 32 * MV_ROWS, 0, s>>>(a, wg, wu, x, a_, sel, G);
}
static int g_seed_exact;
void kc_seed_gemm_f32(int on) { g_seed_exact = on; }
static MmKernel mm_kernel(int fmt) { return (g_seed_exact ? MM3_K : MM_K)[fmt]; }
void kc_mm(cudaStream_t s, int fmt, WSlice w, int K, int R, const float* x, int xs, float* y, int ys, int T, int add,
           const uint32_t* G) {
    MmArgs a = {K, R, T, xs, ys, 1, add, 0};
    mm_kernel(fmt)<<<dim3((unsigned) (R + MMT_BM - 1) / MMT_BM, (unsigned) (T + MMT_BN - 1) / MMT_BN), 128, 0, s>>>(a, w, NULL, x, y, NULL, NULL, NULL, G);
}
void kc_mm_grouped(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
                   int xdiv, const int32_t* perm, const MmTile* tiles, int ntiles, const uint32_t* G) {
    if (!ntiles) return;
    MmArgs a = {K, R, 0, xs, ys, xdiv, 0, 0};
    WSlice none;
    memset(&none, 0, sizeof none);
    mm_kernel(fmt)<<<dim3((unsigned) (R + MMT_BM - 1) / MMT_BM, (unsigned) ntiles), 128, 0, s>>>(a, none, ws, x, y, perm, tiles, NULL, G);
}
void kc_mm_grouped_dev(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
                       int xdiv, const int32_t* perm, const MmTile* tiles, const int32_t* ntiles, int max_tiles,
                       const uint32_t* G) {
    MmArgs a = {K, R, 0, xs, ys, xdiv, 0, 0};
    WSlice none;
    memset(&none, 0, sizeof none);
    mm_kernel(fmt)<<<dim3((unsigned) (R + MMT_BM - 1) / MMT_BM, (unsigned) max_tiles), 128, 0, s>>>(a, none, ws, x, y, perm, tiles, ntiles, G);
}
void kc_bucket(cudaStream_t s, const int32_t* inds, int count, int n, int32_t* perm, MmTile* tiles, int32_t* ntiles) {
    k_bucket_count<<<1, 1024, 0, s>>>(inds, count, n, ntiles);
    k_bucket_place<<<(unsigned) n, 1024, 0, s>>>(inds, count, ntiles, perm, tiles);
}
void kc_router(cudaStream_t s, RouterArgs a, const uint16_t* W, const uint16_t* bias, const float* x, float* score,
               float* sel, int32_t* inds, float* wts, int T) {
    k_router_logits<<<dim3((unsigned) (a.n + 7) / 8, (unsigned) T), 256, 0, s>>>(a, W, bias, x, score, sel);
    k_router_topk<<<(unsigned) T, 32, 0, s>>>(a, score, sel, inds, wts);
}
void kc_router_topk(cudaStream_t s, RouterArgs a, const float* score, const float* sel, int32_t* inds, float* wts, int T) {
    k_router_topk<<<(unsigned) T, 32, 0, s>>>(a, score, sel, inds, wts);
}
void kc_swiglu(cudaStream_t s, const float* g, const float* u, float* a, int n) {
    k_swiglu<<<(unsigned) (n + 255) / 256, 256, 0, s>>>(g, u, a, n);
}
void kc_moe_combine(cudaStream_t s, const float* D, const float* w, const float* shared, float* x, int d, int k, int T) {
    k_moe_combine<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(D, w, shared, x, d, k);
}
void kc_vcombine(cudaStream_t s, const float* V, const float* w, float* v, int d, int k, int T) {
    k_vcombine<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(V, w, v, d, k);
}
void kc_rope_kv(cudaStream_t s, float* q, const float* k, const float* v, KvView kv, const RowInfo* ri, const float* inv,
                int n_head, int n_kv, int T) {
    k_rope_kv<<<dim3((unsigned) n_head, (unsigned) T), 64, 0, s>>>(q, k, v, kv, ri, inv, n_head, n_kv);
}
void kc_attn(cudaStream_t s, AttnArgs a, const float* q, KvView kv, const RowInfo* ri, float* part, const float* g, float* o,
             int T) {
    (kv.fmt == KV_Q8 ? k_attn<KV_Q8> : k_attn<-1>)<<<dim3((unsigned) a.n_splits, (unsigned) a.n_kv, (unsigned) T),
                                                          32 * ATT_SG, 0, s>>>(a, q, kv, ri, part);
    k_attn_reduce<<<dim3((unsigned) a.n_head, (unsigned) T), ATT_HD, 0, s>>>(a, part, g, o);
}
void kc_attn_prefill(cudaStream_t s, AttnArgs a, const float* q, KvView kv, const RowInfo* ri, const float* g, float* o, int T) {
    k_attn_prefill_tc<<<dim3((unsigned) (T + FA_ROWS * FA_RG - 1) / (FA_ROWS * FA_RG), (unsigned) a.n_kv), 32 * ATTF_G * FA_RG, 0, s>>>(a, q, kv, ri, g, o, T);
}
void kc_cache_admit(cudaStream_t s, const CachePool* p, int sl, const int32_t* inds, int count, int flags) {
    k_cache_admit<<<1, ADMIT_THREADS, 0, s>>>(*p, sl, inds, count, flags);
}
void kc_cache_copy(cudaStream_t s, const CachePool* p, int blocks) { k_cache_copy<<<(unsigned) blocks, 512, 0, s>>>(*p); }
void kc_argmax(cudaStream_t s, const float* logits, int32_t* out, int V, int n) {
    k_argmax<<<(unsigned) n, 1024, 0, s>>>(logits, out, V);
}

}   // extern "C"
