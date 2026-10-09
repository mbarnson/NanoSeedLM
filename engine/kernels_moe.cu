// engine/kernels_moe.cu - CUDA kernels of the MoVA engine (engine/mova_cuda.c), ports of engine/kernels_moe.metal with
// the same arithmetic and the same BF16 rounding points (activations are f32 buffers of BF16-rounded values, rounded
// wherever the MLX reference implementation produces a BF16 tensor).  A Metal simdgroup is a warp; a threadgroup is a
// block.  Written in C style: one __device__ body per kernel, specialised per weight format by the FMT_KERNELS macro.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kernels_cuda.h"
#include "kvq.h"

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

// Q8, two adjacent blocks (one 16-byte load, one group): the products of one token, in the order mv_lane and mv_body
// (several tokens) both add them, so a token's dense matvec does not depend on the forward's row count.
static __device__ __forceinline__ float q8x16(const float* wq, const float* xb) {
    const float4* xr = (const float4*) xb;
    float acc = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 xv = xr[i];
        acc += wq[4 * i] * xv.x + wq[4 * i + 1] * xv.y + wq[4 * i + 2] * xv.z + wq[4 * i + 3] * xv.w;
    }
    return acc;
}
static __device__ __forceinline__ void q8w16(uint4 q, float s, float b, float* wq) {
    const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) wq[4 * i + k] = bfr(s * (float) ((qw[i] >> (8 * k)) & 255u) + b);
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
            float wq[16];
            q8w16(qr[jq], bf(S[gi]), bf(B[gi]), wq);
            acc += q8x16(wq, xb + (size_t) jq * 16);
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
static __device__ __forceinline__ float softplus_gate(float g) {   // softplus(x, beta = ln 2) in f32, then BF16
    const float gx = g * 0.69314718055994531f;
    return bfr((fmaxf(gx, 0.0f) + logf(1.0f + expf(-fabsf(gx)))) / 0.69314718055994531f);
}

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
    // Every token's products add exactly as mv_lane adds them (nt == 1): a row's result does not depend on how many
    // rows share the forward (batched decode equals decoding alone).
    if (fmt == MF_SEED4P4 && nbk % 64 == 0) {   // mv_lane's two adjacent blocks per lane
        const uint32_t* s2 = (const uint32_t*) ((const uint16_t*) w.p[0] + (size_t) r * nbk);
        const uint32_t* c2 = (const uint32_t*) ((const uint16_t*) w.p[1] + (size_t) r * nbk);
        const uint8_t* EN = (const uint8_t*) w.p[3];
        const uint32_t kb = (uint32_t) r * (uint32_t) nbk;
        for (int jq = lane; jq < nbk / 2; jq += 32) {
            const uint32_t ss = s2[jq], cc = c2[jq], eb2 = EN[(kb + 2 * jq) >> 1];
            const uint32_t sa = ss & 0xFFFFu, sb = ss >> 16;
            const uint32_t ga = __ldg(G + sa), gb = __ldg(G + sb);
            const int ea = w.eb + (int) (eb2 & 15u), eb = w.eb + (int) ((eb2 >> 4) & 15u);
#pragma unroll
            for (int t = 0; t < MV_MAXT; ++t) {
                if (t >= nt) break;
                const float4* xr = (const float4*) (xb + (size_t) t * a.xs + (size_t) (2 * jq) * 8);
                const float4 x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
                acc[t] += seed4_dot(sa, ga, cc & 0xFFFFu, ea, x0, x1, -1.5f * sum8(x0, x1));
                acc[t] += seed4_dot(sb, gb, cc >> 16, eb, x2, x3, -1.5f * sum8(x2, x3));
            }
        }
#pragma unroll
        for (int t = 0; t < MV_MAXT; ++t) acc[t] *= (1.0f / 32767.0f);
    } else if (fmt == MF_Q8 && nbk % 64 == 0) {   // mv_lane's 16-byte path
        const uint16_t* S = (const uint16_t*) w.p[1];
        const uint16_t* B = (const uint16_t*) w.p[2];
        const uint4* qr = (const uint4*) ((const uint32_t*) w.p[0] + ((size_t) r * a.K) / 4);
        for (int jq = lane; jq < nbk / 2; jq += 32) {
            const size_t gi = ((size_t) r * a.K + (size_t) jq * 16) / 64;
            float wq[16];
            q8w16(qr[jq], bf(S[gi]), bf(B[gi]), wq);
#pragma unroll
            for (int t = 0; t < MV_MAXT; ++t) {
                if (t >= nt) break;
                acc[t] += q8x16(wq, xb + (size_t) t * a.xs + (size_t) jq * 16);
            }
        }
    } else if (fmt == MF_SEED4P4 || fmt == MF_SEED4) {   // one state expansion per block for every token
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

// Tensor-core GEMM (prefill): Y[n][r] = sum_k W[r][k] X[n][k] for a 64-row x MMT_BN-column tile (128: each
// reconstructed seed weight tile serves twice the tokens of 64, 3% faster at 4k).  BF16 mma.sync with f32
// accumulation: the activations are BF16 values (exact).  BF16 / Q8 / Q4 weights dequantize to BF16 values (exact).
// Seed weights (f32 in the matvec and in the Metal GEMM) are rounded to BF16 here (NT = 1), or, in the k_mm3 kernels
// (NSLM_SEED_GEMM_F32), carried as NT = 3 BF16 terms hi + mid + lo whose sum is the f32 value exactly.  Measured on the
// real model against BF16 reference log-probs, the exact weights change neither KLD (held-out 0.0261 vs 0.0260, standard
// error 0.0005) nor NLL, and cost 15% (64-wide tiles) to 30% (128) of 4k prefill: the tensor cores do three times the
// work.  BF16 is the default.  MMT_THREADS / 32 warps, each a 32 x 32 sub-tile.
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
                                               const uint32_t* G, const float* gate = NULL) {
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
    const int wm = (w / (MMT_BN / 32)) * 32, wn = (w % (MMT_BN / 32)) * 32;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;
    for (int k0 = 0; k0 < a.K; k0 += MMT_BK) {
        // weights: 64 rows x 4 blocks of 8
        for (int b = tid; b < MMT_BM * 4; b += MMT_THREADS) {
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
        // inputs: MMT_BN columns x 32 k, 16 per thread
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
                const size_t yi = (size_t) dst * a.ys + row;
                if (a.add == 2) Y[yi] = bfr(bfr(acc[mt][nt][q]) * softplus_gate(gate[yi]));   // MLA's v_up with the gate
                else store_y(Y + yi, acc[mt][nt][q], a.add);
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
    __global__ void __launch_bounds__(MMT_THREADS) k_mm_##SUF(MmArgs a, WSlice wd, const WSlice* ws, const float* X, float* Y,  \
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
__global__ void __launch_bounds__(MMT_THREADS) k_mm3(MmArgs a, WSlice wd, const WSlice* ws, const float* X, float* Y,
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
    const KvSeg sg = kv_seg(kv, ri[t].kv0 + pos, &row);   // the slot's cache row
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
            const int pu = ri[t].kv0 + min(p + u, p1 - 1);   // the cache row
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

// ---- MLA caches in FP8 / FP4 (nslm/kvq.h is the spec: the same codes bit for bit, as kernels_moe.metal) -----------------
static __device__ __forceinline__ float e4m3_dec(uint32_t c) {
    const uint32_t e = (c >> 3) & 15u, m = c & 7u;
    const float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float) m * 0x1p-9f;
    return c & 0x80u ? -v : v;
}
static __device__ __forceinline__ float e8m0_dec(uint32_t s) { return __uint_as_float(s << 23); }   // 2^(s - 127), s in 1 .. 254
static __device__ __forceinline__ float e2m1_dec(uint32_t n) {
    const uint32_t k = n & 7u;   // 0, 0.5, 1, 1.5, 2, 3, 4, 6
    const float v = k < 4u ? (float) k * 0.5f : (float) (1u << ((k >> 1) - 1u)) * ((k & 1u) ? 1.5f : 1.0f);
    return n & 8u ? -v : v;
}
static __device__ __forceinline__ uint32_t e4m3_enc_abs(float a) {   // a >= 0: nearest even, saturated at 448
    if (!(a < 448.0f)) return 0x7Eu;
    if (a < 0x1p-6f) return (uint32_t) rintf(a * 512.0f);
    int e;
    const float f = frexpf(a, &e);
    int E = e - 1, m = (int) rintf((2 * f - 1) * 8);
    if (m == 8) { ++E; m = 0; }
    return min((uint32_t) (((E + 7) << 3) | m), 0x7Eu);
}
static __device__ __forceinline__ uint32_t e2m1_enc(float x, float s) {   // x / s against the midpoints times s (exact); ties even
    const float a = fabsf(x);
    uint32_t k = (uint32_t) (a > 0.25f * s) + (uint32_t) (a > 0.75f * s) + (uint32_t) (a > 1.25f * s) + (uint32_t) (a > 1.75f * s) +
                 (uint32_t) (a > 2.5f * s) + (uint32_t) (a > 3.5f * s) + (uint32_t) (a > 5.0f * s);
    k += (uint32_t) (a == 0.75f * s || a == 1.75f * s || a == 3.5f * s);   // a tie above an odd code: the even one
    return k && x < 0 ? k | 8u : k;
}
static __device__ __forceinline__ uint32_t fp8_scale(float amax) {
    if (!(amax > 0)) return 127u;
    int k;
    frexpf(amax, &k);
    const int e = min(max(amax <= ldexpf(448.0f, k - 9) ? k - 9 : k - 8, -126), 127);
    return (uint32_t) (e + 127);
}
static __device__ __forceinline__ uint32_t fp4_scale(float amax) {
    if (!(amax > 0)) return 0u;
    const int c0 = (int) e4m3_enc_abs(amax / 6);
    int best = -1;
    float bd = 0;
    for (int c = max(c0 - 1, 0); c <= min(c0 + 1, 0x7E); ++c) {
        const float d = fabsf(6 * e4m3_dec((uint32_t) c) - amax);
        if (best < 0 || d < bd || (d == bd && !(c & 1))) { best = c; bd = d; }
    }
    return (uint32_t) best;
}
// A row of n values: the first `lead` (a multiple of 32) FP8 with an E8M0 scale per 32, the rest FP4 (two a byte, the
// even value low) with an E4M3 scale per 16: lead + (n - lead) / 2 code bytes, lead / 32 + (n - lead) / 16 scales.
static __device__ __forceinline__ int kvq_rowb(int lead, int n) { return lead + (n - lead) / 2; }
static __device__ __forceinline__ int kvq_srowb(int lead, int n) { return lead / 32 + (n - lead) / 16; }
static __device__ __forceinline__ int mla_lead(int fmt, int r) { return fmt == KV_FP8 || r < KVQ_FP4_LEAD ? r : KVQ_FP4_LEAD; }
// Values d .. d + 7 (d a multiple of 8) of row `row` as BF16 pairs (exact: a code times its scale)
static __device__ __forceinline__ uint4 kvq_get8(const uint8_t* C, const uint8_t* S, int lead, int n, size_t row, int d) {
    C += row * (size_t) kvq_rowb(lead, n);
    S += row * (size_t) kvq_srowb(lead, n);
    float v[8];
    if (d < lead) {
        const uint2 w = *(const uint2*) (C + d);
        const float s = e8m0_dec(S[d / 32]);
#pragma unroll
        for (int i = 0; i < 4; ++i) { v[i] = e4m3_dec((w.x >> (8 * i)) & 255u) * s; v[4 + i] = e4m3_dec((w.y >> (8 * i)) & 255u) * s; }
    } else {
        const int e = d - lead;
        const uint32_t w = *(const uint32_t*) (C + lead + e / 2);
        const float s = e4m3_dec(S[lead / 32 + e / 16]);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = e2m1_dec(w >> (4 * i)) * s;
    }
    return make_uint4(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3]), pack_bf2(v[4], v[5]), pack_bf2(v[6], v[7]));
}
// Dims d .. d + 7 of an MLA cache row as BF16 bits: the RoPE key (isk; 128 dims) or the latent (r dims), any format
static __device__ __forceinline__ uint4 mla_get8(int fmt, const KvSeg& sg, size_t row, int isk, int r, int d) {
    if (fmt == KV_BF16) return *(const uint4*) ((const uint16_t*) (isk ? sg.k : sg.v) + row * (isk ? ATT_HD : r) + d);
    if (isk) return kvq_get8((const uint8_t*) sg.k, (const uint8_t*) sg.ks, ATT_HD, ATT_HD, row, d);
    return kvq_get8((const uint8_t*) sg.v, (const uint8_t*) sg.vs, mla_lead(fmt, r), r, row, d);
}
// Store value v at dim d of a quantized row (lead FP8 dims, then FP4) when `store`: d % 32 is the lane, and a block's
// dims are consecutive lanes, which share its max |v| (FP4 pairs lanes into bytes).  Every lane of the warp calls it.
static __device__ __forceinline__ void kvq_put(bool store, int lead, float v, uint8_t* C, uint8_t* S, size_t row, int n, int d) {
    float a16 = fabsf(v);
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) a16 = fmaxf(a16, __shfl_xor_sync(FULL, a16, o));
    const float a32 = fmaxf(a16, __shfl_xor_sync(FULL, a16, 16));
    const bool f8 = d < lead;
    const uint32_t sc = f8 ? fp8_scale(a32) : fp4_scale(a16);
    const float s4 = e4m3_dec(sc);
    const uint32_t c = f8 ? e4m3_enc_abs(ldexpf(fabsf(v), 127 - (int) sc)) : (s4 > 0 ? e2m1_enc(v, s4) : 0u);
    const uint32_t hi = __shfl_xor_sync(FULL, c, 1);
    if (!store) return;
    C += row * (size_t) kvq_rowb(lead, n);
    S += row * (size_t) kvq_srowb(lead, n);
    const int e = d - lead;
    if (f8) {
        C[d] = (uint8_t) (c && v < 0 ? c | 0x80u : c);
        if (d % 32 == 0) S[d / 32] = (uint8_t) sc;
    } else {
        if (!(e & 1)) C[lead + e / 2] = (uint8_t) (c | hi << 4);
        if (e % 16 == 0) S[lead / 32 + e / 16] = (uint8_t) sc;
    }
}

// y = `rows` FP8 / FP4 cache rows of n values (a latent's layout, mla_lead), decoded (8 values a thread; tests)
__global__ void k_kv_f32(int fmt, const uint8_t* C, const uint8_t* S, int n, int rows, float* y) {
    const int i = ((int) blockIdx.x * (int) blockDim.x + threadIdx.x) * 8;
    if (i >= n * rows) return;
    const uint4 w = kvq_get8(C, S, mla_lead(fmt, n), n, (size_t) (i / n), i % n);
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { y[i + 2 * j] = __uint_as_float(u[j] << 16); y[i + 2 * j + 1] = __uint_as_float(u[j] & 0xFFFF0000u); }
}

// ---- MLA (TransMLA): per-head maps, RoPE + latent cache write, latent attention ---------------------------------------
// The KvView of an MLA layer: k holds the RoPE key (ATT_HD BF16 per position), v the latent (r BF16 per position).

// Values e .. e + 3 / e .. e + 7 (e a multiple of 4 / 8) of a BF16, Q8 or Q4 tensor (w.p: values, or codes, BF16 scales
// and biases per 64; nslm/model_st.h), as wblock dequantizes them: bf16(scale x code + bias), exact BF16 values.
static __device__ __forceinline__ float4 welem4(int fmt, const WSlice& w, size_t e) {
    if (fmt == MF_BF16) {
        const uint2 u = *(const uint2*) ((const uint16_t*) w.p[0] + e);
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u), __uint_as_float(u.y << 16),
                           __uint_as_float(u.y & 0xFFFF0000u));
    }
    const float sc = bf(((const uint16_t*) w.p[1])[e / 64]), bi = bf(((const uint16_t*) w.p[2])[e / 64]);
    const uint32_t q = fmt == MF_Q8 ? *(const uint32_t*) ((const uint8_t*) w.p[0] + e) : *(const uint16_t*) ((const uint8_t*) w.p[0] + e / 2);
    const int b = fmt == MF_Q8 ? 8 : 4;
    const uint32_t m = (1u << b) - 1u;
    return make_float4(bfr(sc * (float) (q & m) + bi), bfr(sc * (float) ((q >> b) & m) + bi), bfr(sc * (float) ((q >> 2 * b) & m) + bi),
                       bfr(sc * (float) ((q >> 3 * b) & m) + bi));
}
static __device__ __forceinline__ uint4 welem8(int fmt, const WSlice& w, size_t e) {
    if (fmt == MF_BF16) return *(const uint4*) ((const uint16_t*) w.p[0] + e);
    const float4 a = welem4(fmt, w, e), b = welem4(fmt, w, e + 4);
    return make_uint4(pack_bf2(a.x, a.y), pack_bf2(a.z, a.w), pack_bf2(b.x, b.y), pack_bf2(b.z, b.w));
}

// Per-head maps (q_rope_mix, q_lat, v_up): warp = (output row o, head h, token t); lanes take 4 inputs at a time
// (I is a multiple of 32; Q8 / Q4: of 64); f32 sums.  y[t][h][o] = bf16(W_h[o] . x), or with the gate bf16(bf16(W_h[o] .
// x) * bf16(softplus_ln2(g[t][h][o]))).  W BF16, Q8 or Q4 (welem4).  Block = 8 output rows; grid (ceil(O / 8), H, T).
#define HMV_ROWS 8
__global__ void __launch_bounds__(32 * HMV_ROWS) k_heads_mv(HmvArgs a, int fmt, WSlice W, const float* x, const float* g, float* y) {
    const int lane = threadIdx.x & 31, o = (int) blockIdx.x * HMV_ROWS + (threadIdx.x >> 5), h = (int) blockIdx.y, t = (int) blockIdx.z;
    if (o >= a.O) return;   // whole warps
    const size_t w0 = ((size_t) h * a.O + o) * a.I;
    const float* xv = x + (size_t) t * a.xs + (size_t) h * a.hs;
    float s = 0;
    for (int i = lane * 4; i < a.I; i += 128) {
        const float4 wf = welem4(fmt, W, w0 + i);
        const float4 xf = *(const float4*) (xv + i);
        s += wf.x * xf.x + wf.y * xf.y + wf.z * xf.z + wf.w * xf.w;
    }
    s = warp_sum(s);
    if (lane == 0) {
        const size_t yi = ((size_t) t * a.H + h) * a.O + o;
        float v = bfr(s);
        if (a.gate) v = bfr(v * softplus_gate(g[yi]));
        y[yi] = v;
    }
}

// RoPE (as k_rope_kv) on the query RoPE parts qr [T][n_head][128] in place and on the RoPE key kr [T][128] into the
// cache's k; the latent c [T][r] into the cache's v.  Thread i: latent dim i (i < r) and the pair (i % 64, + 64) of head
// i / 64 (head n_head: the key).  Blocks of 64; grid (ceil(max((n_head + 1) * 64, r) / 64), T).
__global__ void __launch_bounds__(64) k_mla_rope(MlaArgs a, float* qr, const float* kr, const float* c, KvView kv, const RowInfo* ri,
                                                 const float* inv) {
    const int i = (int) blockIdx.x * 64 + threadIdx.x, t = (int) blockIdx.y, pos = ri[t].pos;
    int row;
    const KvSeg sg = kv_seg(kv, ri[t].kv0 + pos, &row);   // the slot's cache row
    const int head = i / 64, p = i % 64;
    const bool lat = i < a.r, key = head == a.n_head;
    float y0 = 0, y1 = 0;
    if (head <= a.n_head) {
        float sn, cs;
        sincosf((float) pos * inv[p], &sn, &cs);
        float* qh = qr + ((size_t) t * a.n_head + (key ? 0 : head)) * ATT_HD;
        const float* xh = key ? kr + (size_t) t * ATT_HD : qh;
        const float x0 = xh[p], x1 = xh[p + 64];
        y0 = x0 * cs - x1 * sn;
        y1 = x1 * cs + x0 * sn;
        if (!key) { qh[p] = bfr(y0); qh[p + 64] = bfr(y1); }
    }
    if (kv.fmt == KV_BF16) {
        if (lat) ((uint16_t*) sg.v)[(size_t) row * a.r + i] = tobf(c[(size_t) t * a.r + i]);
        if (key) {
            uint16_t* kc = (uint16_t*) sg.k + (size_t) row * ATT_HD;
            kc[p] = tobf(y0);
            kc[p + 64] = tobf(y1);
        }
        return;
    }
    // FP8 / FP4 (uniform over the grid): every lane runs the shuffles; a warp's 32 dims are one FP8 block or two FP4 ones
    kvq_put(lat, mla_lead(kv.fmt, a.r), lat ? bfr(c[(size_t) t * a.r + i]) : 0.0f, (uint8_t*) sg.v, (uint8_t*) sg.vs, row, a.r, i);
    kvq_put(key, ATT_HD, bfr(y0), (uint8_t*) sg.k, (uint8_t*) sg.ks, row, ATT_HD, p);
    kvq_put(key, ATT_HD, bfr(y1), (uint8_t*) sg.k, (uint8_t*) sg.ks, row, ATT_HD, p + 64);
}

// Latent attention (absorbed MLA: multi-query over the shared latent), as Metal's k_mla_attn.  Block (split, head group,
// row) of MLA_HG warps; warp = query head blockIdx.y * MLA_HG + warp; lane l holds latent dims l, l + 32, .. (up to
// MLA_MAXL, unrolled with a guard so they stay in registers) and RoPE dims l, l + 32, l + 64, l + 96.  Each step stages
// MLA_KU keys (latent, then RoPE key: BF16) in shared memory once for the block's heads.  Score = scale (ql . c + qr . k),
// online softmax, sums: f32.  n_splits > 1: partials [row][head][split] = (m, l, acc[r]) for k_mla_reduce;
// n_splits == 1: olat[row][head] = bf16(acc / l).
#define MLA_HG 8   // query heads per block (MLA_HG x 32 threads: room for the 64 accumulators per lane)
__global__ void __launch_bounds__(32 * MLA_HG) k_mla_attn(MlaArgs a, const float* ql, const float* qr, KvView kv, const RowInfo* ri,
                                                         float* out) {
    __shared__ __align__(16) uint16_t sh[MLA_KU * (32 * MLA_MAXL + ATT_HD)];
    const int lane = threadIdx.x & 31, tid = threadIdx.x, ntg = (int) blockDim.x;
    const int split = (int) blockIdx.x, t = (int) blockIdx.z, r = a.r, nl = r / 32, kw = r + ATT_HD;
    const int h = (int) blockIdx.y * MLA_HG + (threadIdx.x >> 5);
    const int pos = ri[t].pos;
    const int nsr = min(a.n_splits, (pos + a.chunk) / a.chunk);
    const int chunk = (pos + nsr) / nsr;
    const int p0 = split * chunk, p1 = min(pos + 1, p0 + chunk);
    const bool live = h < a.n_head;
    float q[MLA_MAXL], acc[MLA_MAXL], qp[4];
    const float* qlh = ql + ((size_t) t * a.n_head + (live ? h : 0)) * r;
    const float* qrh = qr + ((size_t) t * a.n_head + (live ? h : 0)) * ATT_HD;
#pragma unroll
    for (int j = 0; j < MLA_MAXL; ++j) {
        q[j] = j < nl ? qlh[lane + 32 * j] * a.scale : 0.0f;
        acc[j] = 0;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) qp[j] = qrh[lane + 32 * j] * a.scale;
    float m = -INFINITY, l = 0;
    for (int p = p0; p < p1; p += MLA_KU) {
        const int nk = min(MLA_KU, p1 - p);
        __syncthreads();
        for (int e = tid; e < nk * kw / 8; e += ntg) {   // 8 dims at a time (r and 128: multiples of 8)
            const int u = e / (kw / 8), d = (e - u * (kw / 8)) * 8;
            int row;
            const KvSeg sg = kv_seg(kv, ri[t].kv0 + p + u, &row);
            *(uint4*) &sh[u * kw + d] = d < r ? mla_get8(kv.fmt, sg, row, 0, r, d) : mla_get8(kv.fmt, sg, row, 1, r, d - r);
        }
        __syncthreads();
        float s[MLA_KU];
        float mx = m;
#pragma unroll
        for (int u = 0; u < MLA_KU; ++u) {
            float d = 0;
            if (u < nk) {
                const uint16_t* kk = sh + u * kw;
#pragma unroll
                for (int j = 0; j < MLA_MAXL; ++j)
                    if (j < nl) d += q[j] * bf(kk[lane + 32 * j]);
#pragma unroll
                for (int j = 0; j < 4; ++j) d += qp[j] * bf(kk[r + lane + 32 * j]);
            }
            d = warp_sum(d);
            s[u] = u < nk ? d : -INFINITY;
            mx = fmaxf(mx, s[u]);
        }
        const float cor = expf(m - mx);
        l *= cor;
#pragma unroll
        for (int j = 0; j < MLA_MAXL; ++j) acc[j] *= cor;
#pragma unroll
        for (int u = 0; u < MLA_KU; ++u) {
            if (u >= nk) break;
            const float e = expf(s[u] - mx);
            l += e;
            const uint16_t* kk = sh + u * kw;
#pragma unroll
            for (int j = 0; j < MLA_MAXL; ++j)
                if (j < nl) acc[j] += e * bf(kk[lane + 32 * j]);
        }
        m = mx;
    }
    if (!live) return;
    if (a.n_splits == 1) {
        float* o = out + ((size_t) t * a.n_head + h) * r;
#pragma unroll
        for (int j = 0; j < MLA_MAXL; ++j)
            if (j < nl) o[lane + 32 * j] = bfr(acc[j] / l);
    } else {
        float* pp = out + (((size_t) t * a.n_head + h) * a.n_splits + split) * (r + 2);
        if (lane == 0) { pp[0] = m; pp[1] = l; }
#pragma unroll
        for (int j = 0; j < MLA_MAXL; ++j)
            if (j < nl) pp[2 + lane + 32 * j] = acc[j];
    }
}
// Reduce the latent attention's splits: olat[t][h][d] = bf16(acc / l).  Block (head, row); threads stride d.
__global__ void __launch_bounds__(128) k_mla_reduce(MlaArgs a, const float* part, float* olat) {
    const int h = (int) blockIdx.x, t = (int) blockIdx.y, r = a.r, w = r + 2;
    const float* pp = part + ((size_t) t * a.n_head + h) * a.n_splits * w;
    float m = -INFINITY;
    for (int s = 0; s < a.n_splits; ++s) m = fmaxf(m, pp[s * w]);
    for (int d = (int) (blockIdx.z * blockDim.x + threadIdx.x); d < r; d += (int) (gridDim.z * blockDim.x)) {
        float l = 0, acc = 0;
        for (int s = 0; s < a.n_splits; ++s) {
            const float ms = pp[s * w];
            if (ms == -INFINITY) continue;
            const float c = expf(ms - m);
            l += pp[s * w + 1] * c;
            acc += pp[s * w + 2 + d] * c;
        }
        olat[((size_t) t * a.n_head + h) * r + d] = bfr(acc / l);
    }
}

// Per-head maps of prompt rows (T > MV_MAXT) on tensor cores: head blockIdx.z's [T x I] x [I x O] through the dense
// GEMM body (the inputs are BF16 values and the weights BF16 values, Q8 / Q4 dequantized as wblock: exact products, f32
// sums, as k_heads_mv), the gate fused.  A head's streams: its O x I values (codes), scales and biases.
__global__ void __launch_bounds__(MMT_THREADS) k_heads_mm(HmvArgs h, MmArgs a, int fmt, WSlice W, const float* X, const float* g,
                                                          float* Y) {
    const size_t n = (size_t) h.O * h.I * (size_t) blockIdx.z;   // values before this head
    WSlice w;
    memset(&w, 0, sizeof w);
    w.p[0] = (const uint8_t*) W.p[0] + (fmt == MF_BF16 ? 2 * n : fmt == MF_Q8 ? n : n / 2);
    if (fmt != MF_BF16) { w.p[1] = (const uint16_t*) W.p[1] + n / 64; w.p[2] = (const uint16_t*) W.p[2] + n / 64; }
    mm_body<1>(fmt, a, w, NULL, X + (size_t) blockIdx.z * h.hs, Y + (size_t) blockIdx.z * h.O, NULL, NULL, NULL, NULL,
               h.gate ? g + (size_t) blockIdx.z * h.O : NULL);
}

// Latent attention on tensor cores (absorbed MLA is multi-query attention over the latent: every head reads the same
// keys).  Block (split, group of 16 MT query heads, row): the row's heads share each tile of MLAT_KT keys (latent c, then
// the RoPE key: [key][r + 128] BF16), double-buffered in shared memory by cp.async.  16 warps, NSL = 16 / MT slices:
//   scores  S = Q K^T: warp (M-tile w % MT, k-slice w / MT) sums its share of the r + 128 dims (BF16 mma, f32 sums: the
//           queries are BF16 values, exact); the NSL partial sums meet in shared memory; score = scale (q . k);
//   softmax one thread per (head, key): online softmax in f32, as k_mla_attn;
//   P c     warp (M-tile w % MT, column group w / MT) owns the 16-dim column pairs w / MT, + NSL, ..; P enters the mma as
//           two BF16 terms hi + lo (= P - hi), so P c keeps about 16 bits of P (f32 P in k_mla_attn and on Metal).
// MT = 2 (32 heads a block) for one-pass rows: each key tile serves twice the queries.  MT = 1 for split-key decode: a
// row's two head groups are separate blocks (the second's tile reads hit L2), so a decode step fills twice the SMs.
// Output and partials as k_mla_attn.  r <= MLAT_MAXR (kc_mla_attn falls back to k_mla_attn above).
#define MLAT_KT 16
#define MLAT_W 16
#define MLAT_MAXR 768
#define MLAT_PLD (MLAT_KT + 8)
#ifndef MLAT_PLO
#define MLAT_PLO 1   // 1: P as BF16 hi + lo terms (about 16 bits); 0: P rounded to BF16 (one MMA)
#endif
static __device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t) __cvta_generic_to_shared(dst)), "l"(src), "r"(bytes));
}
static __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
static __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
static __device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t) __cvta_generic_to_shared(p)));
}
static __device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t) __cvta_generic_to_shared(p)));
}
static size_t mlat_smem(int r, int mt) {
    const int kld = r + ATT_HD + 8, q = 16 * mt;
    return (size_t) 2 * MLAT_KT * kld * 2 + (size_t) MLAT_W * 16 * MLAT_KT * 4 + (size_t) 2 * q * MLAT_PLD * 2 + 3 * q * 4;
}
template <int MT>
__global__ void __launch_bounds__(32 * MLAT_W) k_mla_attn_tc(MlaArgs a, const float* ql, const float* qr, KvView kv, const RowInfo* ri,
                                                            float* out) {
    constexpr int Q = 16 * MT, NSL = MLAT_W / MT;
    constexpr int SPS = ((MLAT_MAXR + ATT_HD) / 16 + NSL - 1) / NSL;   // k-steps per slice, at most
    constexpr int NP = MLAT_MAXR / 16 / NSL;                           // column pairs per warp, at most
    extern __shared__ __align__(16) uint8_t mlat_sm[];
    const int r = a.r, D = r + ATT_HD, kld = D + 8, nch = D / 8, H = a.n_head;
    uint16_t* Ks = (uint16_t*) mlat_sm;                      // [2][KT][kld]
    float* Sp = (float*) (Ks + 2 * MLAT_KT * kld);           // [NSL][Q][KT] partial scores
    uint16_t* Ph = (uint16_t*) (Sp + NSL * Q * MLAT_KT);     // [Q][PLD] P: BF16 hi
    uint16_t* Pl = Ph + Q * MLAT_PLD;                        //             and lo
    float* scor = (float*) (Pl + Q * MLAT_PLD);              // [Q] the tile's rescale
    float* sl = scor + Q;                                    // [Q] final l
    float* sm = sl + Q;                                      // [Q] final m
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gq = lane >> 2, t4 = lane & 3;
    const int split = (int) blockIdx.x, h0 = (int) blockIdx.y * Q, t = (int) blockIdx.z;
    const int pos = ri[t].pos, kv0 = ri[t].kv0;
    const int nsr = min(a.n_splits, (pos + a.chunk) / a.chunk);
    const int chunk = (pos + nsr) / nsr;
    const int p0 = split * chunk, p1 = min(pos + 1, p0 + chunk);
    const int ntile = p1 > p0 ? (p1 - p0 + MLAT_KT - 1) / MLAT_KT : 0;
    const int mt = w % MT, ks = w / MT;
    // queries: this warp's k-slice of its 16 heads, as A fragments (BF16 values: exact)
    const int nsteps = D / 16, sps = (nsteps + NSL - 1) / NSL, s0 = ks * sps, ns_w = max(0, min(sps, nsteps - s0));
    uint32_t qf[SPS][4];
    {
        const int ha = h0 + mt * 16 + gq, hb = ha + 8;
#pragma unroll
        for (int i = 0; i < SPS; ++i) {
            const int c = (s0 + i) * 16 + t4 * 2;
            float va[4] = {0, 0, 0, 0}, vb[4] = {0, 0, 0, 0};
            if (i < ns_w) {
                if (ha < H) {
                    const float* q = c < r ? ql + ((size_t) t * H + ha) * r + c : qr + ((size_t) t * H + ha) * ATT_HD + c - r;
                    va[0] = q[0]; va[1] = q[1]; va[2] = q[8]; va[3] = q[9];
                }
                if (hb < H) {
                    const float* q = c < r ? ql + ((size_t) t * H + hb) * r + c : qr + ((size_t) t * H + hb) * ATT_HD + c - r;
                    vb[0] = q[0]; vb[1] = q[1]; vb[2] = q[8]; vb[3] = q[9];
                }
            }
            qf[i][0] = pack_bf2(va[0], va[1]);
            qf[i][1] = pack_bf2(vb[0], vb[1]);
            qf[i][2] = pack_bf2(va[2], va[3]);
            qf[i][3] = pack_bf2(vb[2], vb[3]);
        }
    }
    const int npair = r / 16, cg = w / MT;
    float acc[NP][2][4];
#pragma unroll
    for (int i = 0; i < NP; ++i)
#pragma unroll
        for (int n = 0; n < 2; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0;
    const int sq = tid >> 4, su = tid & 15;   // softmax: (head, key)
    float m_run = -INFINITY, l_run = 0;       // l_run: this thread's key column (the head's sum at the end)
    // this thread's 16-byte chunks of a tile (the same in every tile): key u, column j (packed u << 16 | j)
    int cmap[(MLAT_KT * (MLAT_MAXR + ATT_HD) / 8 + 32 * MLAT_W - 1) / (32 * MLAT_W)];
#pragma unroll
    for (int k = 0; k < (int) (sizeof cmap / sizeof cmap[0]); ++k) {
        const int c = tid + k * 32 * MLAT_W, u = c / nch;
        cmap[k] = c < MLAT_KT * nch ? u << 16 | (c - u * nch) : -1;
    }
    auto load_tile = [&](int buf, int p) {
#pragma unroll
        for (int k = 0; k < (int) (sizeof cmap / sizeof cmap[0]); ++k) {
            if (cmap[k] < 0) continue;
            const int u = cmap[k] >> 16, j = cmap[k] & 0xFFFF, kp = p + u;
            uint16_t* dst = Ks + ((size_t) buf * MLAT_KT + u) * kld + j * 8;
            if (kv.fmt != KV_BF16) {   // FP8 / FP4: decoded to BF16 by plain stores (the next tile barrier publishes them)
                uint4 w = make_uint4(0, 0, 0, 0);
                if (kp < p1) {
                    int row;
                    const KvSeg sg = kv_seg(kv, kv0 + kp, &row);
                    w = j * 8 < r ? mla_get8(kv.fmt, sg, row, 0, r, j * 8) : mla_get8(kv.fmt, sg, row, 1, r, j * 8 - r);
                }
                *(uint4*) dst = w;
                continue;
            }
            const void* src = kv.a.k;
            int ok = 0;
            if (kp < p1) {
                int row;
                const KvSeg sg = kv_seg(kv, kv0 + kp, &row);
                src = j * 8 < r ? (const void*) ((const uint16_t*) sg.v + (size_t) row * r + j * 8)
                                : (const void*) ((const uint16_t*) sg.k + (size_t) row * ATT_HD + j * 8 - r);
                ok = 16;
            }
            cp_async16(dst, src, ok);   // keys past the split: zeros
        }
        cp_async_commit();
    };
    if (ntile) load_tile(0, p0);
    // Three barriers per tile: (A) the tile has landed and every warp is done with the previous one, so the other buffer
    // takes the next tile while this one computes; (B) partial scores written; (C) P and the rescale written.  The next
    // tile's scores (after its A) overwrite Sp, P and scor only once this tile's P c has read them.
    for (int it = 0; it < ntile; ++it) {
        const int buf = it & 1, p = p0 + it * MLAT_KT;
        cp_async_wait<0>();
        __syncthreads();
        if (it + 1 < ntile) load_tile(buf ^ 1, p + MLAT_KT);
        const uint16_t* Kb = Ks + (size_t) buf * MLAT_KT * kld;
        {   // partial scores of this warp's k-slice
            float sc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
            for (int i = 0; i < SPS; ++i) {
                if (i < ns_w) {
                    uint32_t b[4];
                    ldsm_x4(b, Kb + (size_t) ((lane >> 4) * 8 + (lane & 7)) * kld + (s0 + i) * 16 + ((lane >> 3) & 1) * 8);
                    mma_bf16(sc[0], qf[i], b);
                    mma_bf16(sc[1], qf[i], b + 2);
                }
            }
            float* sp = Sp + (ks * Q + mt * 16) * MLAT_KT;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                *(float2*) &sp[gq * MLAT_KT + n * 8 + t4 * 2] = make_float2(sc[n][0], sc[n][1]);
                *(float2*) &sp[(gq + 8) * MLAT_KT + n * 8 + t4 * 2] = make_float2(sc[n][2], sc[n][3]);
            }
        }
        __syncthreads();
        if (sq < Q) {   // online softmax: thread (head sq, key su); a head's 16 keys are one half-warp (MT = 1: 8 warps)
            float s = 0;
#pragma unroll
            for (int k = 0; k < NSL; ++k) s += Sp[(k * Q + sq) * MLAT_KT + su];
            const bool ok = p + su < p1;
            s = ok ? s * a.scale : -INFINITY;
            float mx = s;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
            mx = fmaxf(mx, m_run);
            const float cor = expf(m_run - mx), e = ok ? expf(s - mx) : 0.0f;
            l_run = l_run * cor + e;
            m_run = mx;
            const uint16_t hi = tobf(e);
            Ph[sq * MLAT_PLD + su] = hi;
            Pl[sq * MLAT_PLD + su] = tobf(e - bf(hi));
            if (su == 0) scor[sq] = cor;
        }
        __syncthreads();
        {   // acc = acc * cor + P c
            const float ca = scor[mt * 16 + gq], cb = scor[mt * 16 + gq + 8];
            const bool rescale = __any_sync(FULL, ca != 1.0f || cb != 1.0f);   // no running max moved: no rescale
            uint32_t ah[4], al[4];
            const int prow = mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, pcol = (lane >> 4) * 8;
            ldsm_x4(ah, Ph + prow * MLAT_PLD + pcol);
            if (MLAT_PLO) ldsm_x4(al, Pl + prow * MLAT_PLD + pcol);
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const int pi = cg + NSL * i;
                if (pi < npair) {
                    uint32_t b[4];
                    ldsm_x4_t(b, Kb + (size_t) ((lane & 7) + ((lane >> 3) & 1) * 8) * kld + pi * 16 + (lane >> 4) * 8);
#pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        if (rescale) { acc[i][n][0] *= ca; acc[i][n][1] *= ca; acc[i][n][2] *= cb; acc[i][n][3] *= cb; }
                        mma_bf16(acc[i][n], ah, b + 2 * n);
                        if (MLAT_PLO) mma_bf16(acc[i][n], al, b + 2 * n);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) l_run += __shfl_xor_sync(FULL, l_run, o);
    if (su == 0 && sq < Q) { sl[sq] = l_run; sm[sq] = m_run; }
    __syncthreads();
    const int qa = mt * 16 + gq, qb = qa + 8;
    if (a.n_splits == 1) {
        const float la = sl[qa], lb = sl[qb];
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int pi = cg + NSL * i;
            if (pi < npair) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int d = pi * 16 + n * 8 + t4 * 2;
                    if (h0 + qa < H)
                        *(float2*) &out[((size_t) t * H + h0 + qa) * r + d] = make_float2(bfr(acc[i][n][0] / la), bfr(acc[i][n][1] / la));
                    if (h0 + qb < H)
                        *(float2*) &out[((size_t) t * H + h0 + qb) * r + d] = make_float2(bfr(acc[i][n][2] / lb), bfr(acc[i][n][3] / lb));
                }
            }
        }
    } else {
        const size_t w2 = (size_t) r + 2;
        if (tid < Q && h0 + tid < H) {
            float* pp = out + (((size_t) t * H + h0 + tid) * a.n_splits + split) * w2;
            pp[0] = sm[tid];
            pp[1] = sl[tid];
        }
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int pi = cg + NSL * i;
            if (pi < npair) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int d = pi * 16 + n * 8 + t4 * 2;
                    if (h0 + qa < H) {
                        float* pp = out + (((size_t) t * H + h0 + qa) * a.n_splits + split) * w2 + 2 + d;
                        pp[0] = acc[i][n][0];
                        pp[1] = acc[i][n][1];
                    }
                    if (h0 + qb < H) {
                        float* pp = out + (((size_t) t * H + h0 + qb) * a.n_splits + split) * w2 + 2 + d;
                        pp[0] = acc[i][n][2];
                        pp[1] = acc[i][n][3];
                    }
                }
            }
        }
    }
}

// ---- MLA prompt rows, not absorbed ------------------------------------------------------------------------------------
// Absorbed attention costs every query row a pass over each key's whole latent (r + 128 dims for scores, r for P c); for
// prompt rows the keys are decompressed per head once instead (as DeepSeek's MLA prefill does), and attention is plain
// multi-head attention over 128 + 128 dims (q . Kn + RoPE parts) with 128-dim values, the gate fused:
//   Kn[key][h] = bf16(q_lat_h^T c[key])   (so q_h . Kn_h = (q_lat_h q_h) . c without q_lat q's rounding)
//   Vd[key][h] = bf16(v_up_h c[key])      (so sum p Vd_h = v_up_h (sum p c) without olat's rounding)
// The rounding points move (the decompressed K and V are BF16 tensors; the absorbed path rounds q_lat q and the latent
// output instead); held-out KLD against the BF16 reference is unchanged (README).

// Decompression: block (64 keys, head, K or V), 8 warps of 16 keys x 64 dims, BF16 mma over the latent in steps of 32.
// Keys [k_lo, k_hi) of the slot at cache row kv0 into Kn / Vd rows key - kb0.
#define MLD_BK 64
#define MLD_BJ 32
#define MLD_ALD (MLD_BJ + 8)
#define MLD_KLD (ATT_HD + 8)   // K weights staged [j][d]
#define MLD_VLD (MLD_BJ + 8)   // V weights staged [d][j]
__global__ void __launch_bounds__(256) k_mla_decomp(MlaArgs a, KvView kv, int kv0, int kb0, int k_lo, int k_hi, int qfmt, WSlice Wql,
                                                   int vfmt, WSlice Wvu, uint16_t* Kn, uint16_t* Vd) {
    __shared__ __align__(16) uint16_t As[MLD_BK * MLD_ALD];
    __shared__ __align__(16) uint16_t Bs[MLD_BJ * MLD_KLD > ATT_HD * MLD_VLD ? MLD_BJ * MLD_KLD : ATT_HD * MLD_VLD];
    const int h = (int) blockIdx.y, isv = (int) blockIdx.z, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int gq = lane >> 2, t4 = lane & 3, r = a.r, H = a.n_head;
    const int key0 = k_lo + (int) blockIdx.x * MLD_BK, wk = (w & 3) * 16, wd = (w >> 2) * 64;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0;
    // this thread's latent row (one 16-byte chunk per step)
    const int au = tid >> 2, aj = (tid & 3) * 8, akp = key0 + au;
    int arow = -1;
    KvSeg asg = kv.a;
    if (akp < k_hi) asg = kv_seg(kv, kv0 + akp, &arow);
    for (int j0 = 0; j0 < r; j0 += MLD_BJ) {
        *(uint4*) &As[au * MLD_ALD + aj] = arow >= 0 ? mla_get8(kv.fmt, asg, arow, 0, r, j0 + aj) : make_uint4(0, 0, 0, 0);
        if (!isv)
            for (int c = tid; c < MLD_BJ * ATT_HD / 8; c += 256) {   // Bs[j][d] = q_lat[h][j0 + j][d]
                const int j = c >> 4, d8 = (c & 15) * 8;
                *(uint4*) &Bs[j * MLD_KLD + d8] = welem8(qfmt, Wql, ((size_t) h * r + j0 + j) * ATT_HD + d8);
            }
        else
            for (int c = tid; c < MLD_BJ * ATT_HD / 8; c += 256) {   // Bs[d][j] = v_up[h][d][j0 + j]
                const int d = c >> 2, j8 = (c & 3) * 8;
                *(uint4*) &Bs[d * MLD_VLD + j8] = welem8(vfmt, Wvu, ((size_t) h * ATT_HD + d) * r + j0 + j8);
            }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < MLD_BJ; ks += 16) {
            uint32_t af[4];
            ldsm_x4(af, &As[(wk + (lane & 7) + ((lane >> 3) & 1) * 8) * MLD_ALD + ks + (lane >> 4) * 8]);
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                const int d0 = wd + np * 16;
                uint32_t b[4];
                if (!isv) ldsm_x4_t(b, &Bs[(ks + (lane & 7) + ((lane >> 3) & 1) * 8) * MLD_KLD + d0 + (lane >> 4) * 8]);
                else ldsm_x4(b, &Bs[(d0 + (lane >> 4) * 8 + (lane & 7)) * MLD_VLD + ks + ((lane >> 3) & 1) * 8]);
                mma_bf16(acc[2 * np], af, b);
                mma_bf16(acc[2 * np + 1], af, b + 2);
            }
        }
        __syncthreads();
    }
    uint16_t* out = isv ? Vd : Kn;
    const int ka = key0 + wk + gq, kb = ka + 8;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int d = wd + nt * 8 + t4 * 2;
        if (ka < k_hi) *(uint32_t*) &out[((size_t) (ka - kb0) * H + h) * ATT_HD + d] = pack_bf2(acc[nt][0], acc[nt][1]);
        if (kb < k_hi) *(uint32_t*) &out[((size_t) (kb - kb0) * H + h) * ATT_HD + d] = pack_bf2(acc[nt][2], acc[nt][3]);
    }
}

// Attention of prompt rows over keys [kb0, kb1) (decompressed in Kn / Vd rows key - kb0): block (64 consecutive rows of
// one slot, head), a warp per 16 rows; keys in tiles of MLP_BK through shared memory.  Scores scale (q . Kn + qr . kr)
// from BF16 operands (exact products, f32 sums), causal online softmax in f32, P as BF16 hi + lo (as k_mla_attn_tc).
// first: start the softmax state, else read it from st ([T][H][130]: m, l, O); last: o = bf16(bf16(O / l) *
// softplus_ln2(g)) (k_attn_reduce's gate), else write the state.
#define MLP_RW 4
#define MLP_BK 32
#define MLP_KLD (2 * ATT_HD + 8)
#define MLP_VLD (ATT_HD + 8)
__global__ void __launch_bounds__(32 * MLP_RW) k_mla_prefill(MlaArgs a, const float* q, const float* qr, KvView kv, const RowInfo* ri,
                                                            const uint16_t* Kn, const uint16_t* Vd, int kb0, int kb1, int first,
                                                            int last, float* st, const float* g, float* o, int T) {
    __shared__ __align__(16) uint16_t Ks[MLP_BK * MLP_KLD];
    __shared__ __align__(16) uint16_t Vs[MLP_BK * MLP_VLD];
    const int H = a.n_head, h = (int) blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, t4 = lane & 3;
    const int b0 = (int) blockIdx.x * 16 * MLP_RW, tw = b0 + w * 16, live = tw < T;
    const int t0 = live ? tw : T - 1, nrows = live ? min(16, T - tw) : 1;
    const int ra = t0 + min(gq, nrows - 1), rb = t0 + min(gq + 8, nrows - 1);
    const int pa = ri[ra].pos, pb = ri[rb].pos, plast = ri[t0 + nrows - 1].pos, kv0 = ri[b0].kv0;
    const int kend = min(kb1, ri[min(b0 + 16 * MLP_RW, T) - 1].pos + 1);
    uint32_t qf[16][4];   // 8 steps of q (128 dims), 8 of the roped query part
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const float* src = s < 8 ? q : qr;
        const float* qa = src + ((size_t) ra * H + h) * ATT_HD;
        const float* qb = src + ((size_t) rb * H + h) * ATT_HD;
        const int c = (s & 7) * 16 + t4 * 2;
        qf[s][0] = pack_bf2(qa[c], qa[c + 1]);
        qf[s][1] = pack_bf2(qb[c], qb[c + 1]);
        qf[s][2] = pack_bf2(qa[c + 8], qa[c + 9]);
        qf[s][3] = pack_bf2(qb[c + 8], qb[c + 9]);
    }
    float O[16][4], ma = -INFINITY, mb = -INFINITY, la = 0, lb = 0;
    const float* sa_ = st + ((size_t) ra * H + h) * (ATT_HD + 2);
    const float* sb_ = st + ((size_t) rb * H + h) * (ATT_HD + 2);
    if (first) {
#pragma unroll
        for (int j = 0; j < 16; ++j) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0;
    } else {
        ma = sa_[0]; la = sa_[1]; mb = sb_[0]; lb = sb_[1];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d = 2 + j * 8 + t4 * 2;
            O[j][0] = sa_[d]; O[j][1] = sa_[d + 1]; O[j][2] = sb_[d]; O[j][3] = sb_[d + 1];
        }
    }
    for (int k0 = kb0; k0 < kend; k0 += MLP_BK) {
        for (int c = threadIdx.x; c < MLP_BK * 2 * ATT_HD / 8; c += 32 * MLP_RW) {   // [Kn | kr] per key
            const int u = c >> 5, j = c & 31, kp = k0 + u;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (kp < kend) {
                if (j < 16) v = *(const uint4*) (Kn + ((size_t) (kp - kb0) * H + h) * ATT_HD + j * 8);
                else {
                    int row;
                    const KvSeg sg = kv_seg(kv, kv0 + kp, &row);
                    v = mla_get8(kv.fmt, sg, row, 1, 0, (j - 16) * 8);
                }
            }
            *(uint4*) &Ks[u * MLP_KLD + j * 8] = v;
        }
        for (int c = threadIdx.x; c < MLP_BK * ATT_HD / 8; c += 32 * MLP_RW) {
            const int u = c >> 4, j = c & 15, kp = k0 + u;
            *(uint4*) &Vs[u * MLP_VLD + j * 8] =
                kp < kend ? *(const uint4*) (Vd + ((size_t) (kp - kb0) * H + h) * ATT_HD + j * 8) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (live && k0 <= plast) {
            float S[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) S[j][0] = S[j][1] = S[j][2] = S[j][3] = 0;
#pragma unroll
            for (int s = 0; s < 16; ++s)
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    uint32_t b[4];
                    ldsm_x4(b, &Ks[(np * 16 + (lane >> 4) * 8 + (lane & 7)) * MLP_KLD + s * 16 + ((lane >> 3) & 1) * 8]);
                    mma_bf16(S[2 * np], qf[s], b);
                    mma_bf16(S[2 * np + 1], qf[s], b + 2);
                }
            float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = k0 + j * 8 + t4 * 2;
                S[j][0] = key > pa ? -INFINITY : S[j][0] * a.scale;
                S[j][1] = key + 1 > pa ? -INFINITY : S[j][1] * a.scale;
                S[j][2] = key > pb ? -INFINITY : S[j][2] * a.scale;
                S[j][3] = key + 1 > pb ? -INFINITY : S[j][3] * a.scale;
                mxa = fmaxf(mxa, fmaxf(S[j][0], S[j][1]));
                mxb = fmaxf(mxb, fmaxf(S[j][2], S[j][3]));
            }
            mxa = fmaxf(mxa, __shfl_xor_sync(FULL, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(FULL, mxa, 2));
            mxb = fmaxf(mxb, __shfl_xor_sync(FULL, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(FULL, mxb, 2));
            const float na = fmaxf(ma, mxa), nb = fmaxf(mb, mxb);
            const float ca = na == -INFINITY ? 1.0f : expf(ma - na), cb = nb == -INFINITY ? 1.0f : expf(mb - nb);
            float sa = 0, sb = 0;
            uint32_t ph[2][4], pl[2][4];   // P as A fragments (2 key steps of 16): BF16 hi and lo
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p0 = S[j][0] == -INFINITY ? 0.0f : expf(S[j][0] - na);
                const float p1 = S[j][1] == -INFINITY ? 0.0f : expf(S[j][1] - na);
                const float p2 = S[j][2] == -INFINITY ? 0.0f : expf(S[j][2] - nb);
                const float p3 = S[j][3] == -INFINITY ? 0.0f : expf(S[j][3] - nb);
                sa += p0 + p1;
                sb += p2 + p3;
                const uint16_t h0 = tobf(p0), h1 = tobf(p1), h2 = tobf(p2), h3 = tobf(p3);
                ph[j >> 1][(j & 1) * 2] = (uint32_t) h0 | (uint32_t) h1 << 16;
                ph[j >> 1][(j & 1) * 2 + 1] = (uint32_t) h2 | (uint32_t) h3 << 16;
                pl[j >> 1][(j & 1) * 2] = pack_bf2(p0 - bf(h0), p1 - bf(h1));
                pl[j >> 1][(j & 1) * 2 + 1] = pack_bf2(p2 - bf(h2), p3 - bf(h3));
            }
            sa += __shfl_xor_sync(FULL, sa, 1); sa += __shfl_xor_sync(FULL, sa, 2);
            sb += __shfl_xor_sync(FULL, sb, 1); sb += __shfl_xor_sync(FULL, sb, 2);
            la = la * ca + sa;
            lb = lb * cb + sb;
            ma = na;
            mb = nb;
#pragma unroll
            for (int j = 0; j < 16; ++j) { O[j][0] *= ca; O[j][1] *= ca; O[j][2] *= cb; O[j][3] *= cb; }
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int dp = 0; dp < 8; ++dp) {
                    uint32_t b[4];
                    ldsm_x4_t(b, &Vs[(s * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * MLP_VLD + dp * 16 + (lane >> 4) * 8]);
                    mma_bf16(O[2 * dp], ph[s], b);
                    mma_bf16(O[2 * dp], pl[s], b);
                    mma_bf16(O[2 * dp + 1], ph[s], b + 2);
                    mma_bf16(O[2 * dp + 1], pl[s], b + 2);
                }
        }
        __syncthreads();
    }
    if (!live) return;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int q2 = 0; q2 < 4; ++q2) {
            const int rr = (q2 >> 1) ? gq + 8 : gq;
            if (rr >= nrows) continue;
            const int t = t0 + rr, d = j * 8 + t4 * 2 + (q2 & 1);
            if (last) {
                const size_t i = ((size_t) t * H + h) * ATT_HD + d;
                o[i] = bfr(bfr(O[j][q2] / ((q2 >> 1) ? lb : la)) * softplus_gate(g[i]));
            } else {
                float* sp = st + ((size_t) t * H + h) * (ATT_HD + 2);
                sp[2 + d] = O[j][q2];
                if (j == 0 && t4 == 0 && (q2 & 1) == 0) { sp[0] = (q2 >> 1) ? mb : ma; sp[1] = (q2 >> 1) ? lb : la; }
            }
        }
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
                kk = kv_load8_bf(kv, 0, ri[b0].kv0 + p, kvh, a.n_kv, d8);   // a block's rows: one slot
                vv = kv_load8_bf(kv, 1, ri[b0].kv0 + p, kvh, a.n_kv, d8);
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
    mm_kernel(fmt)<<<dim3((unsigned) (R + MMT_BM - 1) / MMT_BM, (unsigned) (T + MMT_BN - 1) / MMT_BN), MMT_THREADS, 0, s>>>(a, w, NULL, x, y, NULL, NULL, NULL, G);
}
void kc_mm_grouped(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
                   int xdiv, const int32_t* perm, const MmTile* tiles, int ntiles, const uint32_t* G) {
    if (!ntiles) return;
    MmArgs a = {K, R, 0, xs, ys, xdiv, 0, 0};
    WSlice none;
    memset(&none, 0, sizeof none);
    mm_kernel(fmt)<<<dim3((unsigned) (R + MMT_BM - 1) / MMT_BM, (unsigned) ntiles), MMT_THREADS, 0, s>>>(a, none, ws, x, y, perm, tiles, NULL, G);
}
void kc_mm_grouped_dev(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
                       int xdiv, const int32_t* perm, const MmTile* tiles, const int32_t* ntiles, int max_tiles,
                       const uint32_t* G) {
    MmArgs a = {K, R, 0, xs, ys, xdiv, 0, 0};
    WSlice none;
    memset(&none, 0, sizeof none);
    mm_kernel(fmt)<<<dim3((unsigned) (R + MMT_BM - 1) / MMT_BM, (unsigned) max_tiles), MMT_THREADS, 0, s>>>(a, none, ws, x, y, perm, tiles, ntiles, G);
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
int kc_heads_mm(cudaStream_t s, HmvArgs a, int fmt, WSlice W, const float* x, const float* g, float* y, int T) {
    if (a.I % MMT_BK || a.xs % 4 || a.hs % 4) return -1;
    const MmArgs m = {a.I, a.O, T, a.xs, a.H * a.O, 1, a.gate ? 2 : 0, 0};
    k_heads_mm<<<dim3((unsigned) (a.O + MMT_BM - 1) / MMT_BM, (unsigned) (T + MMT_BN - 1) / MMT_BN, (unsigned) a.H), MMT_THREADS, 0, s>>>(
        a, m, fmt, W, x, g, y);
    return 0;
}
void kc_heads_mv(cudaStream_t s, HmvArgs a, int fmt, WSlice W, const float* x, const float* g, float* y, int T) {
    if (T > MV_MAXT && !kc_heads_mm(s, a, fmt, W, x, g, y, T)) return;   // more rows than a matvec takes: the GEMM
    k_heads_mv<<<dim3((unsigned) (a.O + HMV_ROWS - 1) / HMV_ROWS, (unsigned) a.H, (unsigned) T), 32 * HMV_ROWS, 0, s>>>(a, fmt, W, x, g ? g : x, y);
}
void kc_kv_f32(cudaStream_t s, int fmt, const uint8_t* codes, const uint8_t* scales, int n, int rows, float* y) {
    k_kv_f32<<<(unsigned) ((n * rows / 8 + 127) / 128), 128, 0, s>>>(fmt, codes, scales, n, rows, y);
}
void kc_mla_rope(cudaStream_t s, MlaArgs a, float* qr, const float* kr, const float* c, KvView kv, const RowInfo* ri,
                 const float* inv, int T) {
    const int nx = (a.n_head + 1) * 64 > a.r ? (a.n_head + 1) * 64 : a.r;
    k_mla_rope<<<dim3((unsigned) (nx + 63) / 64, (unsigned) T), 64, 0, s>>>(a, qr, kr, c, kv, ri, inv);
}
void kc_mla_attn(cudaStream_t s, MlaArgs a, const float* ql, const float* qr, KvView kv, const RowInfo* ri, float* part,
                 float* olat, int T) {
    static int tc = -1;
    if (tc < 0) {   // NSLM_MLA_SCALAR=1: the one-pass kernel (k_mla_attn) throughout
        tc = !getenv("NSLM_MLA_SCALAR") || !atoi(getenv("NSLM_MLA_SCALAR"));
        if (tc) {
            cudaFuncSetAttribute(k_mla_attn_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mlat_smem(MLAT_MAXR, 1));
            cudaFuncSetAttribute(k_mla_attn_tc<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mlat_smem(MLAT_MAXR, 2));
        }
    }
    if (tc && a.r <= MLAT_MAXR && a.r % 16 == 0) {
        float* o = a.n_splits > 1 ? part : olat;
        if (a.n_splits > 1)   // decode: 16 heads a block
            k_mla_attn_tc<1><<<dim3((unsigned) a.n_splits, (unsigned) (a.n_head + 15) / 16, (unsigned) T), 32 * MLAT_W, mlat_smem(a.r, 1), s>>>(
                a, ql, qr, kv, ri, o);
        else
            k_mla_attn_tc<2><<<dim3(1, (unsigned) (a.n_head + 31) / 32, (unsigned) T), 32 * MLAT_W, mlat_smem(a.r, 2), s>>>(a, ql, qr, kv, ri, o);
        if (a.n_splits > 1) k_mla_reduce<<<dim3((unsigned) a.n_head, (unsigned) T, (unsigned) (a.r + 127) / 128), 128, 0, s>>>(a, part, olat);
        return;
    }
    k_mla_attn<<<dim3((unsigned) a.n_splits, (unsigned) (a.n_head + MLA_HG - 1) / MLA_HG, (unsigned) T), 32 * MLA_HG, 0, s>>>(
        a, ql, qr, kv, ri, a.n_splits > 1 ? part : olat);
    if (a.n_splits > 1) k_mla_reduce<<<dim3((unsigned) a.n_head, (unsigned) T, (unsigned) (a.r + 127) / 128), 128, 0, s>>>(a, part, olat);
}
void kc_mla_decomp(cudaStream_t s, MlaArgs a, KvView kv, int kv0, int kb0, int k_lo, int k_hi, int qfmt, WSlice Wql,
                   int vfmt, WSlice Wvu, uint16_t* Kn, uint16_t* Vd) {
    if (k_hi <= k_lo) return;
    k_mla_decomp<<<dim3((unsigned) (k_hi - k_lo + MLD_BK - 1) / MLD_BK, (unsigned) a.n_head, 2), 256, 0, s>>>(a, kv, kv0, kb0, k_lo, k_hi,
                                                                                                         qfmt, Wql, vfmt, Wvu, Kn, Vd);
}
void kc_mla_prefill(cudaStream_t s, MlaArgs a, const float* q, const float* qr, KvView kv, const RowInfo* ri, const uint16_t* Kn,
                    const uint16_t* Vd, int kb0, int kb1, int first, int last, float* st, const float* g, float* o, int T) {
    k_mla_prefill<<<dim3((unsigned) (T + 16 * MLP_RW - 1) / (16 * MLP_RW), (unsigned) a.n_head), 32 * MLP_RW, 0, s>>>(
        a, q, qr, kv, ri, Kn, Vd, kb0, kb1, first, last, st, g, o, T);
}
void kc_cache_admit(cudaStream_t s, const CachePool* p, int sl, const int32_t* inds, int count, int flags) {
    k_cache_admit<<<1, ADMIT_THREADS, 0, s>>>(*p, sl, inds, count, flags);
}
void kc_cache_copy(cudaStream_t s, const CachePool* p, int blocks) { k_cache_copy<<<(unsigned) blocks, 512, 0, s>>>(*p); }
void kc_argmax(cudaStream_t s, const float* logits, int32_t* out, int V, int n) {
    k_argmax<<<(unsigned) n, 1024, 0, s>>>(logits, out, V);
}

}   // extern "C"
