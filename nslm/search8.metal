// nslm/search8.metal - the P = 8 seed search on the GPU (nslm/search8.h).  Every expression is nslm/lib_search8.c's, in
// the same order, without fast math or contraction: bit-identical results.
// Grid (column groups from a.g0, row tiles of S8_ROWS), S8_TPB threads, S8_BPT rows per thread; the seed table is
// built S8_NCH seeds at a time in threadgroup memory.  Function constant 0: SH holds a lower-triangular 8 x 8 A per
// column group (else sqrt(h) per input channel).
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#include "search8_gpu.h"

#define MAGIC 12582912.0f
#define LI(i, j) ((i) * ((i) + 1) / 2 + (j))
constant bool kFull [[function_constant(0)]];
constant float kR32 = 1.0f / 32767.0f;

static inline float pow2f(int e) { return as_type<float>((uint) (e + 127) << 23); }
static inline int flog2f(float x) { return (int) ((as_type<uint>(x) >> 23) & 255u) - 127; }
static inline float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }
static inline ushort lfsr_step(ushort s) {
    const ushort b = (ushort) ((s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1u);
    return (ushort) ((s >> 1) | (b << 15));
}
static inline float f2bf_f(float f) {
    uint u = as_type<uint>(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xFFFF0000u);
}

static inline void scaled_u(int s, thread const float (&sh)[8], thread const float (&A)[64], thread float (&U)[64]) {
    ushort st = (ushort) s;
    float u0[64];
    for (short k = 0; k < 64; ++k) { st = lfsr_step(st); u0[k] = (float) ((int) st - 32768) * kR32; }
    for (short c = 0; c < 8; ++c)
        for (short p = 0; p < 8; ++p) {
            if (kFull) {
                float acc = 0.0f;
                for (short k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[8 * k + p];
                U[8 * c + p] = acc;
            } else U[8 * c + p] = u0[8 * c + p] * sh[c];
        }
}

// U[64], L[36], 1/diag[8] at [100], ok at [108]
template <typename P>
static inline bool cholesky(thread const float (&U)[64], P L, P D) {
    for (short i = 0; i < 8; ++i)
        for (short j = 0; j <= i; ++j) {
            float g = 0.0f;
            for (short c = 0; c < 8; ++c) g = g + U[8 * c + i] * U[8 * c + j];
            for (short k = 0; k < j; ++k) g = g - L[LI(i, k)] * L[LI(j, k)];
            if (i == j) {
                if (!(g > 0.0f)) return false;
                L[LI(i, i)] = precise::sqrt(g);
                D[i] = 1.0f / L[LI(i, i)];
            } else L[LI(i, j)] = g * D[j];
        }
    return true;
}

static inline void seed_entry(int s, thread const float (&sh)[8], thread const float (&A)[64], threadgroup float * ent) {
    float U[64];
    scaled_u(s, sh, A, U);
    for (short k = 0; k < 64; ++k) ent[k] = U[k];
    ent[108] = cholesky(U, ent + 64, ent + 100) ? 1.0f : 0.0f;
}

template <typename P>
static inline void solve8(P L, P D, thread const float (&b)[8], thread float (&t)[8]) {
    float y[8];
    for (short i = 0; i < 8; ++i) {
        float v = b[i];
        for (short k = 0; k < i; ++k) v = v - L[LI(i, k)] * y[k];
        y[i] = v * D[i];
    }
    for (short i = 7; i >= 0; --i) {
        float v = y[i];
        for (short k = i + 1; k < 8; ++k) v = v - L[LI(k, i)] * t[k];
        t[i] = v * D[i];
    }
}

static inline float cand_err(threadgroup const float * L, thread const float (&q)[8], thread const float (&b)[8], float wn, float sc) {
    float qb = 0.0f, qgq = 0.0f;
    for (short p = 0; p < 8; ++p) qb = qb + q[p] * b[p];
    for (short i = 0; i < 8; ++i) {
        float z = 0.0f;
        for (short k = i; k < 8; ++k) z = z + L[LI(k, i)] * q[k];
        qgq = qgq + z * z;
    }
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
}

static inline float decoded_err8(thread const float (&x)[8], thread const float (&sh)[8], thread const float (&A)[64], int seed, int e,
                                 thread const int (&q)[8]) {
    ushort st = (ushort) seed;
    const float sc = kR32 * pow2f(e);
    float wv[8];
    for (short c = 0; c < 8; ++c) {
        int isum = 0;
        for (short p = 0; p < 8; ++p) { st = lfsr_step(st); isum += ((int) st - 32768) * q[p]; }
        wv[c] = f2bf_f((float) isum * sc);
    }
    float er = 0.0f;
    for (short c = 0; c < 8; ++c) {
        float v;
        if (kFull) { v = 0.0f; for (short k = 0; k <= c; ++k) v = v + A[c * 8 + k] * wv[k]; }
        else v = sh[c] * wv[c];
        const float d = x[c] - v;
        er = er + d * d;
    }
    return er;
}

kernel void k_seed_search8(constant Search8Args & a [[buffer(0)]], device const float * W [[buffer(1)]],
                           device const float * SH [[buffer(2)]], device ushort * seed_out [[buffer(3)]],
                           device uint * coef_out [[buffer(4)]], device uchar * ecode_out [[buffer(5)]],
                           device float * err_out [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float tab[S8_NCH * S8_ENT];
    const int g = a.g0 + (int) tg.x, ng = a.cols / 8, lo = a.bias, hi = a.bias + 15;
    float sh[8], A[64];
    for (short c = 0; c < 8; ++c) sh[c] = kFull ? 1.0f : SH[g * 8 + c];
    for (short k = 0; k < 64; ++k) A[k] = kFull ? SH[g * 64 + k] : 0.0f;
    float x[S8_BPT][8], wn[S8_BPT], best[S8_BPT];
    int bs[S8_BPT], be[S8_BPT], row[S8_BPT];
    uint bw[S8_BPT];   // the winner's coefficients (as the C reference recomputes them from its table entry)
    for (short j = 0; j < S8_BPT; ++j) {
        row[j] = (int) tg.y * S8_ROWS + j * S8_TPB + tid;
        const int r = min(row[j], a.rows - 1);
        for (short c = 0; c < 8; ++c) {
            if (kFull) {
                float acc = 0.0f;
                for (short k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * W[(ulong) r * a.cols + g * 8 + k];
                x[j][c] = acc;
            } else x[j][c] = W[(ulong) r * a.cols + g * 8 + c] * sh[c];
        }
        float n = 0.0f;
        for (short c = 0; c < 8; ++c) n = n + x[j][c] * x[j][c];
        wn[j] = n; best[j] = INFINITY; bs[j] = 1; be[j] = lo; bw[j] = 0;
    }
    for (int s0 = 1; s0 <= a.n_seeds; s0 += S8_NCH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < S8_NCH && s0 + tid <= a.n_seeds) seed_entry(s0 + tid, sh, A, tab + tid * S8_ENT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nk = min(S8_NCH, a.n_seeds - s0 + 1);
        for (int k = 0; k < nk; ++k) {
            threadgroup const float * U = tab + k * S8_ENT;
            if (U[108] == 0.0f) continue;
            threadgroup const float * L = U + 64, * D = U + 100;
            for (short j = 0; j < S8_BPT; ++j) {
                float b[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (short c = 0; c < 8; ++c)
                    for (short p = 0; p < 8; ++p) b[p] = b[p] + U[8 * c + p] * x[j][c];
                float t[8];
                solve8(L, D, b, t);
                float m = 0.0f;
                for (short p = 0; p < 8; ++p) m = fmax(m, fabs(t[p]));
                int e0 = flog2f(m) - 2;
                e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                for (int ci = 0; ci < a.n_exp; ++ci) {
                    int e = e0 + a.exp_delta[ci];
                    e = e < lo ? lo : (e > hi ? hi : e);
                    const float inv = pow2f(-e), sc = pow2f(e);
                    float q[8];
                    for (short p = 0; p < 8; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                    const float er = cand_err(L, q, b, wn[j], sc);
                    if (er < best[j]) {
                        best[j] = er; bs[j] = s0 + k; be[j] = e;
                        uint w = 0;
                        for (short p = 0; p < 8; ++p) w |= (uint) ((int) q[p] & 15) << (4 * p);
                        bw[j] = w;
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = 0; j < S8_BPT; ++j) {
        const int e = be[j];
        int q[8], bq[8];
        for (short p = 0; p < 8; ++p) bq[p] = q[p] = (int) (bw[j] << (28 - 4 * p)) >> 28;
        float bde = decoded_err8(x[j], sh, A, bs[j], e, q);
        if (a.refit)
            for (int d = 0; d < 6561; ++d) {
                int c8[8], dd = d;
                bool okq = true;
                for (short p = 0; p < 8; ++p) { c8[p] = q[p] + dd % 3 - 1; dd /= 3; okq = okq && c8[p] >= -8 && c8[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err8(x[j], sh, A, bs[j], e, c8);
                if (ee < bde) { bde = ee; for (short p = 0; p < 8; ++p) bq[p] = c8[p]; }
            }
        if (row[j] >= a.rows) continue;
        const ulong k = (ulong) row[j] * (ulong) ng + (ulong) g;
        uint cw = 0;
        for (short p = 0; p < 8; ++p) cw |= (uint) (bq[p] & 15) << (4 * p);
        seed_out[k] = (ushort) bs[j];
        coef_out[k] = cw;
        ecode_out[k] = (uchar) (e - a.bias);
        err_out[k] = bde;
    }
}
