// nslm/searchp.metal - the P = 3 / 8 seed search on the GPU (nslm/searchp.h).  Every expression is nslm/lib_searchp.c's,
// in the same order, without fast math or contraction: bit-identical results.
// Grid (column groups from a.g0, row tiles of SP_ROWS), SP_TPB threads, SP_BPT rows per thread; the seed table is
// built SP_NCH seeds at a time in threadgroup memory.  Function constants: 0, SH holds a lower-triangular 8 x 8 A per
// column group (else sqrt(h) per input channel); 1, P (3 or 8).
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#include "searchp_gpu.h"

#define MAGIC 12582912.0f
#define LI(i, j) ((i) * ((i) + 1) / 2 + (j))
#define MP 8
constant bool kFull [[function_constant(0)]];
constant int kP [[function_constant(1)]];
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

static inline void scaled_u(int s, thread const float (&sh)[8], thread const float (&A)[64], thread float (&U)[8 * MP]) {
    ushort st = (ushort) s;
    float u0[8 * MP];
    for (short k = 0; k < 8 * kP; ++k) { st = lfsr_step(st); u0[k] = (float) ((int) st - 32768) * kR32; }
    for (short c = 0; c < 8; ++c)
        for (short p = 0; p < kP; ++p) {
            if (kFull) {
                float acc = 0.0f;
                for (short k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[kP * k + p];
                U[kP * c + p] = acc;
            } else U[kP * c + p] = u0[kP * c + p] * sh[c];
        }
}

template <typename P>
static inline bool cholesky(thread const float (&U)[8 * MP], P L, P D) {
    for (short i = 0; i < kP; ++i)
        for (short j = 0; j <= i; ++j) {
            float g = 0.0f;
            for (short c = 0; c < 8; ++c) g = g + U[kP * c + i] * U[kP * c + j];
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
    float U[8 * MP];
    scaled_u(s, sh, A, U);
    for (short k = 0; k < 8 * kP; ++k) ent[k] = U[k];
    threadgroup float * L = ent + 8 * kP;
    ent[SP_OK] = cholesky(U, L, L + kP * (kP + 1) / 2) ? 1.0f : 0.0f;
}

template <typename P>
static inline void solvep(P L, P D, thread const float (&b)[MP], thread float (&t)[MP]) {
    float y[MP];
    for (short i = 0; i < kP; ++i) {
        float v = b[i];
        for (short k = 0; k < i; ++k) v = v - L[LI(i, k)] * y[k];
        y[i] = v * D[i];
    }
    for (short i = kP - 1; i >= 0; --i) {
        float v = y[i];
        for (short k = i + 1; k < kP; ++k) v = v - L[LI(k, i)] * t[k];
        t[i] = v * D[i];
    }
}

static inline float cand_err(threadgroup const float * L, thread const float (&q)[MP], thread const float (&b)[MP], float wn, float sc) {
    float qb = 0.0f, qgq = 0.0f;
    for (short p = 0; p < kP; ++p) qb = qb + q[p] * b[p];
    for (short i = 0; i < kP; ++i) {
        float z = 0.0f;
        for (short k = i; k < kP; ++k) z = z + L[LI(k, i)] * q[k];
        qgq = qgq + z * z;
    }
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
}

static inline float decoded_err(thread const float (&x)[8], thread const float (&sh)[8], thread const float (&A)[64], int seed, int e,
                                thread const int (&q)[MP]) {
    ushort st = (ushort) seed;
    const float sc = kR32 * pow2f(e);
    float wv[8];
    for (short c = 0; c < 8; ++c) {
        int isum = 0;
        for (short p = 0; p < kP; ++p) { st = lfsr_step(st); isum += ((int) st - 32768) * q[p]; }
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

kernel void k_seed_searchp(constant SearchPArgs & a [[buffer(0)]], device const float * W [[buffer(1)]],
                           device const float * SH [[buffer(2)]], device ushort * seed_out [[buffer(3)]],
                           device uint * coef_out [[buffer(4)]], device uchar * ecode_out [[buffer(5)]],
                           device float * err_out [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float tab[SP_NCH * SP_ENT];
    const int g = a.g0 + (int) tg.x, ng = a.cols / 8, lo = a.bias, hi = a.bias + 15, nn = kP == 8 ? 6561 : 27;
    const short lofs = 8 * kP, dofs = 8 * kP + kP * (kP + 1) / 2;
    float sh[8], A[64];
    for (short c = 0; c < 8; ++c) sh[c] = kFull ? 1.0f : SH[g * 8 + c];
    for (short k = 0; k < 64; ++k) A[k] = kFull ? SH[g * 64 + k] : 0.0f;
    float x[SP_BPT][8], wn[SP_BPT], best[SP_BPT];
    int bs[SP_BPT], be[SP_BPT], row[SP_BPT];
    uint bw[SP_BPT];   // the winner's coefficients (as the C reference recomputes them from its table entry)
    for (short j = 0; j < SP_BPT; ++j) {
        row[j] = (int) tg.y * SP_ROWS + j * SP_TPB + tid;
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
    for (int s0 = 1; s0 <= a.n_seeds; s0 += SP_NCH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < SP_NCH && s0 + tid <= a.n_seeds) seed_entry(s0 + tid, sh, A, tab + tid * SP_ENT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nk = min(SP_NCH, a.n_seeds - s0 + 1);
        for (int k = 0; k < nk; ++k) {
            threadgroup const float * U = tab + k * SP_ENT;
            if (U[SP_OK] == 0.0f) continue;
            threadgroup const float * L = U + lofs, * D = U + dofs;
            for (short j = 0; j < SP_BPT; ++j) {
                float b[MP] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (short c = 0; c < 8; ++c)
                    for (short p = 0; p < kP; ++p) b[p] = b[p] + U[kP * c + p] * x[j][c];
                float t[MP];
                solvep(L, D, b, t);
                float m = 0.0f;
                for (short p = 0; p < kP; ++p) m = fmax(m, fabs(t[p]));
                int e0 = flog2f(m) - 2;
                e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                for (int ci = 0; ci < a.n_exp; ++ci) {
                    int e = e0 + a.exp_delta[ci];
                    e = e < lo ? lo : (e > hi ? hi : e);
                    const float inv = pow2f(-e), sc = pow2f(e);
                    float q[MP];
                    for (short p = 0; p < kP; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                    const float er = cand_err(L, q, b, wn[j], sc);
                    if (er < best[j]) {
                        best[j] = er; bs[j] = s0 + k; be[j] = e;
                        uint w = 0;
                        for (short p = 0; p < kP; ++p) w |= (uint) ((int) q[p] & 15) << (4 * p);
                        bw[j] = w;
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = 0; j < SP_BPT; ++j) {
        const int e = be[j];
        int q[MP], bq[MP];
        for (short p = 0; p < kP; ++p) bq[p] = q[p] = (int) (bw[j] << (28 - 4 * p)) >> 28;
        float bde = decoded_err(x[j], sh, A, bs[j], e, q);
        if (a.refit)
            for (int d = 0; d < nn; ++d) {
                int cq[MP], dd = d;
                bool okq = true;
                for (short p = 0; p < kP; ++p) { cq[p] = q[p] + dd % 3 - 1; dd /= 3; okq = okq && cq[p] >= -8 && cq[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err(x[j], sh, A, bs[j], e, cq);
                if (ee < bde) { bde = ee; for (short p = 0; p < kP; ++p) bq[p] = cq[p]; }
            }
        if (row[j] >= a.rows) continue;
        const ulong k = (ulong) row[j] * (ulong) ng + (ulong) g;
        uint cw = 0;
        for (short p = 0; p < kP; ++p) cw |= (uint) (bq[p] & 15) << (4 * p);
        seed_out[k] = (ushort) bs[j];
        coef_out[k] = cw;
        ecode_out[k] = (uchar) (e - a.bias);
        err_out[k] = bde;
    }
}
