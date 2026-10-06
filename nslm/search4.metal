// nslm/search4.metal - the P = 4 seed search on the GPU (nslm/search4.h).  Every expression is nslm/lib_search4.c's,
// in the same order, without fast math or contraction: bit-identical results.
// Grid (column groups from a.g0, row tiles of S4_ROWS), S4_TPB threads, S4_BPT rows per thread; the seed table is
// built S4_NCH seeds at a time in threadgroup memory.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)   // every f32 expression as nslm/lib_search4.c writes it

#include "search4_gpu.h"

#define MAGIC 12582912.0f
constant float kR32 = 1.0f / 32767.0f;

static inline float pow2f(int e) { return as_type<float>((uint) (e + 127) << 23); }
static inline int flog2f(float x) { return (int) ((as_type<uint>(x) >> 23) & 255u) - 127; }
static inline float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }
static inline ushort lfsr_step(ushort s) {
    const ushort b = (ushort) ((s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1u);
    return (ushort) ((s >> 1) | (b << 15));
}
static inline float f2bf_f(float f) {   // round to BF16 (nearest even), as f32
    uint u = as_type<uint>(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xFFFF0000u);
}

// nslm4_seed_entry: U[32], L[10], 1/diag[4], ok (ent[46])
static inline void seed_entry(int s, thread const float (&sh)[8], threadgroup float * ent) {
    ushort st = (ushort) s;
    float U[32];
    for (short k = 0; k < 32; ++k) {
        st = lfsr_step(st);
        const float u = (float) ((int) st - 32768) * kR32;
        U[k] = u * sh[k / 4];
    }
    float G[4][4];
    for (short i = 0; i < 4; ++i)
        for (short j = 0; j < 4; ++j) {
            float g = 0.0f;
            for (short c = 0; c < 8; ++c) g = g + U[4 * c + i] * U[4 * c + j];
            G[i][j] = g;
        }
    for (short k = 0; k < 32; ++k) ent[k] = U[k];
    ent[46] = 0.0f;
    const float a00 = G[0][0];
    if (!(a00 > 0.0f)) return;
    const float l00 = precise::sqrt(a00);
    const float l10 = G[1][0] / l00, l20 = G[2][0] / l00, l30 = G[3][0] / l00;
    const float a11 = G[1][1] - l10 * l10;
    if (!(a11 > 0.0f)) return;
    const float l11 = precise::sqrt(a11);
    const float l21 = (G[2][1] - l20 * l10) / l11, l31 = (G[3][1] - l30 * l10) / l11;
    const float a22 = (G[2][2] - l20 * l20) - l21 * l21;
    if (!(a22 > 0.0f)) return;
    const float l22 = precise::sqrt(a22);
    const float l32 = ((G[3][2] - l30 * l20) - l31 * l21) / l22;
    const float a33 = ((G[3][3] - l30 * l30) - l31 * l31) - l32 * l32;
    if (!(a33 > 0.0f)) return;
    const float l33 = precise::sqrt(a33);
    ent[32] = l00; ent[33] = l10; ent[34] = l11; ent[35] = l20; ent[36] = l21; ent[37] = l22; ent[38] = l30; ent[39] = l31;
    ent[40] = l32; ent[41] = l33;
    ent[42] = 1.0f / l00; ent[43] = 1.0f / l11; ent[44] = 1.0f / l22; ent[45] = 1.0f / l33;
    ent[46] = 1.0f;
}

template <typename P>
static inline void solve4(P L, P D, thread const float (&b)[4], thread float (&t)[4]) {
    const float y0 = b[0] * D[0];
    const float y1 = (b[1] - L[1] * y0) * D[1];
    const float y2 = ((b[2] - L[3] * y0) - L[4] * y1) * D[2];
    const float y3 = (((b[3] - L[6] * y0) - L[7] * y1) - L[8] * y2) * D[3];
    t[3] = y3 * D[3];
    t[2] = (y2 - L[8] * t[3]) * D[2];
    t[1] = ((y1 - L[4] * t[2]) - L[7] * t[3]) * D[1];
    t[0] = (((y0 - L[1] * t[1]) - L[3] * t[2]) - L[6] * t[3]) * D[0];
}

static inline float cand_err(threadgroup const float * L, thread const float (&q)[4], thread const float (&b)[4], float wn, float sc) {
    const float qb = ((q[0] * b[0] + q[1] * b[1]) + q[2] * b[2]) + q[3] * b[3];
    const float z0 = ((L[0] * q[0] + L[1] * q[1]) + L[3] * q[2]) + L[6] * q[3];
    const float z1 = (L[2] * q[1] + L[4] * q[2]) + L[7] * q[3];
    const float z2 = L[5] * q[2] + L[8] * q[3];
    const float z3 = L[9] * q[3];
    const float qgq = ((z0 * z0 + z1 * z1) + z2 * z2) + z3 * z3;
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
}

// nslm4_decode_block + the weighted error, for (seed, e, q)
static inline float decoded_err4(thread const float (&x)[8], thread const float (&sh)[8], int seed, int e, thread const int (&q)[4]) {
    ushort st = (ushort) seed;
    const float sc = kR32 * pow2f(e);
    float er = 0.0f;
    float wv[8];
    for (short c = 0; c < 8; ++c) {
        int isum = 0;
        for (short p = 0; p < 4; ++p) { st = lfsr_step(st); isum += ((int) st - 32768) * q[p]; }
        wv[c] = f2bf_f((float) isum * sc);
    }
    for (short c = 0; c < 8; ++c) {
        const float d = x[c] - sh[c] * wv[c];
        er = er + d * d;
    }
    return er;
}

kernel void k_seed_search4(constant Search4Args & a [[buffer(0)]], device const float * W [[buffer(1)]],
                           device const float * SH [[buffer(2)]], device ushort * seed_out [[buffer(3)]],
                           device ushort * coef_out [[buffer(4)]], device uchar * ecode_out [[buffer(5)]],
                           device float * err_out [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float tab[S4_NCH * S4_ENT];
    const int g = a.g0 + (int) tg.x, ng = a.cols / 8, lo = a.bias, hi = a.bias + 15;
    float sh[8];
    for (short c = 0; c < 8; ++c) sh[c] = SH[g * 8 + c];
    float x[S4_BPT][8], wn[S4_BPT], best[S4_BPT];
    int bs[S4_BPT], be[S4_BPT], row[S4_BPT];
    for (short j = 0; j < S4_BPT; ++j) {
        row[j] = (int) tg.y * S4_ROWS + j * S4_TPB + tid;
        const int r = min(row[j], a.rows - 1);
        for (short c = 0; c < 8; ++c) x[j][c] = W[(ulong) r * a.cols + g * 8 + c] * sh[c];
        float n = 0.0f;
        for (short c = 0; c < 8; ++c) n = n + x[j][c] * x[j][c];
        wn[j] = n; best[j] = INFINITY; bs[j] = 1; be[j] = lo;
    }
    for (int s0 = 1; s0 <= a.n_seeds; s0 += S4_NCH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < S4_NCH && s0 + tid <= a.n_seeds) seed_entry(s0 + tid, sh, tab + tid * S4_ENT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nk = min(S4_NCH, a.n_seeds - s0 + 1);
        for (int k = 0; k < nk; ++k) {
            threadgroup const float * U = tab + k * S4_ENT;
            if (U[46] == 0.0f) continue;
            threadgroup const float * L = U + 32, * D = U + 42;
            for (short j = 0; j < S4_BPT; ++j) {
                float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (short c = 0; c < 8; ++c)
                    for (short p = 0; p < 4; ++p) b[p] = b[p] + U[4 * c + p] * x[j][c];
                float t[4];
                solve4(L, D, b, t);
                const float m = fmax(fmax(fabs(t[0]), fabs(t[1])), fmax(fabs(t[2]), fabs(t[3])));
                int e0 = flog2f(m) - 2;
                e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                for (int ci = 0; ci < a.n_exp; ++ci) {
                    int e = e0 + a.exp_delta[ci];
                    e = e < lo ? lo : (e > hi ? hi : e);
                    const float inv = pow2f(-e), sc = pow2f(e);
                    float q[4];
                    for (short p = 0; p < 4; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                    const float er = cand_err(L, q, b, wn[j], sc);
                    if (er < best[j]) { best[j] = er; bs[j] = s0 + k; be[j] = e; }
                }
            }
        }
    }
    // refit: rebuild the winner's entry in registers
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = 0; j < S4_BPT; ++j) {
        ushort st = (ushort) bs[j];
        float U[32];
        for (short k = 0; k < 32; ++k) { st = lfsr_step(st); U[k] = ((float) ((int) st - 32768) * kR32) * sh[k / 4]; }
        float G[4][4];
        for (short i = 0; i < 4; ++i)
            for (short jj = 0; jj < 4; ++jj) {
                float gg = 0.0f;
                for (short c = 0; c < 8; ++c) gg = gg + U[4 * c + i] * U[4 * c + jj];
                G[i][jj] = gg;
            }
        const float l00 = precise::sqrt(G[0][0]);
        const float l10 = G[1][0] / l00, l20 = G[2][0] / l00, l30 = G[3][0] / l00;
        const float l11 = precise::sqrt(G[1][1] - l10 * l10);
        const float l21 = (G[2][1] - l20 * l10) / l11, l31 = (G[3][1] - l30 * l10) / l11;
        const float l22 = precise::sqrt((G[2][2] - l20 * l20) - l21 * l21);
        const float l32 = ((G[3][2] - l30 * l20) - l31 * l21) / l22;
        const float l33 = precise::sqrt(((G[3][3] - l30 * l30) - l31 * l31) - l32 * l32);
        const float L[10] = {l00, l10, l11, l20, l21, l22, l30, l31, l32, l33};
        const float D[4] = {1.0f / l00, 1.0f / l11, 1.0f / l22, 1.0f / l33};
        float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (short c = 0; c < 8; ++c)
            for (short p = 0; p < 4; ++p) b[p] = b[p] + U[4 * c + p] * x[j][c];
        float t[4];
        solve4((thread const float *) L, (thread const float *) D, b, t);
        const int e = be[j];
        const float inv = pow2f(-e);
        int q[4], bq[4];
        for (short p = 0; p < 4; ++p) bq[p] = q[p] = (int) clampq((t[p] * inv + MAGIC) - MAGIC);
        float bde = decoded_err4(x[j], sh, bs[j], e, q);
        if (a.refit)
            for (int d = 0; d < 81; ++d) {
                int c4[4], dd = d;
                bool okq = true;
                for (short p = 0; p < 4; ++p) { c4[p] = q[p] + dd % 3 - 1; dd /= 3; okq = okq && c4[p] >= -8 && c4[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err4(x[j], sh, bs[j], e, c4);
                if (ee < bde) { bde = ee; for (short p = 0; p < 4; ++p) bq[p] = c4[p]; }
            }
        if (row[j] >= a.rows) continue;
        const ulong k = (ulong) row[j] * (ulong) ng + (ulong) g;
        seed_out[k] = (ushort) bs[j];
        coef_out[k] = (ushort) ((bq[0] & 15) | ((bq[1] & 15) << 4) | ((bq[2] & 15) << 8) | ((bq[3] & 15) << 12));
        ecode_out[k] = (uchar) (e - a.bias);
        err_out[k] = bde;
    }
}
