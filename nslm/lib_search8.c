// nslm/lib_search8.c - 6.5-bit SeedLM blocks (P = 8; see search8.h): the decode spec, the seed table entry and the
// scalar reference search.  Built with -ffp-contract=off; nslm/search8.metal repeats every expression in this order.
#include <math.h>
#include <string.h>

#include "search8.h"

#define MAGIC 12582912.0f   // 1.5 * 2^23: (x + MAGIC) - MAGIC rounds to nearest even for |x| < 2^22

static inline float pow2f(int e) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t) (e + 127) << 23;
    return v.f;
}
static inline int flog2f(float x) {
    union { uint32_t u; float f; } v;
    v.f = x;
    return (int) ((v.u >> 23) & 255u) - 127;
}
static inline float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }

void nslm8_decode_block(uint16_t seed, uint32_t coef, int e, uint16_t out[NSLM8_C]) {
    uint16_t st[NSLM8_C * NSLM8_P];
    lfsr_states(seed, NSLM8_C * NSLM8_P, st);
    const float sc = NSLM_R32 * pow2f(e);
    for (int c = 0; c < NSLM8_C; ++c) {
        int32_t isum = 0;
        for (int p = 0; p < NSLM8_P; ++p) isum += ((int32_t) st[8 * c + p] - 32768) * nslm8_q(coef, p);
        out[c] = nslm4_f2bf((float) isum * sc);
    }
}

// L: row-major lower triangle, row i at i (i + 1) / 2
#define LI(i, j) ((i) * ((i) + 1) / 2 + (j))

int nslm8_seed_entry(int s, const float* sh, const float* A, float* ent) {
    memset(ent, 0, sizeof(float) * NSLM8_ENT);
    uint16_t st[64];
    lfsr_states((uint16_t) s, 64, st);
    float u0[64];
    for (int k = 0; k < 64; ++k) u0[k] = (float) ((int32_t) st[k] - 32768) * NSLM_R32;
    float* U = ent;
    for (int c = 0; c < 8; ++c)
        for (int p = 0; p < 8; ++p) {
            if (A) {
                float acc = 0.0f;
                for (int k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[8 * k + p];
                U[8 * c + p] = acc;
            } else U[8 * c + p] = sh ? u0[8 * c + p] * sh[c] : u0[8 * c + p];
        }
    float* L = ent + 64, *D = ent + 100;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j <= i; ++j) {
            float g = 0.0f;
            for (int c = 0; c < 8; ++c) g = g + U[8 * c + i] * U[8 * c + j];
            for (int k = 0; k < j; ++k) g = g - L[LI(i, k)] * L[LI(j, k)];
            if (i == j) {
                if (!(g > 0.0f)) return 0;
                L[LI(i, i)] = sqrtf(g);
                D[i] = 1.0f / L[LI(i, i)];
            } else L[LI(i, j)] = g * D[j];
        }
    return 1;
}

// t = G^-1 b through L: L y = b, L^T t = y (reciprocal diagonals).
static inline void solve8(const float* L, const float* D, const float b[8], float t[8]) {
    float y[8];
    for (int i = 0; i < 8; ++i) {
        float v = b[i];
        for (int k = 0; k < i; ++k) v = v - L[LI(i, k)] * y[k];
        y[i] = v * D[i];
    }
    for (int i = 7; i >= 0; --i) {
        float v = y[i];
        for (int k = i + 1; k < 8; ++k) v = v - L[LI(k, i)] * t[k];
        t[i] = v * D[i];
    }
}

static inline float cand_err(const float* L, const float q[8], const float b[8], float wn, float sc) {
    float qb = 0.0f, qgq = 0.0f;
    for (int p = 0; p < 8; ++p) qb = qb + q[p] * b[p];
    for (int i = 0; i < 8; ++i) {   // z_i = (L^T q)_i = sum_{k >= i} L[k][i] q_k
        float z = 0.0f;
        for (int k = i; k < 8; ++k) z = z + L[LI(k, i)] * q[k];
        qgq = qgq + z * z;
    }
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
}

static float decoded_err8(const float x[8], const float* sh, const float* A, int seed, int e, const int q[8]) {
    uint16_t bf[8];
    nslm8_decode_block((uint16_t) seed, nslm8_pack(q), e, bf);
    float er = 0.0f;
    for (int c = 0; c < 8; ++c) {
        float v;
        if (A) {
            v = 0.0f;
            for (int k = 0; k <= c; ++k) v = v + A[c * 8 + k] * nslm4_bf2f(bf[k]);
        } else v = sh ? sh[c] * nslm4_bf2f(bf[c]) : nslm4_bf2f(bf[c]);
        const float d = x[c] - v;
        er = er + d * d;
    }
    return er;
}

void nslm8_search_ref(const float* tab, const uint8_t* ok, const float* w, int nb, const float* sh, const float* A, int bias,
                      const Search4Opts* o, uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err) {
    const int lo = bias, hi = bias + 15;
    for (int k = 0; k < nb; ++k) {
        float x[8], wn = 0.0f;
        for (int c = 0; c < 8; ++c) {
            if (A) {
                float acc = 0.0f;
                for (int j = 0; j <= c; ++j) acc = acc + A[c * 8 + j] * w[k * 8 + j];
                x[c] = acc;
            } else x[c] = sh ? w[k * 8 + c] * sh[c] : w[k * 8 + c];
        }
        for (int c = 0; c < 8; ++c) wn = wn + x[c] * x[c];
        float best = INFINITY;
        int bs = 1, be = lo;
        for (int s = 1; s <= o->n_seeds; ++s) {
            if (!ok[s]) continue;
            const float* U = tab + (size_t) s * NSLM8_ENT, *L = U + 64, *D = U + 100;
            float b[8] = {0};
            for (int c = 0; c < 8; ++c)
                for (int p = 0; p < 8; ++p) b[p] = b[p] + U[8 * c + p] * x[c];
            float t[8];
            solve8(L, D, b, t);
            float m = 0.0f;
            for (int p = 0; p < 8; ++p) m = fmaxf(m, fabsf(t[p]));
            int e0 = flog2f(m) - 2;
            e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
            for (int ci = 0; ci < o->n_exp; ++ci) {
                int e = e0 + o->exp_delta[ci];
                e = e < lo ? lo : (e > hi ? hi : e);
                const float inv = pow2f(-e), sc = pow2f(e);
                float q[8];
                for (int p = 0; p < 8; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                const float er = cand_err(L, q, b, wn, sc);
                if (er < best) { best = er; bs = s; be = e; }
            }
        }
        const float* U = tab + (size_t) bs * NSLM8_ENT, *L = U + 64, *D = U + 100;
        float b[8] = {0};
        for (int c = 0; c < 8; ++c)
            for (int p = 0; p < 8; ++p) b[p] = b[p] + U[8 * c + p] * x[c];
        float t[8];
        solve8(L, D, b, t);
        const float inv = pow2f(-be);
        int q[8], bq[8];
        for (int p = 0; p < 8; ++p) bq[p] = q[p] = (int) clampq((t[p] * inv + MAGIC) - MAGIC);
        float bde = decoded_err8(x, sh, A, bs, be, q);
        if (o->refit)
            for (int d = 0; d < 6561; ++d) {
                int c8[8], okq = 1, dd = d;
                for (int p = 0; p < 8; ++p) { c8[p] = q[p] + dd % 3 - 1; dd /= 3; okq &= c8[p] >= -8 && c8[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err8(x, sh, A, bs, be, c8);
                if (ee < bde) { bde = ee; for (int p = 0; p < 8; ++p) bq[p] = c8[p]; }
            }
        seed[k] = (uint16_t) bs;
        coef[k] = nslm8_pack(bq);
        ecode[k] = (uint8_t) (be - bias);
        if (err) err[k] = bde;
    }
}
