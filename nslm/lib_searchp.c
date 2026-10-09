// nslm/lib_searchp.c - SeedLM blocks with P coefficients (see searchp.h): the decode spec, the seed table entry and the
// scalar reference search.  Built with -ffp-contract=off; nslm/searchp.metal repeats every expression in this order.
#include <math.h>
#include <string.h>

#include "searchp.h"

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

void nslmp_decode_block(int P, uint16_t seed, uint32_t coef, int e, uint16_t out[8]) {
    uint16_t st[8 * NSLMP_MAXP];
    lfsr_states(seed, 8 * P, st);
    const float sc = NSLM_R32 * pow2f(e);
    for (int c = 0; c < 8; ++c) {
        int32_t isum = 0;
        for (int p = 0; p < P; ++p) isum += ((int32_t) st[P * c + p] - 32768) * nslmp_q(coef, p);
        out[c] = nslm4_f2bf((float) isum * sc);
    }
}

// L: row-major lower triangle, row i at i (i + 1) / 2
#define LI(i, j) ((i) * ((i) + 1) / 2 + (j))

int nslmp_seed_entry(int P, int s, const float* sh, const float* A, float* ent) {
    memset(ent, 0, sizeof(float) * (size_t) NSLMP_ENT(P));
    uint16_t st[8 * NSLMP_MAXP];
    lfsr_states((uint16_t) s, 8 * P, st);
    float u0[8 * NSLMP_MAXP];
    for (int k = 0; k < 8 * P; ++k) u0[k] = (float) ((int32_t) st[k] - 32768) * NSLM_R32;
    float* U = ent;
    for (int c = 0; c < 8; ++c)
        for (int p = 0; p < P; ++p) {
            if (A) {
                float acc = 0.0f;
                for (int k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[P * k + p];
                U[P * c + p] = acc;
            } else U[P * c + p] = sh ? u0[P * c + p] * sh[c] : u0[P * c + p];
        }
    float* L = ent + 8 * P, *D = L + P * (P + 1) / 2;
    for (int i = 0; i < P; ++i)
        for (int j = 0; j <= i; ++j) {
            float g = 0.0f;
            for (int c = 0; c < 8; ++c) g = g + U[P * c + i] * U[P * c + j];
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
static inline void solvep(int P, const float* L, const float* D, const float* b, float* t) {
    float y[NSLMP_MAXP];
    for (int i = 0; i < P; ++i) {
        float v = b[i];
        for (int k = 0; k < i; ++k) v = v - L[LI(i, k)] * y[k];
        y[i] = v * D[i];
    }
    for (int i = P - 1; i >= 0; --i) {
        float v = y[i];
        for (int k = i + 1; k < P; ++k) v = v - L[LI(k, i)] * t[k];
        t[i] = v * D[i];
    }
}

static inline float cand_err(int P, const float* L, const float* q, const float* b, float wn, float sc) {
    float qb = 0.0f, qgq = 0.0f;
    for (int p = 0; p < P; ++p) qb = qb + q[p] * b[p];
    for (int i = 0; i < P; ++i) {   // z_i = (L^T q)_i = sum_{k >= i} L[k][i] q_k
        float z = 0.0f;
        for (int k = i; k < P; ++k) z = z + L[LI(k, i)] * q[k];
        qgq = qgq + z * z;
    }
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
}

static float decoded_err(int P, const float x[8], const float* sh, const float* A, int seed, int e, const int* q) {
    uint16_t bf[8];
    nslmp_decode_block(P, (uint16_t) seed, nslmp_pack(P, q), e, bf);
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

void nslmp_search_ref(int P, const float* tab, const uint8_t* ok, const float* w, int nb, const float* sh, const float* A, int bias,
                      const Search4Opts* o, uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err) {
    const int lo = bias, hi = bias + 15, ent = NSLMP_ENT(P), nn = P == 8 ? 6561 : P == 3 ? 27 : 0;
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
            const float* U = tab + (size_t) s * ent, *L = U + 8 * P, *D = L + P * (P + 1) / 2;
            float b[NSLMP_MAXP] = {0};
            for (int c = 0; c < 8; ++c)
                for (int p = 0; p < P; ++p) b[p] = b[p] + U[P * c + p] * x[c];
            float t[NSLMP_MAXP];
            solvep(P, L, D, b, t);
            float m = 0.0f;
            for (int p = 0; p < P; ++p) m = fmaxf(m, fabsf(t[p]));
            int e0 = flog2f(m) - 2;
            e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
            for (int ci = 0; ci < o->n_exp; ++ci) {
                int e = e0 + o->exp_delta[ci];
                e = e < lo ? lo : (e > hi ? hi : e);
                const float inv = pow2f(-e), sc = pow2f(e);
                float q[NSLMP_MAXP];
                for (int p = 0; p < P; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                const float er = cand_err(P, L, q, b, wn, sc);
                if (er < best) { best = er; bs = s; be = e; }
            }
        }
        const float* U = tab + (size_t) bs * ent, *L = U + 8 * P, *D = L + P * (P + 1) / 2;
        float b[NSLMP_MAXP] = {0};
        for (int c = 0; c < 8; ++c)
            for (int p = 0; p < P; ++p) b[p] = b[p] + U[P * c + p] * x[c];
        float t[NSLMP_MAXP];
        solvep(P, L, D, b, t);
        const float inv = pow2f(-be);
        int q[NSLMP_MAXP], bq[NSLMP_MAXP];
        for (int p = 0; p < P; ++p) bq[p] = q[p] = (int) clampq((t[p] * inv + MAGIC) - MAGIC);
        float bde = decoded_err(P, x, sh, A, bs, be, q);
        if (o->refit)
            for (int d = 0; d < nn; ++d) {
                int cq[NSLMP_MAXP], okq = 1, dd = d;
                for (int p = 0; p < P; ++p) { cq[p] = q[p] + dd % 3 - 1; dd /= 3; okq &= cq[p] >= -8 && cq[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err(P, x, sh, A, bs, be, cq);
                if (ee < bde) { bde = ee; for (int p = 0; p < P; ++p) bq[p] = cq[p]; }
            }
        seed[k] = (uint16_t) bs;
        coef[k] = nslmp_pack(P, bq);
        ecode[k] = (uint8_t) (be - bias);
        if (err) err[k] = bde;
    }
}
