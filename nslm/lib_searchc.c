// nslm/lib_searchc.c - see searchc.h.  The expressions of nslm/lib_searchp.c in its order, for C weights a block.
#include <math.h>
#include <string.h>

#include "lfsr.h"
#include "searchc.h"

#define MAGIC 12582912.0f   // 1.5 * 2^23: (x + MAGIC) - MAGIC rounds to nearest even for |x| < 2^22
#define LI(i, j) ((i) * ((i) + 1) / 2 + (j))

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
static inline int qp(uint32_t coef, int p) { return (int32_t) (coef << (28 - 4 * p)) >> 28; }

void nslmc_decode_block(int C, int P, uint16_t seed, uint32_t coef, int e, uint16_t* out) {
    uint16_t st[NSLMC_MAXC * NSLMC_MAXP];
    lfsr_states(seed, C * P, st);
    const float sc = NSLM_R32 * pow2f(e);
    for (int c = 0; c < C; ++c) {
        int32_t isum = 0;
        for (int p = 0; p < P; ++p) isum += ((int32_t) st[P * c + p] - 32768) * qp(coef, p);
        out[c] = nslm4_f2bf((float) isum * sc);
    }
}

// U = S R32 sh, and the Cholesky factor of U^T U (L, reciprocal diagonal D); 0 if it fails
static int entry(int C, int P, int s, const float* sh, float* U, float* L, float* D) {
    uint16_t st[NSLMC_MAXC * NSLMC_MAXP];
    lfsr_states((uint16_t) s, C * P, st);
    for (int c = 0; c < C; ++c)
        for (int p = 0; p < P; ++p) U[P * c + p] = ((float) ((int32_t) st[P * c + p] - 32768) * NSLM_R32) * sh[c];
    for (int i = 0; i < P; ++i)
        for (int j = 0; j <= i; ++j) {
            float g = 0.0f;
            for (int c = 0; c < C; ++c) g = g + U[P * c + i] * U[P * c + j];
            for (int k = 0; k < j; ++k) g = g - L[LI(i, k)] * L[LI(j, k)];
            if (i == j) {
                if (!(g > 0.0f)) return 0;
                L[LI(i, i)] = sqrtf(g);
                D[i] = 1.0f / L[LI(i, i)];
            } else L[LI(i, j)] = g * D[j];
        }
    return 1;
}

static void fit(int C, int P, const float* U, const float* L, const float* D, const float* x, float* b, float* t) {
    for (int p = 0; p < P; ++p) b[p] = 0.0f;
    for (int c = 0; c < C; ++c)
        for (int p = 0; p < P; ++p) b[p] = b[p] + U[P * c + p] * x[c];
    float y[NSLMC_MAXP];   // t = (U^T U)^-1 b: L y = b, L^T t = y
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

static float decoded_err(int C, int P, const float* x, const float* sh, int seed, int e, const int* q) {
    uint16_t bf[NSLMC_MAXC];
    uint32_t coef = 0;
    for (int p = 0; p < P; ++p) coef |= (uint32_t) (q[p] & 15) << (4 * p);
    nslmc_decode_block(C, P, (uint16_t) seed, coef, e, bf);
    float er = 0.0f;
    for (int c = 0; c < C; ++c) {
        const float d = x[c] - sh[c] * nslm4_bf2f(bf[c]);
        er = er + d * d;
    }
    return er;
}

void nslmc_search_ref(int C, int P, const float* w, const float* sh, int nb, int bias, const Search4Opts* o, uint16_t* seed,
                      uint32_t* coef, uint8_t* ecode, float* err) {
    const int lo = bias, hi = bias + 15;
    int nn = 1;
    for (int p = 0; p < P; ++p) nn *= 3;
    float U[NSLMC_MAXC * NSLMC_MAXP], L[NSLMC_MAXP * (NSLMC_MAXP + 1) / 2], D[NSLMC_MAXP];
    for (int k = 0; k < nb; ++k) {
        const float* s_ = sh + (size_t) k * C;
        float x[NSLMC_MAXC], wn = 0.0f;
        for (int c = 0; c < C; ++c) x[c] = w[(size_t) k * C + c] * s_[c];
        for (int c = 0; c < C; ++c) wn = wn + x[c] * x[c];
        float best = INFINITY;
        int bs = 1, be = lo;
        for (int s = 1; s <= o->n_seeds; ++s) {
            if (!entry(C, P, s, s_, U, L, D)) continue;
            float b[NSLMC_MAXP], t[NSLMC_MAXP];
            fit(C, P, U, L, D, x, b, t);
            float m = 0.0f;
            for (int p = 0; p < P; ++p) m = fmaxf(m, fabsf(t[p]));
            int e0 = flog2f(m) - 2;
            e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
            for (int ci = 0; ci < o->n_exp; ++ci) {
                int e = e0 + o->exp_delta[ci];
                e = e < lo ? lo : (e > hi ? hi : e);
                const float inv = pow2f(-e), sc = pow2f(e);
                float q[NSLMC_MAXP], qb = 0.0f, qgq = 0.0f;
                for (int p = 0; p < P; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                for (int p = 0; p < P; ++p) qb = qb + q[p] * b[p];
                for (int i = 0; i < P; ++i) {   // z_i = (L^T q)_i
                    float z = 0.0f;
                    for (int j = i; j < P; ++j) z = z + L[LI(j, i)] * q[j];
                    qgq = qgq + z * z;
                }
                const float rec = (sc * sc) * qgq, er = rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
                if (er < best) { best = er; bs = s; be = e; }
            }
        }
        entry(C, P, bs, s_, U, L, D);
        float b[NSLMC_MAXP], t[NSLMC_MAXP];
        fit(C, P, U, L, D, x, b, t);
        const float inv = pow2f(-be);
        int q[NSLMC_MAXP], bq[NSLMC_MAXP];
        for (int p = 0; p < P; ++p) bq[p] = q[p] = (int) clampq((t[p] * inv + MAGIC) - MAGIC);
        float bde = decoded_err(C, P, x, s_, bs, be, q);
        if (o->refit)
            for (int d = 0; d < nn; ++d) {   // the 3^P neighbours of q by the decoded error
                int cq[NSLMC_MAXP], okq = 1, dd = d;
                for (int p = 0; p < P; ++p) { cq[p] = q[p] + dd % 3 - 1; dd /= 3; okq &= cq[p] >= -8 && cq[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err(C, P, x, s_, bs, be, cq);
                if (ee < bde) { bde = ee; memcpy(bq, cq, sizeof(int) * (size_t) P); }
            }
        uint32_t cw = 0;
        for (int p = 0; p < P; ++p) cw |= (uint32_t) (bq[p] & 15) << (4 * p);
        seed[k] = (uint16_t) bs;
        coef[k] = cw;
        ecode[k] = (uint8_t) (be - bias);
        if (err) err[k] = bde;
    }
}
