// nslm/lib_search4.c - 4.5-bit SeedLM blocks (P = 4; see search4.h): the decode spec, the seed table entry and the
// scalar reference search.  Built with -ffp-contract=off; nslm/search4.metal repeats every expression in this order.
#include <math.h>
#include <string.h>

#include "search4.h"

#define MAGIC 12582912.0f   // 1.5 * 2^23: (x + MAGIC) - MAGIC rounds to nearest even for |x| < 2^22

static inline float pow2f(int e) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t) (e + 127) << 23;
    return v.f;
}
static inline int flog2f(float x) {   // floor(log2 x) for normal x > 0; -127 for 0 and subnormals
    union { uint32_t u; float f; } v;
    v.f = x;
    return (int) ((v.u >> 23) & 255u) - 127;
}
static inline float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }

void nslm4_decode_block(uint16_t seed, uint16_t coef, int e, uint16_t out[NSLM4_C]) {
    uint16_t st[NSLM4_C * NSLM4_P];
    lfsr_states(seed, NSLM4_C * NSLM4_P, st);
    const float sc = NSLM_R32 * pow2f(e);   // exact: a power-of-two scaling of R32
    for (int c = 0; c < NSLM4_C; ++c) {
        int32_t isum = 0;
        for (int p = 0; p < NSLM4_P; ++p) isum += ((int32_t) st[4 * c + p] - 32768) * nslm4_q(coef, p);
        out[c] = nslm4_f2bf((float) isum * sc);
    }
}

// U = (S R32) scaled by sh, or A (S R32) for a lower-triangular A (row-major 8 x 8): acc = acc + A[c][k] u[k][p],
// k = 0 .. c.
static int seed_entry(int s, const float* sh, const float* A, float* ent) {
    memset(ent, 0, sizeof(float) * NSLM4_ENT);
    uint16_t st[32];
    lfsr_states((uint16_t) s, 32, st);
    float* U = ent;
    float u0[32];
    for (int k = 0; k < 32; ++k) u0[k] = (float) ((int32_t) st[k] - 32768) * NSLM_R32;
    for (int c = 0; c < 8; ++c)
        for (int p = 0; p < 4; ++p) {
            if (A) {
                float acc = 0.0f;
                for (int k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[4 * k + p];
                U[4 * c + p] = acc;
            } else U[4 * c + p] = sh ? u0[4 * c + p] * sh[c] : u0[4 * c + p];
        }
    float G[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float g = 0.0f;
            for (int c = 0; c < 8; ++c) g = g + U[4 * c + i] * U[4 * c + j];
            G[i][j] = g;
        }
    // Cholesky G = L L^T (lower), f32
    float* L = ent + 32;
    const float a00 = G[0][0];
    if (!(a00 > 0.0f)) return 0;
    const float l00 = sqrtf(a00);
    const float l10 = G[1][0] / l00, l20 = G[2][0] / l00, l30 = G[3][0] / l00;
    const float a11 = G[1][1] - l10 * l10;
    if (!(a11 > 0.0f)) return 0;
    const float l11 = sqrtf(a11);
    const float l21 = (G[2][1] - l20 * l10) / l11, l31 = (G[3][1] - l30 * l10) / l11;
    const float a22 = (G[2][2] - l20 * l20) - l21 * l21;
    if (!(a22 > 0.0f)) return 0;
    const float l22 = sqrtf(a22);
    const float l32 = ((G[3][2] - l30 * l20) - l31 * l21) / l22;
    const float a33 = ((G[3][3] - l30 * l30) - l31 * l31) - l32 * l32;
    if (!(a33 > 0.0f)) return 0;
    const float l33 = sqrtf(a33);
    L[0] = l00; L[1] = l10; L[2] = l11; L[3] = l20; L[4] = l21; L[5] = l22; L[6] = l30; L[7] = l31; L[8] = l32; L[9] = l33;
    float* D = ent + 42;
    D[0] = 1.0f / l00; D[1] = 1.0f / l11; D[2] = 1.0f / l22; D[3] = 1.0f / l33;
    return 1;
}

int nslm4_seed_entry(int s, const float* sh, float* ent) { return seed_entry(s, sh, NULL, ent); }
int nslm4_seed_entry_a(int s, const float A[64], float* ent) { return seed_entry(s, NULL, A, ent); }

// t = G^-1 b through L: L y = b, L^T t = y (reciprocal diagonals).
static inline void solve4(const float* L, const float* D, const float b[4], float t[4]) {
    const float y0 = b[0] * D[0];
    const float y1 = (b[1] - L[1] * y0) * D[1];
    const float y2 = ((b[2] - L[3] * y0) - L[4] * y1) * D[2];
    const float y3 = (((b[3] - L[6] * y0) - L[7] * y1) - L[8] * y2) * D[3];
    t[3] = y3 * D[3];
    t[2] = (y2 - L[8] * t[3]) * D[2];
    t[1] = ((y1 - L[4] * t[2]) - L[7] * t[3]) * D[1];
    t[0] = (((y0 - L[1] * t[1]) - L[3] * t[2]) - L[6] * t[3]) * D[0];
}

// The search error of q at scale sc: (|x|^2 - 2 sc q.b) + sc^2 |L^T q|^2; INFINITY when the reconstruction term
// exceeds 4 |x|^2 (it cannot beat q = 0).
static inline float cand_err(const float* L, const float q[4], const float b[4], float wn, float sc) {
    const float qb = ((q[0] * b[0] + q[1] * b[1]) + q[2] * b[2]) + q[3] * b[3];
    const float z0 = ((L[0] * q[0] + L[1] * q[1]) + L[3] * q[2]) + L[6] * q[3];
    const float z1 = (L[2] * q[1] + L[4] * q[2]) + L[7] * q[3];
    const float z2 = L[5] * q[2] + L[8] * q[3];
    const float z3 = L[9] * q[3];
    const float qgq = ((z0 * z0 + z1 * z1) + z2 * z2) + z3 * z3;
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
}

// Decoded weighted error of (seed, e, q) against x = w * sh (or A w).
static float decoded_err4(const float x[8], const float* sh, const float* A, int seed, int e, const int q[4]) {
    uint16_t bf[8];
    nslm4_decode_block((uint16_t) seed, nslm4_pack(q), e, bf);
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

static void search_ref(const float* tab, const uint8_t* ok, const float* w, int nb, const float* sh, const float* A,
                       int bias, const Search4Opts* o, uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err) {
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
            const float* U = tab + (size_t) s * NSLM4_ENT, *L = U + 32, *D = U + 42;
            float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int c = 0; c < 8; ++c)
                for (int p = 0; p < 4; ++p) b[p] = b[p] + U[4 * c + p] * x[c];
            float t[4];
            solve4(L, D, b, t);
            const float m = fmaxf(fmaxf(fabsf(t[0]), fabsf(t[1])), fmaxf(fabsf(t[2]), fabsf(t[3])));
            int e0 = flog2f(m) - 2;
            e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
            for (int ci = 0; ci < o->n_exp; ++ci) {
                int e = e0 + o->exp_delta[ci];
                e = e < lo ? lo : (e > hi ? hi : e);
                const float inv = pow2f(-e), sc = pow2f(e);
                float q[4];
                for (int p = 0; p < 4; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                const float er = cand_err(L, q, b, wn, sc);
                if (er < best) { best = er; bs = s; be = e; }
            }
        }
        // refit: the winner's q among its 3^4 neighbours by decoded error
        const float* U = tab + (size_t) bs * NSLM4_ENT, *L = U + 32, *D = U + 42;
        float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = 0; c < 8; ++c)
            for (int p = 0; p < 4; ++p) b[p] = b[p] + U[4 * c + p] * x[c];
        float t[4];
        solve4(L, D, b, t);
        const float inv = pow2f(-be);
        int q[4], bq[4];
        for (int p = 0; p < 4; ++p) bq[p] = q[p] = (int) clampq((t[p] * inv + MAGIC) - MAGIC);
        float bde = decoded_err4(x, sh, A, bs, be, q);
        if (o->refit)
            for (int d = 0; d < 81; ++d) {
                int c4[4], okq = 1, dd = d;
                for (int p = 0; p < 4; ++p) { c4[p] = q[p] + dd % 3 - 1; dd /= 3; okq &= c4[p] >= -8 && c4[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err4(x, sh, A, bs, be, c4);
                if (ee < bde) { bde = ee; for (int p = 0; p < 4; ++p) bq[p] = c4[p]; }
            }
        seed[k] = (uint16_t) bs;
        coef[k] = nslm4_pack(bq);
        ecode[k] = (uint8_t) (be - bias);
        if (err) err[k] = bde;
    }
}

void nslm4_search_ref(const float* tab, const uint8_t* ok, const float* w, int nb, const float* sh, int bias,
                      const Search4Opts* o, uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err) {
    search_ref(tab, ok, w, nb, sh, NULL, bias, o, seed, coef, ecode, err);
}

void nslm4_search_ref_a(const float* tab, const uint8_t* ok, const float* w, int nb, const float A[64], int bias,
                        const Search4Opts* o, uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err) {
    search_ref(tab, ok, w, nb, NULL, A, bias, o, seed, coef, ecode, err);
}
