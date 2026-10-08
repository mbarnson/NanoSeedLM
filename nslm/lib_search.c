// nslm/lib_search.c - the P = 3 seed search (search.h): scalar reference and Clang-vector version.  Both evaluate the
// identical expressions in the identical order (-ffp-contract=off), so they agree bit for bit.
#include <math.h>
#include <string.h>

#include "search.h"

#define MAGIC 12582912.0f   // 1.5 * 2^23: (x + MAGIC) - MAGIC rounds to nearest even for |x| < 2^22

#ifdef __clang__
typedef float f4 __attribute__((ext_vector_type(4)));
typedef int32_t i4 __attribute__((ext_vector_type(4)));
#endif

static int seedtab_one(SeedTab* t, int s, const float* sh) {
    memset(t, 0, sizeof *t);
    uint16_t st[NSLM_CP];
    lfsr_states((uint16_t) s, NSLM_CP, st);
    for (int k = 0; k < NSLM_CP; ++k) {
        t->U[k] = (float) ((int32_t) st[k] - 32768) * NSLM_R32;
        if (sh) t->U[k] = t->U[k] * sh[k / NSLM_P];
    }
    int bad = 0;
    {
        double g[3][3] = {{0}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int c = 0; c < NSLM_C; ++c) g[i][j] += (double) t->U[c * 3 + i] * (double) t->U[c * 3 + j];
        const double det = g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1]) - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0]) +
                           g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]);
        {   // Cholesky G = R^T R in double
            const double r00 = sqrt(g[0][0]), r01 = g[0][1] / r00, r02 = g[0][2] / r00;
            const double r11 = sqrt(fmax(g[1][1] - r01 * r01, 0)), r12 = r11 > 0 ? (g[1][2] - r01 * r02) / r11 : 0;
            const double r22 = sqrt(fmax(g[2][2] - r02 * r02 - r12 * r12, 0));
            t->R[0] = (float) r00; t->R[1] = (float) r01; t->R[2] = (float) r02;
            t->R[3] = (float) r11; t->R[4] = (float) r12; t->R[5] = (float) r22;
        }
        // Near-singular seeds stay candidates (q is clamped, the Gram error is exact for any q); only an exactly
        // singular U (none for K = 16) is excluded.
        if (!(det > 0)) return 1;   // Gi stays 0: t = 0, q = 0, never wins over a real fit
        t->Gi[0] = (float) ((g[1][1] * g[2][2] - g[1][2] * g[2][1]) / det);
        t->Gi[1] = (float) ((g[0][2] * g[2][1] - g[0][1] * g[2][2]) / det);
        t->Gi[2] = (float) ((g[0][1] * g[1][2] - g[0][2] * g[1][1]) / det);
        t->Gi[3] = (float) ((g[0][0] * g[2][2] - g[0][2] * g[2][0]) / det);
        t->Gi[4] = (float) ((g[0][2] * g[1][0] - g[0][0] * g[1][2]) / det);
        t->Gi[5] = (float) ((g[0][0] * g[1][1] - g[0][1] * g[1][0]) / det);
    }
    return bad;
}

int nslm_seedtab_build(SeedTab* tab) {
    int bad = 0;
    memset(&tab[0], 0, sizeof(SeedTab));
    for (int s = 1; s <= 65535; ++s) bad += seedtab_one(&tab[s], s, NULL);
    return bad;
}

void nslm_seedtab_build_weighted(SeedTab* tab, const float sh[8], int s0, int s1) {
    if (s0 <= 0) { memset(&tab[0], 0, sizeof(SeedTab)); s0 = 1; }
    for (int s = s0; s < s1 && s <= 65535; ++s) seedtab_one(&tab[s], s, sh);
}

static inline float pow2f(int e) {
    uint32_t u = (uint32_t) (e + 127) << 23;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline int flog2f(float x) {   // floor(log2 x) for normal x > 0; -127 for 0 and subnormals
    uint32_t u;
    memcpy(&u, &x, 4);
    return (int) ((u >> 23) & 255u) - 127;
}
static inline float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }

// Squared error of the decoded BF16 block against w (weighted: w pre-scaled, sum_c (w_c - sh_c * w_hat_c)^2).
static float decoded_err(const float* w, uint16_t seed, int ecode, const int q[3], int bias, const float* sh) {
    uint16_t bf[NSLM_C];
    nslm_decode_block(seed, nslm_pack(ecode, q), bias, bf);
    float e = 0;
    for (int c = 0; c < NSLM_C; ++c) {
        const float d = w[c] - (sh ? sh[c] * nslm_bf2f(bf[c]) : nslm_bf2f(bf[c]));
        e = e + d * d;
    }
    return e;
}

// Final (q, nibble) for a winning (seed, e): plain rounding, or the best of the 3^P neighbourhood by decoded error.
static void finish_block(const SeedTab* tab, const float* w, int bias, int s, int e, const SearchOpts* o, uint16_t* nib,
                         float* err, const float* sh) {
    const SeedTab* T = &tab[s];
    float b[3] = {0, 0, 0};
    for (int c = 0; c < NSLM_C; ++c) { b[0] = b[0] + T->U[c * 3] * w[c]; b[1] = b[1] + T->U[c * 3 + 1] * w[c]; b[2] = b[2] + T->U[c * 3 + 2] * w[c]; }
    const float t[3] = {(T->Gi[0] * b[0] + T->Gi[1] * b[1]) + T->Gi[2] * b[2], (T->Gi[1] * b[0] + T->Gi[3] * b[1]) + T->Gi[4] * b[2],
                        (T->Gi[2] * b[0] + T->Gi[4] * b[1]) + T->Gi[5] * b[2]};
    const float inv = pow2f(-e);
    int q[3];
    for (int i = 0; i < 3; ++i) q[i] = (int) clampq((t[i] * inv + MAGIC) - MAGIC);
    int best[3] = {q[0], q[1], q[2]};
    float be = decoded_err(w, (uint16_t) s, e - bias, q, bias, sh);
    if (o->refit) {
        for (int d0 = -1; d0 <= 1; ++d0)
            for (int d1 = -1; d1 <= 1; ++d1)
                for (int d2 = -1; d2 <= 1; ++d2) {
                    const int c[3] = {q[0] + d0, q[1] + d1, q[2] + d2};
                    if (c[0] < -8 || c[0] > 7 || c[1] < -8 || c[1] > 7 || c[2] < -8 || c[2] > 7) continue;
                    const float ee = decoded_err(w, (uint16_t) s, e - bias, c, bias, sh);
                    if (ee < be) { be = ee; best[0] = c[0]; best[1] = c[1]; best[2] = c[2]; }
                }
    }
    *nib = nslm_pack(e - bias, best);
    if (err) *err = be;
}

// ---- scalar reference ----------------------------------------------------------------------------------------------

void nslm_search_ref(const SeedTab* tab, const float* w, int nb, int bias, const SearchOpts* o, uint16_t* seed,
                     uint16_t* nib, float* err, const float* sh) {
    const int lo = bias, hi = bias + 15;
    for (int k = 0; k < nb; ++k) {
        const float* x = w + (size_t) k * NSLM_C;
        float wn = 0;
        for (int c = 0; c < NSLM_C; ++c) wn = wn + x[c] * x[c];
        float best = INFINITY;
        int bs = 1, be = lo;
        for (int s = 1; s <= o->n_seeds; ++s) {
            const SeedTab* T = &tab[s];
            float b0 = 0, b1 = 0, b2 = 0;
            for (int c = 0; c < NSLM_C; ++c) {
                b0 = b0 + T->U[c * 3] * x[c];
                b1 = b1 + T->U[c * 3 + 1] * x[c];
                b2 = b2 + T->U[c * 3 + 2] * x[c];
            }
            const float t0 = (T->Gi[0] * b0 + T->Gi[1] * b1) + T->Gi[2] * b2;
            const float t1 = (T->Gi[1] * b0 + T->Gi[3] * b1) + T->Gi[4] * b2;
            const float t2 = (T->Gi[2] * b0 + T->Gi[4] * b1) + T->Gi[5] * b2;
            const float m = fmaxf(fmaxf(fabsf(t0), fabsf(t1)), fabsf(t2));
            int e0 = flog2f(m) - 2;
            e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
            for (int ci = 0; ci < o->n_exp; ++ci) {
                int e = e0 + o->exp_delta[ci];
                e = e < lo ? lo : (e > hi ? hi : e);
                const float inv = pow2f(-e), sc = pow2f(e);
                const float q0 = clampq((t0 * inv + MAGIC) - MAGIC);
                const float q1 = clampq((t1 * inv + MAGIC) - MAGIC);
                const float q2 = clampq((t2 * inv + MAGIC) - MAGIC);
                const float qb = (q0 * b0 + q1 * b1) + q2 * b2;
                const float z0 = (T->R[0] * q0 + T->R[1] * q1) + T->R[2] * q2, z1 = T->R[3] * q1 + T->R[4] * q2, z2 = T->R[5] * q2;
                const float qgq = (z0 * z0 + z1 * z1) + z2 * z2;
                const float rec = (sc * sc) * qgq;
                // |w_hat|^2 > 4 |w|^2 implies eps >= |w|^2 (never beats q = 0); rejecting keeps terms O(|w|^2) so
                // f32 cancellation cannot make a spurious minimum
                const float er = rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
                if (er < best) { best = er; bs = s; be = e; }
            }
        }
        seed[k] = (uint16_t) bs;
        finish_block(tab, x, bias, bs, be, o, &nib[k], err ? &err[k] : NULL, sh);
    }
}

#ifdef __clang__
// ---- Clang vectors: 4 blocks per lane group, NG groups per tile kept in L1, the seed loop outside ------------------

#define NG 8   // groups per tile (32 blocks)

static inline f4 vclampq(f4 r) { return __builtin_elementwise_min(__builtin_elementwise_max(r, (f4) -8.0f), (f4) 7.0f); }
static inline f4 vpow2(i4 e) { return (f4) ((e + 127) << 23); }   // bit cast

void nslm_search_vec(const SeedTab* tab, const float* w, int nb, int bias, const SearchOpts* o, uint16_t* seed,
                     uint16_t* nib, float* err, const float* sh) {
    const int lo = bias, hi = bias + 15;
    for (int k0 = 0; k0 < nb; k0 += 4 * NG) {
        f4 X[NG][NSLM_C], WN[NG], BEST[NG];
        i4 BS[NG], BE[NG];
        for (int g = 0; g < NG; ++g) {
            for (int c = 0; c < NSLM_C; ++c)
                for (int l = 0; l < 4; ++l) {
                    const int k = k0 + 4 * g + l;
                    X[g][c][l] = k < nb ? w[(size_t) k * NSLM_C + c] : 0.0f;
                }
            f4 wn = 0;
            for (int c = 0; c < NSLM_C; ++c) wn = wn + X[g][c] * X[g][c];
            WN[g] = wn;
            BEST[g] = (f4) INFINITY;
            BS[g] = (i4) 1;
            BE[g] = (i4) lo;
        }
        for (int s = 1; s <= o->n_seeds; ++s) {
            const SeedTab* T = &tab[s];
            const float* U = T->U;
            const float r00 = T->R[0], r01 = T->R[1], r02 = T->R[2], r11 = T->R[3], r12 = T->R[4], r22 = T->R[5];
            const float i00 = T->Gi[0], i01 = T->Gi[1], i02 = T->Gi[2], i11 = T->Gi[3], i12 = T->Gi[4], i22 = T->Gi[5];
            for (int g = 0; g < NG; ++g) {
                f4 b0 = 0, b1 = 0, b2 = 0;
                for (int c = 0; c < NSLM_C; ++c) {
                    const f4 x = X[g][c];
                    b0 = b0 + U[c * 3] * x;
                    b1 = b1 + U[c * 3 + 1] * x;
                    b2 = b2 + U[c * 3 + 2] * x;
                }
                const f4 t0 = (i00 * b0 + i01 * b1) + i02 * b2;
                const f4 t1 = (i01 * b0 + i11 * b1) + i12 * b2;
                const f4 t2 = (i02 * b0 + i12 * b1) + i22 * b2;
                const f4 m = __builtin_elementwise_max(__builtin_elementwise_max(__builtin_elementwise_abs(t0),
                                                                                 __builtin_elementwise_abs(t1)),
                                                       __builtin_elementwise_abs(t2));
                i4 e0 = (((i4) m >> 23) & 255) - 127 - 2;
                e0 = __builtin_elementwise_min(__builtin_elementwise_max(e0, (i4) lo), (i4) hi);
                for (int ci = 0; ci < o->n_exp; ++ci) {
                    const i4 e = __builtin_elementwise_min(__builtin_elementwise_max(e0 + o->exp_delta[ci], (i4) lo), (i4) hi);
                    const f4 inv = vpow2(-e), sc = vpow2(e);
                    const f4 q0 = vclampq((t0 * inv + MAGIC) - MAGIC);
                    const f4 q1 = vclampq((t1 * inv + MAGIC) - MAGIC);
                    const f4 q2 = vclampq((t2 * inv + MAGIC) - MAGIC);
                    const f4 qb = (q0 * b0 + q1 * b1) + q2 * b2;
                    const f4 z0 = (r00 * q0 + r01 * q1) + r02 * q2, z1 = r11 * q1 + r12 * q2, z2 = r22 * q2;
                    const f4 qgq = (z0 * z0 + z1 * z1) + z2 * z2;
                    const f4 rec = (sc * sc) * qgq;
                    const f4 er = rec > 4.0f * WN[g] ? (f4) INFINITY : (WN[g] - (2.0f * sc) * qb) + rec;
                    const i4 better = er < BEST[g];
                    BEST[g] = better ? er : BEST[g];
                    BS[g] = better ? (i4) s : BS[g];
                    BE[g] = better ? e : BE[g];
                }
            }
        }
        for (int g = 0; g < NG; ++g)
            for (int l = 0; l < 4; ++l) {
                const int k = k0 + 4 * g + l;
                if (k >= nb) continue;
                seed[k] = (uint16_t) BS[g][l];
                finish_block(tab, w + (size_t) k * NSLM_C, bias, BS[g][l], BE[g][l], o, &nib[k], err ? &err[k] : NULL, sh);
            }
    }
}
#else
// ---- without Clang vectors (MSVC, GCC): the same tiles and the same operations per lane, as loops over 4 lanes that
// the compiler may vectorize.  It must equal nslm_search_ref bit for bit, as the Clang version does (tests/test_search.c).

#define NG 8   // groups per tile (32 blocks)
#define LANES for (int l = 0; l < 4; ++l)

static inline float sclampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }
static inline float spow2(int e) { const uint32_t u = (uint32_t) (e + 127) << 23; float f; memcpy(&f, &u, 4); return f; }
static inline int fexp(float m) { uint32_t u; memcpy(&u, &m, 4); return (int) ((u >> 23) & 255); }

void nslm_search_vec(const SeedTab* tab, const float* w, int nb, int bias, const SearchOpts* o, uint16_t* seed,
                     uint16_t* nib, float* err, const float* sh) {
    const int lo = bias, hi = bias + 15;
    for (int k0 = 0; k0 < nb; k0 += 4 * NG) {
        float X[NG][NSLM_C][4], WN[NG][4], BEST[NG][4];
        int BS[NG][4], BE[NG][4];
        for (int g = 0; g < NG; ++g) {
            for (int c = 0; c < NSLM_C; ++c)
                LANES {
                    const int k = k0 + 4 * g + l;
                    X[g][c][l] = k < nb ? w[(size_t) k * NSLM_C + c] : 0.0f;
                }
            LANES {
                float wn = 0;
                for (int c = 0; c < NSLM_C; ++c) wn = wn + X[g][c][l] * X[g][c][l];
                WN[g][l] = wn;
                BEST[g][l] = INFINITY;
                BS[g][l] = 1;
                BE[g][l] = lo;
            }
        }
        for (int s = 1; s <= o->n_seeds; ++s) {
            const SeedTab* T = &tab[s];
            const float* U = T->U;
            const float r00 = T->R[0], r01 = T->R[1], r02 = T->R[2], r11 = T->R[3], r12 = T->R[4], r22 = T->R[5];
            const float i00 = T->Gi[0], i01 = T->Gi[1], i02 = T->Gi[2], i11 = T->Gi[3], i12 = T->Gi[4], i22 = T->Gi[5];
            for (int g = 0; g < NG; ++g) {
                float b0[4] = {0, 0, 0, 0}, b1[4] = {0, 0, 0, 0}, b2[4] = {0, 0, 0, 0};
                for (int c = 0; c < NSLM_C; ++c)
                    LANES {
                        const float x = X[g][c][l];
                        b0[l] = b0[l] + U[c * 3] * x;
                        b1[l] = b1[l] + U[c * 3 + 1] * x;
                        b2[l] = b2[l] + U[c * 3 + 2] * x;
                    }
                LANES {
                    const float t0 = (i00 * b0[l] + i01 * b1[l]) + i02 * b2[l];
                    const float t1 = (i01 * b0[l] + i11 * b1[l]) + i12 * b2[l];
                    const float t2 = (i02 * b0[l] + i12 * b1[l]) + i22 * b2[l];
                    const float a0 = fabsf(t0), a1 = fabsf(t1), a2 = fabsf(t2);
                    const float m01 = a0 > a1 ? a0 : a1, m = m01 > a2 ? m01 : a2;
                    int e0 = fexp(m) - 127 - 2;
                    e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                    for (int ci = 0; ci < o->n_exp; ++ci) {
                        int e = e0 + o->exp_delta[ci];
                        e = e < lo ? lo : (e > hi ? hi : e);
                        const float inv = spow2(-e), sc = spow2(e);
                        const float q0 = sclampq((t0 * inv + MAGIC) - MAGIC);
                        const float q1 = sclampq((t1 * inv + MAGIC) - MAGIC);
                        const float q2 = sclampq((t2 * inv + MAGIC) - MAGIC);
                        const float qb = (q0 * b0[l] + q1 * b1[l]) + q2 * b2[l];
                        const float z0 = (r00 * q0 + r01 * q1) + r02 * q2, z1 = r11 * q1 + r12 * q2, z2 = r22 * q2;
                        const float qgq = (z0 * z0 + z1 * z1) + z2 * z2;
                        const float rec = (sc * sc) * qgq;
                        const float er = rec > 4.0f * WN[g][l] ? INFINITY : (WN[g][l] - (2.0f * sc) * qb) + rec;
                        if (er < BEST[g][l]) { BEST[g][l] = er; BS[g][l] = s; BE[g][l] = e; }
                    }
                }
            }
        }
        for (int g = 0; g < NG; ++g)
            LANES {
                const int k = k0 + 4 * g + l;
                if (k >= nb) continue;
                seed[k] = (uint16_t) BS[g][l];
                finish_block(tab, w + (size_t) k * NSLM_C, bias, BS[g][l], BE[g][l], o, &nib[k], err ? &err[k] : NULL, sh);
            }
    }
}
#undef LANES
#endif

float nslm_candidate_err(const SeedTab* T, const float* x, float wn, int e) {
    float b0 = 0, b1 = 0, b2 = 0;
    for (int c = 0; c < NSLM_C; ++c) {
        b0 = b0 + T->U[c * 3] * x[c];
        b1 = b1 + T->U[c * 3 + 1] * x[c];
        b2 = b2 + T->U[c * 3 + 2] * x[c];
    }
    const float t0 = (T->Gi[0] * b0 + T->Gi[1] * b1) + T->Gi[2] * b2;
    const float t1 = (T->Gi[1] * b0 + T->Gi[3] * b1) + T->Gi[4] * b2;
    const float t2 = (T->Gi[2] * b0 + T->Gi[4] * b1) + T->Gi[5] * b2;
    const float inv = pow2f(-e), sc = pow2f(e);
    const float q0 = clampq((t0 * inv + MAGIC) - MAGIC), q1 = clampq((t1 * inv + MAGIC) - MAGIC), q2 = clampq((t2 * inv + MAGIC) - MAGIC);
    const float qb = (q0 * b0 + q1 * b1) + q2 * b2;
    const float z0 = (T->R[0] * q0 + T->R[1] * q1) + T->R[2] * q2, z1 = T->R[3] * q1 + T->R[4] * q2, z2 = T->R[5] * q2;
    const float qgq = (z0 * z0 + z1 * z1) + z2 * z2;
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - (2.0f * sc) * qb) + rec;
}

int nslm_choose_bias(const float* w, int64_t nblocks, int64_t* clamped_low) {
    int top = -1000;
    for (int64_t k = 0; k < nblocks; ++k) {
        float n2 = 0;
        for (int c = 0; c < NSLM_C; ++c) n2 += w[k * NSLM_C + c] * w[k * NSLM_C + c];
        if (n2 > 0) { const int v = flog2f(2.0f * sqrtf(n2)) - 2; if (v > top) top = v; }
    }
    const int bias = top + 1 - 15;
    if (clamped_low) {
        int64_t n = 0;
        for (int64_t k = 0; k < nblocks; ++k) {
            float n2 = 0;
            for (int c = 0; c < NSLM_C; ++c) n2 += w[k * NSLM_C + c] * w[k * NSLM_C + c];
            if (n2 > 0 && flog2f(2.0f * sqrtf(n2)) - 2 < bias) ++n;
        }
        *clamped_low = n;
    }
    return bias;
}
