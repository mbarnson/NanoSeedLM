// tests/test_mova_kernels.c - the MoVA engine's kernels against C scalar references, one at a time, on every GPU backend
// (tests/kernel_backend.h: CUDA's engine/kernels_moe.cu, Metal's engine/kernels_moe.metal).  References compute in double
// with BF16 rounding at the kernels' points; outputs match within one BF16 ulp plus the f32 accumulation bound.  Seed
// formats are checked against the exact (unrounded) seed weights, and, where the GEMM has a BF16-rounded mode (CUDA's
// default), that mode against the BF16-rounded weights.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "kernel_backend.h"
#include "lfsr.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define RUN(call) do { if ((call) != 0) { ++fails; return; } } while (0)

static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static float bfr(double x) { return bf2f(f2bf((float) x)); }
static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }
// |a - b| within one BF16 ulp of b, plus extra
static int close_bf(float a, double b, double extra) {
    const double ulp = fabs(b) * (1.0 / 128.0) + 1e-30;
    return fabs(a - b) <= ulp + extra;
}
static const char* kNames[5] = {"bf16", "seed4", "q8", "q4", "seed4p4"};

// ---- weights in every format: S slices of R x K, and the reference value of each element ---------------------------
typedef struct {
    int S, R, K;
    uint16_t* w;                                   // BF16
    uint32_t *q8, *q4;
    uint16_t *s8, *b8, *s4, *b4, *d8, *d4;         // affine streams; d8 / d4: dequantized to BF16
    uint16_t *seeds, *nibs;                        // SEED4
    int32_t ebias[3];
    double* wseed;                                 // SEED4: exact weights
    uint16_t *p4seeds, *p4coefs;                   // SEED4P4
    uint8_t* p4nib;
    int32_t p4bias[3];
    double* p4w;                                   // SEED4P4: exact weights
} Weights;

static Weights make_weights(int S, int R, int K, unsigned* sd) {
    Weights m;
    memset(&m, 0, sizeof m);
    m.S = S; m.R = R; m.K = K;
    const size_t n = (size_t) S * R * K, nb = n / 8;
    m.w = malloc(2 * n);
    for (size_t i = 0; i < n; ++i) m.w[i] = f2bf((float) (frand(sd) * 0.05));
    m.q8 = malloc(n); m.q4 = malloc(n / 2);
    m.s8 = malloc(2 * n / 64); m.b8 = malloc(2 * n / 64); m.s4 = malloc(2 * n / 64); m.b4 = malloc(2 * n / 64);
    m.d8 = malloc(2 * n); m.d4 = malloc(2 * n);
    nslm_affine_quantize(m.w, S * R, K, 8, m.q8, m.s8, m.b8);
    nslm_affine_quantize(m.w, S * R, K, 4, m.q4, m.s4, m.b4);
    nslm_affine_dequantize(m.q8, m.s8, m.b8, S * R, K, 8, m.d8);
    nslm_affine_dequantize(m.q4, m.s4, m.b4, S * R, K, 4, m.d4);
    // SEED4: random seeds and nibble words; reference = the exact unrounded weights
    m.seeds = malloc(2 * nb); m.nibs = malloc(2 * nb);
    m.ebias[0] = -18; m.ebias[1] = -16; m.ebias[2] = -20;
    for (size_t i = 0; i < nb; ++i) { m.seeds[i] = (uint16_t) (1 + (*sd = *sd * 1103515245u + 12345u) % 65535); m.nibs[i] = (uint16_t) ((*sd >> 3) & 0xFFFF); }
    m.wseed = malloc(sizeof(double) * n);
    for (size_t bi = 0; bi < nb; ++bi) {
        const int s = (int) (bi / ((size_t) R * K / 8));
        uint16_t st[24];
        lfsr_states(m.seeds[bi], 24, st);
        const uint16_t nw = m.nibs[bi];
        const double sc = (double) NSLM_R32 * pow(2.0, m.ebias[s] + nslm_ecode(nw));
        for (int c = 0; c < 8; ++c)
            m.wseed[bi * 8 + c] = sc * (((double) st[3 * c] - 32768) * nslm_q(nw, 0) + ((double) st[3 * c + 1] - 32768) * nslm_q(nw, 1) +
                                        ((double) st[3 * c + 2] - 32768) * nslm_q(nw, 2));
    }
    // SEED4P4 (4.5-bit, P = 4): seeds, coefficient words, exponent codes (two blocks per byte, low first)
    m.p4seeds = malloc(2 * nb); m.p4coefs = malloc(2 * nb); m.p4nib = calloc(nb / 2 + 1, 1); m.p4w = malloc(sizeof(double) * n);
    m.p4bias[0] = -19; m.p4bias[1] = -17; m.p4bias[2] = -21;
    for (size_t b = 0; b < nb; ++b) {
        *sd = *sd * 1103515245u + 12345u;
        m.p4seeds[b] = (uint16_t) (1 + (*sd >> 8) % 65535);
        *sd = *sd * 1103515245u + 12345u;
        m.p4coefs[b] = (uint16_t) (*sd >> 9);
        const int ec = (int) ((*sd >> 3) % 6);
        m.p4nib[b / 2] |= (uint8_t) (ec << (4 * (b & 1)));
        const int s = (int) (b / ((size_t) R * K / 8));
        uint16_t st[32];
        lfsr_states(m.p4seeds[b], 32, st);
        const double sc = (double) NSLM_R32 * pow(2.0, m.p4bias[s] + ec);
        for (int c = 0; c < 8; ++c) {
            double v = 0;
            for (int p = 0; p < 4; ++p) v += ((double) st[4 * c + p] - 32768) * (double) ((int32_t) ((uint32_t) m.p4coefs[b] << (28 - 4 * p)) >> 28);
            m.p4w[b * 8 + c] = sc * v;
        }
    }
    return m;
}
static void free_weights(Weights* m) {
    free(m->w); free(m->q8); free(m->q4); free(m->s8); free(m->b8); free(m->s4); free(m->b4); free(m->d8); free(m->d4);
    free(m->seeds); free(m->nibs); free(m->wseed); free(m->p4seeds); free(m->p4coefs); free(m->p4nib); free(m->p4w);
}
static KtWeight kt_weight(const Weights* m, int fmt) {
    const size_t n = (size_t) m->S * m->R * m->K, nb = n / 8;
    KtWeight k;
    memset(&k, 0, sizeof k);
    k.fmt = fmt; k.S = m->S; k.R = m->R; k.K = m->K;
    switch (fmt) {
    case MF_BF16: k.w = m->w; k.wn = 2 * n; break;
    case MF_Q8: k.w = m->q8; k.wn = n; k.s = m->s8; k.b = m->b8; k.sn = k.bn = 2 * n / 64; break;
    case MF_Q4: k.w = m->q4; k.wn = n / 2; k.s = m->s4; k.b = m->b4; k.sn = k.bn = 2 * n / 64; break;
    case MF_SEED4: k.w = m->seeds; k.s = m->nibs; k.wn = k.sn = 2 * nb; k.b = m->ebias; k.bn = 4 * (size_t) m->S; break;
    default: k.w = m->p4seeds; k.s = m->p4coefs; k.wn = k.sn = 2 * nb; k.b = m->p4bias; k.bn = 4 * (size_t) m->S; k.e = m->p4nib; k.en = nb / 2 + 1;
    }
    return k;
}
// The reference weight of element (slice, row, col).  mv: Q4 is MLX's qmv form (s * q + b unrounded, i.e. per group
// s * sum q x + b * sum x); mm: Q4 dequantizes to BF16 as mx.dequantize.  The other formats dequantize to BF16 exactly.
static double wref(const Weights* m, int fmt, int mv, int s, int r, int c) {
    const size_t i = ((size_t) s * m->R + r) * m->K + c;
    switch (fmt) {
    case MF_BF16: return bf2f(m->w[i]);
    case MF_Q8: return bf2f(m->d8[i]);
    case MF_Q4: return mv ? (double) bf2f(m->s4[i / 64]) * (double) ((m->q4[i / 8] >> (4 * (i % 8))) & 15u) + bf2f(m->b4[i / 64]) : bf2f(m->d4[i]);
    case MF_SEED4: return m->wseed[i];
    default: return m->p4w[i];
    }
}

// ---- matvec: every format, dense T = 1..8 and gather ---------------------------------------------------------------
static void test_mv(int K) {   // K = 256 and 512: the matvec's one- and two-blocks-per-lane seed paths
    const int R = 24, S = 3;
    unsigned sd = 1;
    float* x = malloc(4 * (size_t) MV_MAXT * K), *xg = malloc(4 * (size_t) 16 * K);
    Weights m = make_weights(S, R, K, &sd);
    for (int i = 0; i < MV_MAXT * K; ++i) x[i] = bfr(frand(&sd));
    for (int i = 0; i < 16 * K; ++i) xg[i] = bfr(frand(&sd));
    for (int fmt = 0; fmt < 5; ++fmt) {
        const KtWeight kw = kt_weight(&m, fmt);
        const double f32b = (fmt == MF_SEED4 || fmt == MF_SEED4P4 ? 64 : 48) * 1.2e-7;   // f32 accumulation, per |w x|
        double worst = 0;
        for (int T = 1; T <= MV_MAXT; ++T) {   // dense on slice 0; also the residual add for T = 3
            const int add = T == 3;
            float y0[MV_MAXT * 24], y[MV_MAXT * 24];
            for (int i = 0; i < MV_MAXT * R; ++i) y[i] = y0[i] = bfr(frand(&sd));
            if (kt_mv(&kw, T, add, x, y)) { ++fails; break; }
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < R; ++r) {
                    double acc = 0, mag = 0;
                    for (int c = 0; c < K; ++c) { const double v = wref(&m, fmt, 1, 0, r, c) * x[t * K + c]; acc += v; mag += fabs(v); }
                    const double ref = add ? bfr(y0[t * R + r] + bfr(acc)) : bfr(acc), bound = f32b * mag * (add ? 2 : 1) + 1e-30;
                    if (!close_bf(y[t * R + r], ref, bound)) {
                        ++fails;
                        if (fails < 10) printf("FAIL mv %s dense T=%d t=%d r=%d: %g vs %g\n", kNames[fmt], T, t, r, y[t * R + r], ref);
                    }
                    worst = fmax(worst, fabs(y[t * R + r] - ref) / (fabs(ref) / 128 + bound));
                }
        }
        {   // gather: 6 pairs, slices sel[p], inputs p / xdiv (xdiv 2)
            const int P = 6, xdiv = 2;
            const int32_t sel[6] = {2, 0, 1, 2, 1, 0};
            float y[6 * 24];
            if (kt_mv_sel(&kw, P, xdiv, sel, xg, y)) ++fails;
            else
                for (int p = 0; p < P; ++p)
                    for (int r = 0; r < R; ++r) {
                        double acc = 0, mag = 0;
                        for (int c = 0; c < K; ++c) { const double v = wref(&m, fmt, 1, sel[p], r, c) * xg[(p / xdiv) * K + c]; acc += v; mag += fabs(v); }
                        const double ref = bfr(acc), bound = f32b * mag;
                        if (!close_bf(y[p * R + r], ref, bound)) {
                            ++fails;
                            if (fails < 10) printf("FAIL mv %s gather p=%d r=%d: %g vs %g\n", kNames[fmt], p, r, y[p * R + r], ref);
                        }
                        worst = fmax(worst, fabs(y[p * R + r] - ref) / (fabs(ref) / 128 + bound));
                    }
        }
        {   // gate + up + SwiGLU: gate = slice sel[p], up = slice sel[p] + 1; a = bf16(silu(bf16 g) * bf16 u), within one
            // BF16 flip of g or u
            const int P = 6, xdiv = 2;
            const int32_t sel[6] = {1, 0, 1, 1, 0, 0};
            float a[6 * 24];
            int bad = 0;
            if (kt_mv_gu(&kw, P, xdiv, sel, xg, a)) ++fails;
            else
                for (int p = 0; p < P; ++p)
                    for (int r = 0; r < R; ++r) {
                        double ag = 0, au = 0;
                        for (int c = 0; c < K; ++c) {
                            ag += wref(&m, fmt, 1, sel[p], r, c) * xg[(p / xdiv) * K + c];
                            au += wref(&m, fmt, 1, sel[p] + 1, r, c) * xg[(p / xdiv) * K + c];
                        }
                        const double gg = bfr(ag), uu = bfr(au);
                        const double ref = bfr(bfr(gg * bfr(1 / (1 + exp(-gg)))) * uu);
                        const double slack = (fabs(uu) * (fabs(gg) + 1) + fabs(gg) * fabs(uu)) / 128 + 1e-7;
                        if (!close_bf(a[p * R + r], ref, slack)) {
                            ++bad;
                            if (bad < 4) printf("FAIL mv_gu %s p=%d r=%d: %g vs %g\n", kNames[fmt], p, r, a[p * R + r], ref);
                        }
                    }
            CHECK(!bad, "mv_gu %s: %d mismatches", kNames[fmt], bad);
        }
        printf("mv %-7s K %d dense T=1..8 + gather: worst error / (1 bf16 ulp + f32 bound) %.3f\n", kNames[fmt], K, worst);
    }
    free_weights(&m);
    free(x); free(xg);
}

// ---- GEMM: dense (T tokens, partial tiles, residual) and grouped (a tile table over a permuted pair list) -----------
// exact: the seed reference weights unrounded (the GEMM's exact mode), else rounded to BF16 (a backend's BF16 mode)
static void test_mm_pass(const Weights* m, const float* x, int T, int P, int xdiv, int fmt, int exact, unsigned* sd) {
    const int R = m->R, K = m->K, S = m->S, BN = kt_mm_tile();
    const KtWeight kw = kt_weight(m, fmt);
    const int seedf = fmt == MF_SEED4 || fmt == MF_SEED4P4;
    int bad = 0;
    {   // dense, slice 0, residual add
        float* y0 = malloc(4 * (size_t) T * R), *y = malloc(4 * (size_t) T * R);
        for (int i = 0; i < T * R; ++i) y[i] = y0[i] = bfr(frand(sd));
        if (kt_mm(&kw, T, x, y)) ++bad;
        else
            for (int n = 0; n < T; ++n)
                for (int r = 0; r < R; ++r) {
                    double acc = 0, mag = 0;
                    for (int c = 0; c < K; ++c) {
                        const double wv = seedf && !exact ? bfr(wref(m, fmt, 0, 0, r, c)) : wref(m, fmt, 0, 0, r, c);
                        acc += wv * x[n * K + c];
                        mag += fabs(wv * x[n * K + c]);
                    }
                    const double ref = bfr(y0[n * R + r] + bfr(acc));
                    if (!close_bf(y[n * R + r], ref, 2 * 64 * 1.2e-7 * mag) && bad++ < 3)
                        printf("FAIL mm %s dense n=%d r=%d: %g vs %g\n", kNames[fmt], n, r, y[n * R + r], ref);
                }
        free(y0); free(y);
    }
    {   // grouped: P pairs (input pair / xdiv), slices assigned per pair, sorted into a tile table (<= BN pairs per tile)
        int32_t* slc = malloc(4 * (size_t) P), *perm = malloc(4 * (size_t) P);
        MmTile* tl = malloc(sizeof(MmTile) * (size_t) (P + S));
        int np = 0, nt = 0;
        for (int p = 0; p < P; ++p) slc[p] = (p * 7) % S;
        for (int s = 0; s < S; ++s) {
            const int st = np;
            for (int p = 0; p < P; ++p) if (slc[p] == s) perm[np++] = p;
            for (int o = st; o < np; o += BN) tl[nt++] = (MmTile) {s, o, np - o < BN ? np - o : BN, 0};
        }
        float* y = malloc(4 * (size_t) P * R);
        if (kt_mm_grouped(&kw, P, xdiv, perm, tl, nt, x, y)) ++bad;
        else
            for (int p = 0; p < P; ++p)
                for (int r = 0; r < R; ++r) {
                    double acc = 0, mag = 0;
                    for (int c = 0; c < K; ++c) {
                        const double wv = seedf && !exact ? bfr(wref(m, fmt, 0, slc[p], r, c)) : wref(m, fmt, 0, slc[p], r, c);
                        acc += wv * x[(p / xdiv) * K + c];
                        mag += fabs(wv * x[(p / xdiv) * K + c]);
                    }
                    if (!close_bf(y[p * R + r], bfr(acc), 64 * 1.2e-7 * mag) && bad++ < 6)
                        printf("FAIL mm %s grouped p=%d r=%d: %g vs %g\n", kNames[fmt], p, r, y[p * R + r], bfr(acc));
                }
        free(slc); free(perm); free(tl); free(y);
    }
    CHECK(!bad, "mm %s%s: %d mismatches", kNames[fmt], exact ? "" : " (BF16 seed weights)", bad);
    printf("mm %-7s%s dense (T=%d, residual) + grouped (%d pairs, %d slices): %s\n", kNames[fmt],
           seedf ? (exact ? " exact" : " bf16 ") : "      ", T, P, S, bad ? "FAIL" : "ok");
}
static void test_mm(void) {
    unsigned sd = 3;
    Weights m = make_weights(3, 40, 128, &sd);   // R 40: not a multiple of a tile; K 128: four K steps, two affine groups
    const int T = 45, P = 50, xdiv = 2;
    float* x = malloc(4 * (size_t) T * m.K);
    for (int i = 0; i < T * m.K; ++i) x[i] = bfr(frand(&sd));
    const int has_bf16 = kt_seed_gemm_exact(1);
    for (int fmt = 0; fmt < 5; ++fmt) test_mm_pass(&m, x, T, P, xdiv, fmt, 1, &sd);
    if (has_bf16) {
        kt_seed_gemm_exact(0);
        test_mm_pass(&m, x, T, P, xdiv, MF_SEED4, 0, &sd);
        test_mm_pass(&m, x, T, P, xdiv, MF_SEED4P4, 0, &sd);
    }
    free_weights(&m);
    free(x);
}

// ---- grouped norm, router, swiglu, combines, rope + KV + attention, embed, argmax ------------------------------------
static void test_misc(void) {
    unsigned sd = 7;
    enum { d = 256, T = 3 };
    float x[T * d], y[T * d];
    uint16_t wn[d];
    for (int i = 0; i < T * d; ++i) x[i] = bfr(frand(&sd) * 3);
    for (int i = 0; i < d; ++i) wn[i] = f2bf((float) (1 + 0.3 * frand(&sd)));
    {   // grouped RMSNorm (2 groups)
        RUN(kt_gnorm(d, 1e-6f, x, wn, y, T));
        int bad = 0;
        for (int t = 0; t < T; ++t)
            for (int g = 0; g < 2; ++g) {
                double ss = 0;
                for (int c = 0; c < d / 2; ++c) ss += (double) x[t * d + g * d / 2 + c] * x[t * d + g * d / 2 + c];
                const double r = 1.0 / sqrt(ss / (d / 2) + 1e-6);
                for (int c = 0; c < d / 2; ++c) {
                    const int i = g * d / 2 + c;
                    bad += !close_bf(y[t * d + i], bfr(bf2f(wn[i]) * (x[t * d + i] * r)), 1e-6);
                }
            }
        CHECK(!bad, "gnorm: %d mismatches", bad);
    }
    {   // router: n 20, top 4, partitions of d / 2, BF16 weights + bias
        enum { n = 20, k = 4 };
        uint16_t W[n * d], bias[n];
        for (int i = 0; i < n * d; ++i) W[i] = f2bf((float) (frand(&sd) * 0.05));
        for (int i = 0; i < n; ++i) bias[i] = f2bf((float) (frand(&sd) * 0.01));
        const RouterArgs a = {d, n, k, d / 2, 2.5f};
        int32_t ind[T * k];
        float wt[T * k];
        RUN(kt_router(a, W, bias, x, T, ind, wt));
        int bad = 0;
        for (int t = 0; t < T; ++t) {
            double score[64], sel[64];
            for (int e = 0; e < n; ++e) {
                double s0 = 0, s1 = 0;
                for (int c = 0; c < d / 2; ++c) s0 += bf2f(W[e * d + c]) * x[t * d + c];
                for (int c = d / 2; c < d; ++c) s1 += bf2f(W[e * d + c]) * x[t * d + c];
                score[e] = 1 / (1 + exp(-((double) bfr(s0) + bfr(s1))));
                sel[e] = score[e] + bf2f(bias[e]);
            }
            int ch[8];
            double sum = 0;
            for (int j = 0; j < k; ++j) {
                int best = -1;
                for (int e = 0; e < n; ++e) {
                    int tk = 0;
                    for (int q = 0; q < j; ++q) tk |= ch[q] == e;
                    if (!tk && (best < 0 || sel[e] > sel[best])) best = e;
                }
                ch[j] = best;
                sum += score[best];
            }
            for (int j = 0; j < k; ++j) {
                bad += ind[t * k + j] != ch[j];
                bad += fabs(wt[t * k + j] - score[ch[j]] / sum * 2.5) > 1e-5;
            }
        }
        CHECK(!bad, "router: %d mismatches", bad);
    }
    for (int shape = 0; shape < 2; ++shape) {   // the top-k step alone at MoVA's shapes, with exact ties (lowest id wins)
        const int n = shape ? 64 : 100, k = shape ? 4 : 8;
        enum { TK = 6 };
        float sc[TK * 100], se[TK * 100], wt[TK * 8];
        int32_t ind[TK * 8];
        for (int i = 0; i < TK * n; ++i) {
            sc[i] = (float) frand(&sd) * 0.5f + 0.5f;
            se[i] = (float) ((int) (frand(&sd) * 12)) / 16.0f;   // few distinct values: many ties
        }
        const RouterArgs a = {d, n, k, d / 2, 2.5f};
        RUN(kt_router_topk(a, sc, se, TK, ind, wt));
        int bad = 0;
        for (int t = 0; t < TK; ++t) {
            int ch[8];
            double sum = 0;
            for (int j = 0; j < k; ++j) {
                int best = -1;
                for (int e = 0; e < n; ++e) {
                    int tk = 0;
                    for (int q = 0; q < j; ++q) tk |= ch[q] == e;
                    if (!tk && (best < 0 || se[t * n + e] > se[t * n + best])) best = e;
                }
                ch[j] = best;
                sum += sc[t * n + best];
            }
            for (int j = 0; j < k; ++j) {
                bad += ind[t * k + j] != ch[j];
                bad += fabs(wt[t * k + j] - sc[t * n + ch[j]] / sum * 2.5) > 1e-5;
            }
        }
        CHECK(!bad, "router top-k n %d k %d (ties): %d mismatches", n, k, bad);
    }
    {   // swiglu, MoE combine, value combine
        enum { n = 64 };
        float g[n], u[n], a[n];
        for (int i = 0; i < n; ++i) { g[i] = bfr(frand(&sd) * 4); u[i] = bfr(frand(&sd)); }
        RUN(kt_swiglu(g, u, a, n));
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const double s = bfr(1 / (1 + exp(-(double) g[i]))), si = bfr(g[i] * s), ref = bfr(si * u[i]);
            bad += !close_bf(a[i], ref, 1e-7);
        }
        CHECK(!bad, "swiglu: %d mismatches", bad);
        enum { dd = 32, k = 3, TT = 2 };
        float D[TT * k * dd], w[TT * k], sh[TT * dd], xx[TT * dd], x0[TT * dd], v[TT * dd];
        for (int i = 0; i < TT * k * dd; ++i) D[i] = bfr(frand(&sd));
        for (int i = 0; i < TT * k; ++i) w[i] = (float) (0.3 + 0.5 * fabs(frand(&sd)));
        for (int i = 0; i < TT * dd; ++i) { sh[i] = bfr(frand(&sd)); xx[i] = x0[i] = bfr(frand(&sd)); }
        RUN(kt_combine(D, w, sh, xx, v, dd, k, TT));
        int bc = 0, bv = 0;
        for (int t = 0; t < TT; ++t)
            for (int c = 0; c < dd; ++c) {
                double s = 0, vv = 0;
                for (int j = 0; j < k; ++j) {
                    const double dv = D[(t * k + j) * dd + c], wj = bfr(w[t * k + j]);
                    s += bfr(dv * wj);
                    const double sg = bfr(1 / (1 + exp(-dv))), si = bfr(dv * sg);
                    vv += bfr(si * wj);
                }
                bc += !close_bf(xx[t * dd + c], bfr(x0[t * dd + c] + bfr(bfr(s) + sh[t * dd + c])), 1e-6);
                bv += !close_bf(v[t * dd + c], bfr(vv), 1e-6);
            }
        CHECK(!bc && !bv, "moe combine %d, value combine %d mismatches", bc, bv);
    }
    {   // rope + KV + decode attention + gate: 4 query heads on 1 KV head, 3 cached positions, rows at 3 and 4
        enum { nh = 4, nkv = 1, P = 5, T2 = 2 };
        float q[T2 * nh * 128], k[T2 * nkv * 128], v[T2 * nkv * 128], g[T2 * nh * 128], o[T2 * nh * 128], qcopy[T2 * nh * 128];
        for (int i = 0; i < T2 * nh * 128; ++i) { q[i] = bfr(frand(&sd)); g[i] = bfr(frand(&sd) * 3); }
        for (int i = 0; i < T2 * nkv * 128; ++i) { k[i] = bfr(frand(&sd)); v[i] = bfr(frand(&sd)); }
        uint16_t Kc[P * 128], Vc[P * 128];
        memset(Kc, 0, sizeof Kc);
        memset(Vc, 0, sizeof Vc);
        for (int i = 0; i < 3 * 128; ++i) { Kc[i] = f2bf((float) frand(&sd)); Vc[i] = f2bf((float) frand(&sd)); }
        float inv[64];
        for (int p = 0; p < 64; ++p) inv[p] = (float) pow(1e7, -2.0 * p / 128);
        const RowInfo ri[2] = {{3, {0}}, {4, {0}}};
        memcpy(qcopy, q, sizeof q);
        double kc[P][128], vc[P][128];
        for (int p = 0; p < 3; ++p) for (int e = 0; e < 128; ++e) { kc[p][e] = bf2f(Kc[p * 128 + e]); vc[p][e] = bf2f(Vc[p * 128 + e]); }
        const AttnArgs a = {nh, nkv, 3, 2, (float) (1 / sqrt(128.0))};
        RUN(kt_rope_attn(a, q, k, v, Kc, Vc, P, ri, inv, g, o, T2));
        // reference: rope, cache, causal attention, gate
        double qr[T2][nh][128];
        for (int t = 0; t < T2; ++t) {
            const int pos = ri[t].pos;
            for (int i = 0; i < 64; ++i) {
                const double th = (double) (float) ((float) pos * inv[i]), cs = cos(th), sn = sin(th);
                for (int h = 0; h < nh; ++h) {
                    const double x0 = qcopy[t * nh * 128 + h * 128 + i], x1 = qcopy[t * nh * 128 + h * 128 + i + 64];
                    qr[t][h][i] = bfr(x0 * cs - x1 * sn);
                    qr[t][h][i + 64] = bfr(x1 * cs + x0 * sn);
                }
                const double k0 = k[t * 128 + i], k1 = k[t * 128 + i + 64];
                kc[pos][i] = bfr(k0 * cs - k1 * sn);
                kc[pos][i + 64] = bfr(k1 * cs + k0 * sn);
            }
            for (int e = 0; e < 128; ++e) vc[pos][e] = v[t * 128 + e];
        }
        int bad = 0;
        for (int t = 0; t < T2; ++t)
            for (int h = 0; h < nh; ++h) {
                double s[P], mx = -1e300, l = 0;
                for (int p = 0; p <= ri[t].pos; ++p) {
                    double acc = 0;
                    for (int e = 0; e < 128; ++e) acc += qr[t][h][e] * a.scale * kc[p][e];
                    s[p] = acc;
                    mx = fmax(mx, acc);
                }
                for (int p = 0; p <= ri[t].pos; ++p) l += exp(s[p] - mx);
                for (int e = 0; e < 128; ++e) {
                    double acc = 0;
                    for (int p = 0; p <= ri[t].pos; ++p) acc += exp(s[p] - mx) / l * vc[p][e];
                    const double gx = g[t * nh * 128 + h * 128 + e] * 0.69314718055994531;
                    const double sp = bfr((fmax(gx, 0) + log1p(exp(-fabs(gx)))) / 0.69314718055994531);
                    if (!close_bf(o[t * nh * 128 + h * 128 + e], bfr(bfr(acc) * sp), 2e-6)) ++bad;
                }
            }
        CHECK(!bad, "rope + attention + gate: %d mismatches of %d", bad, T2 * nh * 128);
    }
    {   // embedding (BF16 and Q8) and argmax
        enum { V = 40, dd = 64 };
        uint16_t E[V * dd], s8[V], b8[V], deq[V * dd];
        uint32_t q8[V * dd / 4];
        for (int i = 0; i < V * dd; ++i) E[i] = f2bf((float) frand(&sd));
        nslm_affine_quantize(E, V, dd, 8, q8, s8, b8);
        nslm_affine_dequantize(q8, s8, b8, V, dd, 8, deq);
        const int32_t ids[3] = {7, 0, 39};
        float xe[3 * dd];
        int bad = 0;
        for (int fmt = 0; fmt < 3; fmt += 2) {
            RUN(kt_embed(fmt, E, q8, s8, b8, V, dd, ids, 3, xe));
            for (int t = 0; t < 3; ++t)
                for (int c = 0; c < dd; ++c) bad += xe[t * dd + c] != bf2f(fmt == MF_Q8 ? deq[ids[t] * dd + c] : E[ids[t] * dd + c]);
        }
        CHECK(!bad, "embed: %d mismatches", bad);
        const int VV = 250624;
        float* lg = malloc(4 * (size_t) VV * 2);
        for (int i = 0; i < 2 * VV; ++i) lg[i] = bfr(frand(&sd) * 10);
        lg[123456] = 50; lg[200000] = 50;   // a tie: the lowest id wins
        lg[VV + 99] = 60;
        int32_t am[2] = {-1, -1};
        if (kt_argmax(lg, VV, 2, am)) ++fails;
        CHECK(am[0] == 123456 && am[1] == 99, "argmax: %d %d", am[0], am[1]);
        free(lg);
    }
}

// The softplus gate (log2 base, as the kernels) and the reference output of one (row, head, dim).
static double gated(double acc, float g) {
    const double gx = g * 0.69314718055994531;
    const double sp = bfr((fmax(gx, 0) + log1p(exp(-fabs(gx)))) / 0.69314718055994531);
    return bfr(bfr(acc) * sp);
}

// ---- the 8-bit KV cache: a BF16 cache of np positions x nkv heads quantized per (position, head) (scale = max |x| / 127,
// q = round(x / scale)), and the BF16 cache replaced by the values the kernels read, bf16(q * scale) ----------------------
static KtKvQ8 q8_cache(uint16_t* Kc, uint16_t* Vc, int np, int nkv) {
    KtKvQ8 kv = {malloc((size_t) np * nkv * 128), malloc((size_t) np * nkv * 128), malloc(4 * (size_t) np * nkv),
                 malloc(4 * (size_t) np * nkv)};
    for (int which = 0; which < 2; ++which) {
        uint16_t* c = which ? Vc : Kc;
        int8_t* q = which ? kv.v : kv.k;
        float* sc = which ? kv.vs : kv.ks;
        for (size_t r = 0; r < (size_t) np * nkv; ++r) {
            float m = 0;
            for (int e = 0; e < 128; ++e) m = fmaxf(m, fabsf(bf2f(c[r * 128 + e])));
            const float s = m > 0 ? m / 127.0f : 1.0f;
            sc[r] = s;
            for (int e = 0; e < 128; ++e) {
                q[r * 128 + e] = (int8_t) lrintf(bf2f(c[r * 128 + e]) / s);
                c[r * 128 + e] = f2bf((float) q[r * 128 + e] * s);
            }
        }
    }
    return kv;
}
static void q8_free(KtKvQ8 kv) { free(kv.k); free(kv.v); free(kv.ks); free(kv.vs); }

// ---- rope + the 8-bit cache: K rotated and rounded to BF16, V as BF16, each (position, head) quantized with its own
// scale.  GPU sines and divisions may differ from the reference in the last bit: a K value within one BF16 ulp, a scale
// within one ulp of its maximum, and every code within 1 of round(reference / scale).  Other positions untouched. --------
static void test_rope_kv_q8(void) {
    unsigned sd = 31;
    enum { nh = 8, nkv = 2, P = 6, T = 2 };
    float q[T * nh * 128], k[T * nkv * 128], v[T * nkv * 128];
    for (int i = 0; i < T * nh * 128; ++i) q[i] = bfr(frand(&sd));
    for (int i = 0; i < T * nkv * 128; ++i) { k[i] = bfr(frand(&sd) * (1 + i % 5)); v[i] = bfr(frand(&sd) * 0.3); }
    for (int e = 0; e < 128; ++e) v[128 + e] = 0;   // a zero head: scale 1, codes 0
    int8_t K8[P * nkv * 128], V8[P * nkv * 128];
    float Ks[P * nkv], Vs[P * nkv];
    memset(K8, 0x55, sizeof K8);
    memset(V8, 0x55, sizeof V8);
    for (int i = 0; i < P * nkv; ++i) Ks[i] = Vs[i] = -1;
    float inv[64];
    for (int p = 0; p < 64; ++p) inv[p] = (float) pow(1e7, -2.0 * p / 128);
    const RowInfo ri[T] = {{2, {0}}, {5, {0}}};
    const AttnArgs a = {nh, nkv, 0, 1, (float) (1 / sqrt(128.0))};
    const KtKvQ8 kv = {K8, V8, Ks, Vs};
    int bad = 0, badk = 0, untouched = 0;
    if (kt_rope_kv_q8(a, q, k, v, kv, P, ri, inv, T)) { ++fails; return; }
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < nkv; ++h) {
            double x[2][128];   // reference K (rotated, BF16) and V
            const int pos = ri[t].pos;
            for (int i = 0; i < 64; ++i) {
                const double th = (double) (float) ((float) pos * inv[i]), cs = cos(th), sn = sin(th);
                const double k0 = k[(t * nkv + h) * 128 + i], k1 = k[(t * nkv + h) * 128 + i + 64];
                x[0][i] = bfr(k0 * cs - k1 * sn);
                x[0][i + 64] = bfr(k1 * cs + k0 * sn);
            }
            for (int e = 0; e < 128; ++e) x[1][e] = v[(t * nkv + h) * 128 + e];
            for (int which = 0; which < 2; ++which) {
                double m = 0;
                for (int e = 0; e < 128; ++e) m = fmax(m, fabs(x[which][e]));
                const double sref = m > 0 ? m / 127.0 : 1.0;
                const float s = (which ? Vs : Ks)[pos * nkv + h];
                if (fabs(s - sref) > sref * (which ? 1e-6 : 1.0 / 128)) {
                    if (bad < 3) printf("  rope q8 %s t %d h %d: scale %.8g vs %.8g\n", which ? "V" : "K", t, h, s, sref);
                    ++bad;
                }
                for (int e = 0; e < 128; ++e) {
                    const int code = (which ? V8 : K8)[(pos * nkv + h) * 128 + e];
                    // K: the reference may sit one BF16 ulp away from the GPU's rotation
                    const double slack = which ? 0 : fabs(x[0][e]) / 128.0 / s;
                    if (fabs(code - x[which][e] / s) > 0.5 + 1e-4 + slack + (which ? 0 : 0.5)) {
                        if (bad < 3) printf("  rope q8 %s t %d h %d d %d: code %d for %.8g / %.8g\n", which ? "V" : "K", t, h, e, code, x[which][e], s);
                        ++bad;
                        badk += !which;
                    }
                }
            }
        }
    for (int p = 0; p < P; ++p) {
        if (p == ri[0].pos || p == ri[1].pos) continue;
        for (int i = 0; i < nkv * 128; ++i) untouched += K8[p * nkv * 128 + i] != 0x55 || V8[p * nkv * 128 + i] != 0x55;
        for (int h = 0; h < nkv; ++h) untouched += Ks[p * nkv + h] != -1 || Vs[p * nkv + h] != -1;
    }
    CHECK(!bad && !untouched, "rope into the 8-bit cache: %d mismatches (%d in K), %d writes outside the rows' positions", bad, badk,
          untouched);
    printf("rope into the 8-bit cache: %d rows x %d KV heads, %d mismatches\n", T, nkv, bad);
}

// ---- decode attention at a GQA 4 layout: several splits, rows at different positions; q8: on the 8-bit cache ---------
static void test_attn_decode(int q8) {
    unsigned sd = 21;
    const int nh = 8, nkv = 2, NP = 301, T = 3, ns = 5;
    const int pos[3] = {300, 299, 37};
    float* q = malloc(4 * (size_t) T * nh * 128), *g = malloc(4 * (size_t) T * nh * 128), *o = malloc(4 * (size_t) T * nh * 128);
    uint16_t* Kc = malloc(2 * (size_t) NP * nkv * 128), *Vc = malloc(2 * (size_t) NP * nkv * 128);
    for (int i = 0; i < T * nh * 128; ++i) { q[i] = bfr(frand(&sd) * 2); g[i] = bfr(frand(&sd) * 3); }
    for (int i = 0; i < NP * nkv * 128; ++i) { Kc[i] = f2bf((float) (frand(&sd) * 2)); Vc[i] = f2bf((float) frand(&sd)); }
    RowInfo ri[3];
    memset(ri, 0, sizeof ri);
    for (int t = 0; t < T; ++t) ri[t].pos = pos[t];
    const AttnArgs a = {nh, nkv, (NP + ns - 1) / ns, ns, (float) (1 / sqrt(128.0))};
    int bad = 0;
    KtKvQ8 kv = {0};
    if (q8) kv = q8_cache(Kc, Vc, NP, nkv);   // Kc / Vc: the values the kernel reads, for the reference
    if (q8 ? kt_attn_q8(a, q, kv, NP, ri, g, o, T) : kt_attn(a, q, Kc, Vc, NP, ri, g, o, T)) ++fails;
    else {
        double* s = malloc(sizeof(double) * NP);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < nh; ++h) {
                const int kvh = h / (nh / nkv);
                double mx = -1e300, l = 0;
                for (int p = 0; p <= pos[t]; ++p) {
                    double acc = 0;
                    for (int e = 0; e < 128; ++e) acc += (double) q[((size_t) t * nh + h) * 128 + e] * a.scale * bf2f(Kc[((size_t) p * nkv + kvh) * 128 + e]);
                    s[p] = acc;
                    mx = fmax(mx, acc);
                }
                for (int p = 0; p <= pos[t]; ++p) l += exp(s[p] - mx);
                for (int e = 0; e < 128; ++e) {
                    double acc = 0;
                    for (int p = 0; p <= pos[t]; ++p) acc += exp(s[p] - mx) / l * bf2f(Vc[((size_t) p * nkv + kvh) * 128 + e]);
                    const size_t i = ((size_t) t * nh + h) * 128 + e;
                    if (!close_bf(o[i], gated(acc, g[i]), 2e-6)) {
                        if (bad < 3) printf("  attn t %d h %d d %d: %.8g vs %.8g\n", t, h, e, o[i], gated(acc, g[i]));
                        ++bad;
                    }
                }
            }
        free(s);
    }
    CHECK(!bad, "attention decode%s (%d rows, %d splits, GQA %d/%d): %d mismatches of %d", q8 ? ", 8-bit KV" : "", T, ns, nh, nkv,
          bad, T * nh * 128);
    printf("attention decode%s: %d rows x %d heads, %d splits, %d mismatches\n", q8 ? ", 8-bit KV" : "", T, nh, ns, bad);
    if (q8) q8_free(kv);
    free(q); free(g); free(o); free(Kc); free(Vc);
}

// ---- prefill attention: causal, the softplus gate.  MLX's prefill rounding points: q * scale and the probabilities to
// BF16 for the matrix products (f32 sums), the row sum from the unrounded probabilities.  Reference in double with those
// roundings, online over the backend's key tiles; outputs within 1 BF16 ulp, except at most 2 of the (row, head) pairs (a
// probability within f32 noise of a BF16 rounding boundary can flip).  q8: on the 8-bit cache.
static void test_attn_prefill(int q8) {
    unsigned sd = 11;
    const int nh = 8, nkv = 2, P0 = 37, T = 45, NP = P0 + T;   // rows at positions P0 .. P0 + T - 1, keys 0 .. their own
    float* q = malloc(4 * (size_t) T * nh * 128), *g = malloc(4 * (size_t) T * nh * 128), *o = malloc(4 * (size_t) T * nh * 128);
    uint16_t* Kc = malloc(2 * (size_t) NP * nkv * 128), *Vc = malloc(2 * (size_t) NP * nkv * 128);
    for (int i = 0; i < T * nh * 128; ++i) { q[i] = bfr(frand(&sd) * 2); g[i] = bfr(frand(&sd) * 3); }
    for (int i = 0; i < NP * nkv * 128; ++i) { Kc[i] = f2bf((float) (frand(&sd) * 2)); Vc[i] = f2bf((float) frand(&sd)); }
    RowInfo* ri = calloc((size_t) T, sizeof(RowInfo));
    for (int t = 0; t < T; ++t) ri[t].pos = P0 + t;
    const AttnArgs a = {nh, nkv, 0, 1, (float) (1 / sqrt(128.0))};
    const int BK = kt_attn_key_tile();
    int bad = 0, badpairs = 0;
    KtKvQ8 kv = {0};
    if (q8) kv = q8_cache(Kc, Vc, NP, nkv);
    if (q8 ? kt_attn_prefill_q8(a, q, kv, NP, ri, g, o, T) : kt_attn_prefill(a, q, Kc, Vc, NP, ri, g, o, T)) ++fails;
    else {
        double* s = malloc(sizeof(double) * NP);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < nh; ++h) {
                const int bad0 = bad, kvh = h / (nh / nkv), pos = ri[t].pos;
                for (int p = 0; p <= pos; ++p) {
                    double acc = 0;
                    for (int e = 0; e < 128; ++e) acc += (double) bfr(q[((size_t) t * nh + h) * 128 + e] * a.scale) * bf2f(Kc[((size_t) p * nkv + kvh) * 128 + e]);
                    s[p] = acc;
                }
                double O[128] = {0}, mr = -1e300, l = 0;
                for (int k0 = 0; k0 <= pos; k0 += BK) {
                    double tm = -1e300;
                    for (int p = k0; p < k0 + BK && p <= pos; ++p) tm = fmax(tm, s[p]);
                    const double mn = fmax(mr, tm), cor = mr == -1e300 ? 1.0 : exp(mr - mn);
                    l *= cor;
                    for (int e = 0; e < 128; ++e) O[e] *= cor;
                    for (int p = k0; p < k0 + BK && p <= pos; ++p) {
                        const double pv = exp(s[p] - mn);
                        l += pv;
                        for (int e = 0; e < 128; ++e) O[e] += (double) bfr(pv) * bf2f(Vc[((size_t) p * nkv + kvh) * 128 + e]);
                    }
                    mr = mn;
                }
                for (int e = 0; e < 128; ++e) {
                    const size_t i = ((size_t) t * nh + h) * 128 + e;
                    const double ref = gated(O[e] / l, g[i]);
                    if (!close_bf(o[i], ref, 2e-6)) {
                        if (bad < 3) printf("  attn prefill t %d h %d d %d: %.8g vs %.8g\n", t, h, e, o[i], ref);
                        ++bad;
                    }
                }
                badpairs += bad > bad0;
            }
        free(s);
    }
    CHECK(badpairs <= 2, "attention prefill%s (T %d after %d cached, GQA %d/%d, gate): %d (row, head) pairs off (%d outputs of %d)",
          q8 ? ", 8-bit KV" : "", T, P0, nh, nkv, badpairs, bad, T * nh * 128);
    printf("attention prefill%s: %d rows x %d heads, %d (row, head) pairs off, %d outputs\n", q8 ? ", 8-bit KV" : "", T, nh,
           badpairs, bad);
    if (q8) q8_free(kv);
    free(q); free(g); free(o); free(Kc); free(Vc); free(ri);
}

// ---- MLA (TransMLA conversion): per-head maps (+ gate), RoPE + latent cache write, latent attention ------------------
static void test_mla(void) {
    if (!strcmp(kt_name(), "CUDA")) { printf("MLA kernels: Metal only so far, skipped on CUDA\n"); return; }
    unsigned sd = 31;
    {   // per-head maps: q_lat-like (H 4, O 96, I 128, q's [T][H*128] layout) and v_up-like with the gate (O 128, I 96)
        enum { H = 4, T = 3 };
        const int shapes[2][2] = {{96, 128}, {128, 96}};
        for (int c = 0; c < 2; ++c) {
            const int O = shapes[c][0], I = shapes[c][1], gate = c == 1;
            uint16_t* W = malloc(2 * (size_t) H * O * I);
            float* x = malloc(4 * (size_t) T * H * I), *g = malloc(4 * (size_t) T * H * O), *y = malloc(4 * (size_t) T * H * O);
            for (int i = 0; i < H * O * I; ++i) W[i] = f2bf((float) (frand(&sd) * 0.1));
            for (int i = 0; i < T * H * I; ++i) x[i] = bfr(frand(&sd) * 2);
            for (int i = 0; i < T * H * O; ++i) g[i] = bfr(frand(&sd) * 3);
            int bad = 0;
            if (kt_heads_mv(H, O, I, W, x, H * I, I, gate ? g : NULL, y, T)) ++fails;
            else
                for (int t = 0; t < T; ++t)
                    for (int h = 0; h < H; ++h)
                        for (int o = 0; o < O; ++o) {
                            double s = 0;
                            for (int i = 0; i < I; ++i) s += bf2f(W[((size_t) h * O + o) * I + i]) * (double) x[(size_t) t * H * I + h * I + i];
                            const size_t yi = ((size_t) t * H + h) * O + o;
                            const double want = gate ? gated(s, g[yi]) : bfr(s);
                            bad += !close_bf(y[yi], want, 2e-6);
                        }
            CHECK(!bad, "per-head maps (O %d, I %d%s): %d mismatches", O, I, gate ? ", gate" : "", bad);
            free(W); free(x); free(g); free(y);
        }
    }
    {   // RoPE + latent cache write: 4 heads, r 96, rows at positions 3 and 4 of a 5-position cache
        enum { nh = 4, r = 96, P = 5, T = 2 };
        float qr[T * nh * 128], kr[T * 128], c[T * r], q0[T * nh * 128];
        for (int i = 0; i < T * nh * 128; ++i) qr[i] = bfr(frand(&sd));
        for (int i = 0; i < T * 128; ++i) kr[i] = bfr(frand(&sd));
        for (int i = 0; i < T * r; ++i) c[i] = bfr(frand(&sd));
        memcpy(q0, qr, sizeof qr);
        uint16_t Kc[P * 128], Vc[P * r];
        memset(Kc, 0, sizeof Kc);
        memset(Vc, 0, sizeof Vc);
        float inv[64];
        for (int p = 0; p < 64; ++p) inv[p] = (float) pow(1e7, -2.0 * p / 128);
        const RowInfo ri[2] = {{3, {0}}, {4, {0}}};
        const MlaArgs a = {nh, r, 128, 1, (float) (1 / sqrt(128.0)), {0}};
        int bad = 0;
        if (kt_mla_rope(a, qr, kr, c, Kc, Vc, P, ri, inv, T)) ++fails;
        else
            for (int t = 0; t < T; ++t) {
                const int pos = ri[t].pos;
                for (int i = 0; i < 64; ++i) {
                    const double th = (double) (float) ((float) pos * inv[i]), cs = cos(th), sn = sin(th);
                    for (int h = 0; h < nh; ++h) {
                        const double x0 = q0[t * nh * 128 + h * 128 + i], x1 = q0[t * nh * 128 + h * 128 + i + 64];
                        bad += !close_bf(qr[t * nh * 128 + h * 128 + i], bfr(x0 * cs - x1 * sn), 1e-6);
                        bad += !close_bf(qr[t * nh * 128 + h * 128 + i + 64], bfr(x1 * cs + x0 * sn), 1e-6);
                    }
                    const double k0 = kr[t * 128 + i], k1 = kr[t * 128 + i + 64];
                    bad += !close_bf(bf2f(Kc[pos * 128 + i]), bfr(k0 * cs - k1 * sn), 1e-6);
                    bad += !close_bf(bf2f(Kc[pos * 128 + i + 64]), bfr(k1 * cs + k0 * sn), 1e-6);
                }
                for (int e = 0; e < r; ++e) bad += bf2f(Vc[pos * r + e]) != c[t * r + e];
            }
        CHECK(!bad, "MLA rope + latent cache write: %d mismatches", bad);
    }
    for (int split = 0; split < 2; ++split) {   // latent attention: 32 heads, r 768; 1 split or 5
        const int nh = 32, r = 768, NP = 301, T = 3, ns = split ? 5 : 1;
        const int pos[3] = {300, 299, 37};
        float* ql = malloc(4 * (size_t) T * nh * r), *qr = malloc(4 * (size_t) T * nh * 128), *ol = malloc(4 * (size_t) T * nh * r);
        uint16_t* Kc = malloc(2 * (size_t) NP * 128), *Vc = malloc(2 * (size_t) NP * r);
        for (int i = 0; i < T * nh * r; ++i) ql[i] = bfr(frand(&sd) * 0.3);
        for (int i = 0; i < T * nh * 128; ++i) qr[i] = bfr(frand(&sd));
        for (int i = 0; i < NP * 128; ++i) Kc[i] = f2bf((float) frand(&sd));
        for (int i = 0; i < NP * r; ++i) Vc[i] = f2bf((float) frand(&sd));
        RowInfo ri[3];
        memset(ri, 0, sizeof ri);
        for (int t = 0; t < T; ++t) ri[t].pos = pos[t];
        const MlaArgs a = {nh, r, (NP + ns - 1) / ns, ns, (float) (1 / sqrt(128.0)), {0}};
        int bad = 0;
        if (kt_mla_attn(a, ql, qr, Kc, Vc, NP, ri, ol, T)) ++fails;
        else {
            double* s = malloc(sizeof(double) * NP), *acc = malloc(sizeof(double) * r);
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < nh; ++h) {
                    double mx = -1e300, l = 0;
                    for (int p = 0; p <= pos[t]; ++p) {
                        double d = 0;
                        for (int e = 0; e < r; ++e) d += (double) ql[((size_t) t * nh + h) * r + e] * bf2f(Vc[(size_t) p * r + e]);
                        for (int e = 0; e < 128; ++e) d += (double) qr[((size_t) t * nh + h) * 128 + e] * bf2f(Kc[(size_t) p * 128 + e]);
                        s[p] = d * a.scale;
                        mx = fmax(mx, s[p]);
                    }
                    memset(acc, 0, sizeof(double) * r);
                    for (int p = 0; p <= pos[t]; ++p) {
                        const double e = exp(s[p] - mx);
                        l += e;
                        for (int k = 0; k < r; ++k) acc[k] += e * bf2f(Vc[(size_t) p * r + k]);
                    }
                    for (int k = 0; k < r; ++k) {
                        const size_t i = ((size_t) t * nh + h) * r + k;
                        if (!close_bf(ol[i], bfr(acc[k] / l), 2e-6)) {
                            if (bad < 3) printf("  mla attn t %d h %d d %d: %.8g vs %.8g\n", t, h, k, ol[i], bfr(acc[k] / l));
                            ++bad;
                        }
                    }
                }
            free(s); free(acc);
        }
        CHECK(!bad, "MLA latent attention (%d split%s): %d mismatches of %d", ns, ns > 1 ? "s" : "", bad, T * nh * r);
        printf("MLA latent attention: %d rows x %d heads, r %d, %d split(s), %d mismatches\n", T, nh, r, ns, bad);
        free(ql); free(qr); free(ol); free(Kc); free(Vc);
    }
}

int main(void) {
    char err[256] = "";
    if (kt_open(err, sizeof err)) {
        if (strstr(err, "no CUDA device") || strstr(err, "no Metal device")) { printf("SKIP: %s\n", err); return 77; }
        printf("FAIL: %s\n", err);
        return 1;
    }
    printf("kernels: %s\n", kt_name());
    test_mv(256);
    test_mv(512);
    test_misc();
    test_mm();
    test_attn_prefill(0);
    test_attn_decode(0);
    test_rope_kv_q8();
    test_attn_prefill(1);
    test_attn_decode(1);
    test_mla();
    kt_close();
    printf("test_mova_kernels: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
