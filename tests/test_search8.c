// tests/test_search8.c - P = 8 blocks: decode spec, coefficient packing, the scalar reference search vs a brute-force
// double search over the same candidates, and the transform path (diag(sh) = the sh search; |A (w - w')|^2).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "search8.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

static double exact_err(const float x[8], const float* sh, int s, int e, const int q[8]) {
    uint16_t st[64];
    lfsr_states((uint16_t) s, 64, st);
    double er = 0;
    for (int c = 0; c < 8; ++c) {
        double v = 0;
        for (int p = 0; p < 8; ++p) v += ((double) ((int32_t) st[8 * c + p] - 32768) * NSLM_R32) * sh[c] * q[p];
        const double d = x[c] - ldexp(v, e);
        er += d * d;
    }
    return er;
}

int main(void) {
    for (int a = -8; a <= 7; ++a) {   // packing
        int q[8], ok = 1;
        for (int p = 0; p < 8; ++p) q[p] = ((a + 3 * p + 8) % 16) - 8;
        const uint32_t c = nslm8_pack(q);
        for (int p = 0; p < 8; ++p) ok &= nslm8_q(c, p) == q[p];
        CHECK(ok, "pack / unpack (a %d)", a);
    }
    {   // decode spec against double
        unsigned sd = 3;
        int bad = 0;
        for (int i = 0; i < 2000; ++i) {
            const uint16_t s = (uint16_t) (1 + (unsigned) ((frand(&sd) + 1) * 32767));
            int q[8];
            for (int p = 0; p < 8; ++p) q[p] = (int) ((frand(&sd) + 1) * 8) - 8;
            const int e = -20 + i % 6;
            uint16_t out[8], st[64];
            nslm8_decode_block(s, nslm8_pack(q), e, out);
            lfsr_states(s, 64, st);
            for (int c = 0; c < 8; ++c) {
                double v = 0;
                for (int p = 0; p < 8; ++p) v += ((double) st[8 * c + p] - 32768) * q[p];
                bad += out[c] != nslm4_f2bf((float) (ldexp(v, e) * NSLM_R32 / 1.0));
            }
        }
        CHECK(!bad, "decode: %d weights differ from the double spec", bad);
    }
    enum { NS = 512, NB = 120 };
    const int bias = -22;
    unsigned sd = 5;
    float* tab = calloc((size_t) 65536 * NSLM8_ENT, sizeof(float)), *taba = calloc((size_t) 65536 * NSLM8_ENT, sizeof(float));
    uint8_t* ok = calloc(65536, 1), *oka = calloc(65536, 1);
    float w[NB * 8], sh[8], A[64] = {0}, x[8];
    for (int i = 0; i < NB * 8; ++i) w[i] = (float) (frand(&sd) * 0.03);
    for (int c = 0; c < 8; ++c) A[c * 9] = sh[c] = (float) (0.5 + (frand(&sd) + 1));
    Search4Opts o = {NS, 3, {0, -1, 1}, 0};
    for (int s = 1; s <= NS; ++s) {
        ok[s] = (uint8_t) nslm8_seed_entry(s, sh, NULL, tab + (size_t) s * NSLM8_ENT);
        oka[s] = (uint8_t) nslm8_seed_entry(s, NULL, A, taba + (size_t) s * NSLM8_ENT);
    }
    {   // the search vs brute force in double over the same candidates (no refit)
        uint16_t seed[NB];
        uint32_t coef[NB];
        uint8_t ec[NB];
        float err[NB];
        nslm8_search_ref(tab, ok, w, NB, sh, NULL, bias, &o, seed, coef, ec, err);
        int worse = 0;
        double tr = 0, tb = 0;
        for (int k = 0; k < NB; ++k) {
            for (int c = 0; c < 8; ++c) x[c] = w[k * 8 + c] * sh[c];
            double bf = INFINITY;
            for (int s = 1; s <= NS; ++s) {
                if (!ok[s]) continue;
                // least squares in double, then the same exponent / rounding rule
                uint16_t st[64];
                lfsr_states((uint16_t) s, 64, st);
                double U[64], G[64], b[8], t[8], Lm[64] = {0};
                for (int c = 0; c < 8; ++c) for (int p = 0; p < 8; ++p) U[c * 8 + p] = ((double) ((int32_t) st[8 * c + p] - 32768) * NSLM_R32) * sh[c];
                for (int i = 0; i < 8; ++i) { b[i] = 0; for (int c = 0; c < 8; ++c) b[i] += U[c * 8 + i] * x[c]; }
                for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) { G[i * 8 + j] = 0; for (int c = 0; c < 8; ++c) G[i * 8 + j] += U[c * 8 + i] * U[c * 8 + j]; }
                int pd = 1;
                for (int i = 0; i < 8 && pd; ++i)
                    for (int j = 0; j <= i; ++j) {
                        double v = G[i * 8 + j];
                        for (int m = 0; m < j; ++m) v -= Lm[i * 8 + m] * Lm[j * 8 + m];
                        if (i == j) { if (!(v > 0)) { pd = 0; break; } Lm[i * 8 + i] = sqrt(v); } else Lm[i * 8 + j] = v / Lm[j * 8 + j];
                    }
                if (!pd) continue;
                double y[8];
                for (int i = 0; i < 8; ++i) { double v = b[i]; for (int m = 0; m < i; ++m) v -= Lm[i * 8 + m] * y[m]; y[i] = v / Lm[i * 8 + i]; }
                for (int i = 7; i >= 0; --i) { double v = y[i]; for (int m = i + 1; m < 8; ++m) v -= Lm[m * 8 + i] * t[m]; t[i] = v / Lm[i * 8 + i]; }
                double mm = 0;
                for (int p = 0; p < 8; ++p) mm = fabs(t[p]) > mm ? fabs(t[p]) : mm;
                const int e0 = (int) floor(log2(mm)) - 2;
                for (int d = 0; d < 3; ++d) {
                    int e = e0 + (d == 0 ? 0 : d == 1 ? -1 : 1);
                    e = e < bias ? bias : (e > bias + 15 ? bias + 15 : e);
                    int q[8];
                    for (int p = 0; p < 8; ++p) { const double r = nearbyint(ldexp(t[p], -e)); q[p] = r < -8 ? -8 : (r > 7 ? 7 : (int) r); }
                    const double er = exact_err(x, sh, s, e, q);
                    if (er < bf) bf = er;
                }
            }
            int q[8];
            for (int p = 0; p < 8; ++p) q[p] = nslm8_q(coef[k], p);
            const double er = exact_err(x, sh, seed[k], ec[k] + bias, q);
            double wn = 0;
            for (int c = 0; c < 8; ++c) wn += (double) x[c] * x[c];
            worse += er > bf * (1 + 1e-3) + 1e-12 * wn;
            tr += er; tb += bf;
        }
        CHECK(worse <= NB / 50, "search: %d of %d blocks above the brute-force best", worse, NB);
        printf("search8 (%d seeds, %d blocks): ref err / brute-force err = %.6f\n", NS, NB, tr / tb);
    }
    {   // diag A = sh, bit for bit; refit never worse
        o.refit = 1;
        CHECK(!memcmp(tab, taba, sizeof(float) * (size_t) (NS + 1) * NSLM8_ENT) && !memcmp(ok, oka, NS + 1), "diag A: seed table differs");
        uint16_t s1[NB], s2[NB];
        uint32_t c1[NB], c2[NB];
        uint8_t e1[NB], e2[NB];
        float r1[NB], r2[NB];
        nslm8_search_ref(tab, ok, w, NB, sh, NULL, bias, &o, s1, c1, e1, r1);
        nslm8_search_ref(taba, oka, w, NB, NULL, A, bias, &o, s2, c2, e2, r2);
        CHECK(!memcmp(s1, s2, sizeof s1) && !memcmp(c1, c2, sizeof c1) && !memcmp(e1, e2, sizeof e1) && !memcmp(r1, r2, sizeof r1),
              "diag A: blocks differ from the sh search");
        for (int i = 0; i < 8; ++i) for (int j = 0; j <= i; ++j) A[i * 8 + j] = i == j ? 1.0f + (float) fabs(frand(&sd)) : (float) (frand(&sd) * 0.5);
        for (int s = 1; s <= NS; ++s) oka[s] = (uint8_t) nslm8_seed_entry(s, NULL, A, taba + (size_t) s * NSLM8_ENT);
        nslm8_search_ref(taba, oka, w, NB, NULL, A, bias, &o, s2, c2, e2, r2);
        int bad = 0;
        for (int k = 0; k < NB; ++k) {
            uint16_t bf[8];
            nslm8_decode_block(s2[k], c2[k], e2[k] + bias, bf);
            double er = 0;
            for (int c = 0; c < 8; ++c) {
                double d = 0;
                for (int j = 0; j <= c; ++j) d += (double) A[c * 8 + j] * ((double) w[k * 8 + j] - nslm4_bf2f(bf[j]));
                er += d * d;
            }
            bad += fabs(er - r2[k]) > 1e-4 * er + 1e-12;
        }
        CHECK(!bad, "full A: %d of %d reported errors differ from |A (w - w')|^2", bad, NB);
    }
    free(tab); free(taba); free(ok); free(oka);
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}
