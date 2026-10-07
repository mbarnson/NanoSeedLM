// tests/test_search4.c - P = 4 blocks: decode spec, coefficient packing, and the scalar reference search vs a
// brute-force double search over the same candidates.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "search4.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

// The candidate error in double for (seed, e, q): sum_c (x_c - sc sum_p U_cp q_p)^2 with U = (S R32) sh.
static double exact_err(const float x[8], const float* sh, int s, int e, const double q[4]) {
    uint16_t st[32];
    lfsr_states((uint16_t) s, 32, st);
    double er = 0;
    for (int c = 0; c < 8; ++c) {
        double v = 0;
        for (int p = 0; p < 4; ++p) v += ((double) ((int32_t) st[4 * c + p] - 32768) * NSLM_R32) * sh[c] * q[p];
        const double d = x[c] - ldexp(v, e);
        er += d * d;
    }
    return er;
}

int main(void) {
    // packing
    for (int a = -8; a <= 7; ++a) {
        const int q[4] = {a, -a - 1 < -8 ? 7 : -a - 1, a / 2, 7 - (a + 8) % 16};
        const uint16_t c = nslm4_pack(q);
        int ok = 1;
        for (int p = 0; p < 4; ++p) ok &= nslm4_q(c, p) == q[p];
        CHECK(ok, "pack / unpack q (%d %d %d %d)", q[0], q[1], q[2], q[3]);
    }
    // decode spec against a double recomputation
    {
        unsigned sd = 3;
        int bad = 0;
        for (int it = 0; it < 2000; ++it) {
            const uint16_t s = (uint16_t) (1 + (unsigned) ((frand(&sd) + 1) * 32767));
            int q[4];
            for (int p = 0; p < 4; ++p) q[p] = (int) ((frand(&sd) + 1) * 8) - 8;
            const int e = -20 + it % 9;
            uint16_t out[8], st[32];
            nslm4_decode_block(s, nslm4_pack(q), e, out);
            lfsr_states(s, 32, st);
            for (int c = 0; c < 8; ++c) {
                long is = 0;
                for (int p = 0; p < 4; ++p) is += ((long) st[4 * c + p] - 32768) * q[p];
                bad += out[c] != nslm4_f2bf((float) is * (NSLM_R32 * ldexpf(1.0f, e)));
            }
        }
        CHECK(!bad, "decode spec: %d mismatches", bad);
    }
    // search: reference vs brute force (double) over seeds 1..NS
    {
        enum { NS = 1024, NB = 200 };   // constants: no VLAs (MSVC)
        unsigned sd = 7;
        float sh[8];
        for (int c = 0; c < 8; ++c) sh[c] = (float) (0.5 + (frand(&sd) + 1));
        float* tab = calloc((size_t) 65536 * NSLM4_ENT, sizeof(float));
        uint8_t* ok = calloc(65536, 1);
        int nok = 0;
        for (int s = 1; s <= NS; ++s) nok += ok[s] = (uint8_t) nslm4_seed_entry(s, sh, tab + (size_t) s * NSLM4_ENT);
        CHECK(nok > NS * 9 / 10, "seed entries: %d of %d factor", nok, NS);
        float* w = malloc(sizeof(float) * NB * 8);
        for (int i = 0; i < NB * 8; ++i) w[i] = (float) (frand(&sd) * 0.02);
        const int bias = -22;
        Search4Opts o = {NS, 3, {0, -1, 1}, 0};
        uint16_t seed[NB], coef[NB];
        uint8_t ec[NB];
        float err[NB];
        nslm4_search_ref(tab, ok, w, NB, sh, bias, &o, seed, coef, ec, err);
        int worse = 0;
        double tot_ref = 0, tot_bf = 0;
        for (int k = 0; k < NB; ++k) {
            float x[8];
            double wn = 0;
            for (int c = 0; c < 8; ++c) { x[c] = w[k * 8 + c] * sh[c]; wn += (double) x[c] * x[c]; }
            double bf = wn;   // q = 0
            for (int s = 1; s <= NS; ++s) {
                if (!ok[s]) continue;
                // exact least squares in double, then the same exponent / rounding rule
                uint16_t st[32];
                lfsr_states((uint16_t) s, 32, st);
                double U[8][4], G[4][4] = {{0}}, b[4] = {0};
                for (int c = 0; c < 8; ++c) for (int p = 0; p < 4; ++p) U[c][p] = ((double) ((int32_t) st[4 * c + p] - 32768) * NSLM_R32) * sh[c];
                for (int i = 0; i < 4; ++i) { for (int c = 0; c < 8; ++c) b[i] += U[c][i] * x[c]; for (int j = 0; j < 4; ++j) for (int c = 0; c < 8; ++c) G[i][j] += U[c][i] * U[c][j]; }
                double A[4][5];
                for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) A[i][j] = G[i][j]; A[i][4] = b[i]; }
                for (int i = 0; i < 4; ++i) for (int r = 0; r < 4; ++r) if (r != i) { const double f = A[r][i] / A[i][i]; for (int j = i; j < 5; ++j) A[r][j] -= f * A[i][j]; }
                double t[4], m = 0;
                for (int i = 0; i < 4; ++i) { t[i] = A[i][4] / A[i][i]; m = fmax(m, fabs(t[i])); }
                int e0 = (int) floor(log2(m)) - 2;
                e0 = e0 < bias ? bias : (e0 > bias + 15 ? bias + 15 : e0);
                for (int de = -1; de <= 1; ++de) {
                    int e = e0 + de;
                    e = e < bias ? bias : (e > bias + 15 ? bias + 15 : e);
                    double q[4];
                    for (int p = 0; p < 4; ++p) { const double r = nearbyint(ldexp(t[p], -e)); q[p] = r < -8 ? -8 : (r > 7 ? 7 : r); }
                    const double er = exact_err(x, sh, s, e, q);
                    if (er < bf) bf = er;
                }
            }
            double qr[4];
            for (int p = 0; p < 4; ++p) qr[p] = nslm4_q(coef[k], p);
            const double er = exact_err(x, sh, seed[k], ec[k] + bias, qr);
            worse += er > bf * (1 + 1e-4) + 1e-14 * wn;
            tot_ref += er;
            tot_bf += bf;
        }
        CHECK(worse == 0, "search: %d of %d blocks above the brute-force best", worse, NB);
        printf("search4 (%d seeds, %d blocks): ref err / brute-force err = %.6f\n", NS, NB, tot_ref / tot_bf);
        // refit: never worse than the plain rounding (decoded error)
        o.refit = 1;
        uint16_t seed2[NB], coef2[NB];
        uint8_t ec2[NB];
        float err2[NB];
        nslm4_search_ref(tab, ok, w, NB, sh, bias, &o, seed2, coef2, ec2, err2);
        int rf = 0;
        for (int k = 0; k < NB; ++k) rf += err2[k] > err[k] || seed2[k] != seed[k] || ec2[k] != ec[k];
        CHECK(!rf, "refit: %d blocks worse or with another seed / exponent", rf);
        free(tab); free(ok); free(w);
    }
    // full transform A: diag(sh) reproduces the sh search bit for bit; a lower-triangular A reports |A (w - w')|^2
    {
        enum { NS = 1024, NB = 64 };
        const int bias = -22;
        unsigned sd = 21;
        float* tab = calloc((size_t) 65536 * NSLM4_ENT, sizeof(float)), *taba = calloc((size_t) 65536 * NSLM4_ENT, sizeof(float));
        uint8_t* ok = calloc(65536, 1), *oka = calloc(65536, 1);
        float w[NB * 8], sh[8], A[64] = {0};
        for (int i = 0; i < NB * 8; ++i) w[i] = (float) (frand(&sd) * 0.03);
        for (int c = 0; c < 8; ++c) A[c * 9] = sh[c] = (float) (0.5 + (frand(&sd) + 1));
        Search4Opts o = {NS, 3, {0, -1, 1}, 1};
        for (int s = 1; s <= NS; ++s) {
            ok[s] = (uint8_t) nslm4_seed_entry(s, sh, tab + (size_t) s * NSLM4_ENT);
            oka[s] = (uint8_t) nslm4_seed_entry_a(s, A, taba + (size_t) s * NSLM4_ENT);
        }
        CHECK(!memcmp(tab, taba, sizeof(float) * (size_t) (NS + 1) * NSLM4_ENT) && !memcmp(ok, oka, NS + 1), "diag A: seed table differs");
        uint16_t s1[NB], c1[NB], s2[NB], c2[NB];
        uint8_t e1[NB], e2[NB];
        float r1[NB], r2[NB];
        nslm4_search_ref(tab, ok, w, NB, sh, bias, &o, s1, c1, e1, r1);
        nslm4_search_ref_a(taba, oka, w, NB, A, bias, &o, s2, c2, e2, r2);
        CHECK(!memcmp(s1, s2, sizeof s1) && !memcmp(c1, c2, sizeof c1) && !memcmp(e1, e2, sizeof e1) && !memcmp(r1, r2, sizeof r1),
              "diag A: blocks differ from the sh search");
        for (int i = 0; i < 8; ++i) for (int j = 0; j <= i; ++j) A[i * 8 + j] = i == j ? 1.0f + (float) fabs(frand(&sd)) : (float) (frand(&sd) * 0.5);
        for (int s = 1; s <= NS; ++s) oka[s] = (uint8_t) nslm4_seed_entry_a(s, A, taba + (size_t) s * NSLM4_ENT);
        nslm4_search_ref_a(taba, oka, w, NB, A, bias, &o, s2, c2, e2, r2);
        int bad = 0;
        for (int k = 0; k < NB; ++k) {
            uint16_t bf[8];
            nslm4_decode_block(s2[k], c2[k], e2[k] + bias, bf);
            double er = 0;
            for (int c = 0; c < 8; ++c) {
                double d = 0;
                for (int j = 0; j <= c; ++j) d += (double) A[c * 8 + j] * ((double) w[k * 8 + j] - nslm4_bf2f(bf[j]));
                er += d * d;
            }
            bad += fabs(er - r2[k]) > 1e-4 * er + 1e-12;
        }
        CHECK(!bad, "full A: %d of %d reported errors differ from |A (w - w')|^2", bad, NB);
        free(tab); free(taba); free(ok); free(oka);
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}
