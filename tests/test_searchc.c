// tests/test_searchc.c - SeedLM blocks of C weights (nslm/searchc.h): at C = 8 the search equals nslm/searchp.h's
// reference bit for bit; at C = 12 and 16 the decode against the double spec, the reported error against the decoded
// block, and no fit worse than zero coefficients.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "searchc.h"
#include "searchp.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

int main(void) {
    enum { NS = 600, NB = 64 };
    const Search4Opts o = {NS, 3, {0, -1, 1}, 1};
    const int bias = -22;
    unsigned sd = 9;
    CHECK(nslmc_bits(8, 3) == 4.0 && nslmc_bits(8, 4) == 4.5 && nslmc_bits(12, 4) == 3.0 && nslmc_bits(16, 7) == 3.0, "bits");
    {   // C = 8, P = 3: the searchp reference, the same seeds, coefficients, exponents and errors
        const int P = 3, ENT = NSLMP_ENT(P);
        float* tab = calloc((size_t) 65536 * ENT, sizeof(float));
        uint8_t* ok = calloc(65536, 1);
        float w[NB * 8], sh[8], shb[NB * 8];
        for (int i = 0; i < NB * 8; ++i) w[i] = (float) (frand(&sd) * 0.03);
        for (int c = 0; c < 8; ++c) sh[c] = (float) (0.5 + (frand(&sd) + 1));
        for (int k = 0; k < NB; ++k) memcpy(shb + 8 * k, sh, sizeof sh);
        for (int s = 1; s <= NS; ++s) ok[s] = (uint8_t) nslmp_seed_entry(P, s, sh, NULL, tab + (size_t) s * ENT);
        uint16_t s1[NB], s2[NB];
        uint32_t c1[NB], c2[NB];
        uint8_t e1[NB], e2[NB];
        float r1[NB], r2[NB];
        nslmp_search_ref(P, tab, ok, w, NB, sh, NULL, bias, &o, s1, c1, e1, r1);
        nslmc_search_ref(8, P, w, shb, NB, bias, &o, s2, c2, e2, r2);
        int bad = 0;
        for (int k = 0; k < NB; ++k) bad += s1[k] != s2[k] || c1[k] != c2[k] || e1[k] != e2[k] || r1[k] != r2[k];
        CHECK(bad == 0, "C = 8, P = 3: %d of %d blocks differ from nslmp_search_ref", bad, NB);
        free(tab); free(ok);
    }
    const int layouts[2][2] = {{12, 4}, {16, 7}};
    for (int li = 0; li < 2; ++li) {
        const int C = layouts[li][0], P = layouts[li][1];
        int bad = 0;
        for (int i = 0; i < 500; ++i) {   // decode against the double spec
            const uint16_t s = (uint16_t) (1 + (unsigned) ((frand(&sd) + 1) * 32767));
            int q[8];
            for (int p = 0; p < P; ++p) q[p] = (int) ((frand(&sd) + 1) * 8) - 8;
            const int e = -20 + i % 6;
            uint16_t out[NSLMC_MAXC], st[NSLMC_MAXC * NSLMC_MAXP];
            nslmc_decode_block(C, P, s, nslmp_pack(P, q), e, out);
            lfsr_states(s, C * P, st);
            for (int c = 0; c < C; ++c) {
                double v = 0;
                for (int p = 0; p < P; ++p) v += ((double) st[P * c + p] - 32768) * q[p];
                bad += out[c] != nslm4_f2bf((float) (ldexp(v, e) * NSLM_R32));
            }
        }
        CHECK(bad == 0, "C = %d, P = %d decode: %d weights differ from the double spec", C, P, bad);
        float w[NB * NSLMC_MAXC], sh[NB * NSLMC_MAXC];
        for (int i = 0; i < NB * C; ++i) { w[i] = (float) (frand(&sd) * 0.03); sh[i] = (float) (0.5 + (frand(&sd) + 1)); }
        uint16_t seed[NB];
        uint32_t coef[NB];
        uint8_t ec[NB];
        float err[NB];
        nslmc_search_ref(C, P, w, sh, NB, bias, &o, seed, coef, ec, err);
        int wrong = 0, worse = 0;
        for (int k = 0; k < NB; ++k) {   // the reported error is the decoded block's; no fit is worse than zero weights
            uint16_t out[NSLMC_MAXC];
            nslmc_decode_block(C, P, seed[k], coef[k], bias + ec[k], out);
            float de = 0.0f, wn = 0.0f;
            for (int c = 0; c < C; ++c) {
                const float x = w[k * C + c] * sh[k * C + c], d = x - sh[k * C + c] * nslm4_bf2f(out[c]);
                de = de + d * d;
                wn = wn + x * x;
            }
            wrong += de != err[k];
            worse += err[k] > wn;
        }
        CHECK(wrong == 0 && worse == 0, "C = %d, P = %d: %d errors differ from the decoded block, %d fits worse than zero", C, P,
              wrong, worse);
    }
    printf("test_searchc: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
