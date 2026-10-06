// tests/test_search.c - the seed search (nslm/search.h).
//   1. synthetic recovery: blocks generated from a known (seed, e, q) decode back exactly (zero error)
//   2. the Clang-vector search returns the identical (seed, nibble) as the scalar reference, on Gaussian blocks of
//      several scales and on heavy-tailed blocks, with a reduced budget (many blocks) and the full budget (fewer)
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "search.h"

static SeedTab g_tab[65536];

static double gauss(void) {
    const double u = (rand() + 1.0) / (RAND_MAX + 2.0), v = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

static int compare(const float* w, int nb, int bias, const SearchOpts* o, const char* what) {
    uint16_t* s1 = malloc(2 * (size_t) nb), *s2 = malloc(2 * (size_t) nb), *n1 = malloc(2 * (size_t) nb), *n2 = malloc(2 * (size_t) nb);
    float* e1 = malloc(4 * (size_t) nb), *e2 = malloc(4 * (size_t) nb);
    nslm_search_ref(g_tab, w, nb, bias, o, s1, n1, e1, NULL);
    nslm_search_vec(g_tab, w, nb, bias, o, s2, n2, e2, NULL);
    int bad = 0;
    for (int k = 0; k < nb; ++k)
        if (s1[k] != s2[k] || n1[k] != n2[k] || e1[k] != e2[k]) {
            if (bad < 5) printf("  %s block %d: ref (%u, %04x) vec (%u, %04x)\n", what, k, s1[k], n1[k], s2[k], n2[k]);
            ++bad;
        }
    printf("%s: %d blocks, %d seeds, %d exponents, refit %d: %s\n", what, nb, o->n_seeds, o->n_exp, o->refit,
           bad ? "MISMATCH" : "identical");
    free(s1); free(s2); free(n1); free(n2); free(e1); free(e2);
    return bad;
}

int main(void) {
    const int bad_seeds = nslm_seedtab_build(g_tab);
    printf("seed table: %d singular seeds\n", bad_seeds);
    assert(bad_seeds == 0);
    srand(12345);

    // 1. synthetic recovery at the full budget (scalar reference and vector)
    {
        const int nb = 48, bias = -20;
        float w[48 * 8];
        uint16_t ts[48], tn[48];
        for (int k = 0; k < nb; ++k) {
            ts[k] = (uint16_t) (1 + rand() % 65535);
            int q[3];
            // max |q| in 5..7, so the paper's exponent is the generating one despite BF16 rounding
            int mq;
            do { for (int p = 0; p < 3; ++p) q[p] = -7 + rand() % 15; mq = abs(q[0]) > abs(q[1]) ? abs(q[0]) : abs(q[1]);
                 mq = abs(q[2]) > mq ? abs(q[2]) : mq; } while (mq < 5);
            tn[k] = nslm_pack(3 + rand() % 10, q);
            uint16_t bf[8];
            nslm_decode_block(ts[k], tn[k], bias, bf);
            for (int c = 0; c < 8; ++c) w[k * 8 + c] = nslm_bf2f(bf[c]);
        }
        SearchOpts o = {65535, 2, {0, -1, 0}, 1};
        uint16_t s[48], n[48];
        float e[48];
        for (int pass = 0; pass < 2; ++pass) {
            (pass ? nslm_search_vec : nslm_search_ref)(g_tab, w, nb, bias, &o, s, n, e, NULL);
            int exact = 0, same = 0;
            for (int k = 0; k < nb; ++k) {
                exact += e[k] == 0.0f;
                same += s[k] == ts[k] && n[k] == tn[k];
                if (e[k] != 0.0f) {
                    const SeedTab* T = &g_tab[ts[k]];
                    const double r = (double) T->R[0] * T->R[3] * T->R[5];
                    printf("  block %d: true seed %u nib %04x (Gi00 %.3g, R diag prod %.3g) -> found seed %u nib %04x err %.3g\n",
                           k, ts[k], tn[k], T->Gi[0], r, s[k], n[k], e[k]);
                }
            }
            printf("synthetic recovery (%s): %d/%d decode exactly, %d/%d recover the generating triple\n",
                   pass ? "vec" : "ref", exact, nb, same, nb);
            assert(exact == nb);
        }
    }
    // 2. reference == vector
    int bad = 0;
    {
        const int nb = 3000;
        float* w = malloc(sizeof(float) * nb * 8);
        for (int k = 0; k < nb; ++k) {
            const double scale = k < 1000 ? 0.02 : k < 2000 ? 0.0007 : 0.05;
            for (int c = 0; c < 8; ++c) {
                double v = gauss() * scale;
                if (k >= 2000 && rand() % 16 == 0) v *= 12;   // heavy tails: outliers
                w[k * 8 + c] = (float) v;
            }
        }
        const int bias = nslm_choose_bias(w, nb, NULL);
        SearchOpts o1 = {4096, 3, {0, -1, 1}, 1}, o2 = {4096, 1, {0, 0, 0}, 0}, o3 = {65535, 3, {0, -1, 1}, 1};
        bad += compare(w, nb, bias, &o1, "random, reduced budget");
        bad += compare(w, nb, bias, &o2, "random, reduced budget, no refinements");
        bad += compare(w, 96, bias, &o3, "random, FULL budget");
        free(w);
    }
    // activation-weighted: scalar == vector with a weighted table; unit weights reproduce the unweighted search exactly
    {
        static SeedTab wt[65536];
        const int nb = 1000;
        float* w = malloc(sizeof(float) * nb * 8), *ws = malloc(sizeof(float) * nb * 8);
        const float sh[8] = {1.0f, 3.0f, 0.2f, 1.0f, 12.0f, 0.5f, 1.0f, 2.0f}, one[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        for (int k = 0; k < nb * 8; ++k) { w[k] = (float) (gauss() * 0.02); ws[k] = w[k] * sh[k % 8]; }
        const int bias = nslm_choose_bias(w, nb, NULL);
        SearchOpts o = {4096, 3, {0, -1, 1}, 1};
        nslm_seedtab_build_weighted(wt, sh, 0, 65536);
        uint16_t* s1 = malloc(2 * nb), *s2 = malloc(2 * nb), *n1 = malloc(2 * nb), *n2 = malloc(2 * nb);
        nslm_search_ref(wt, ws, nb, bias, &o, s1, n1, NULL, sh);
        nslm_search_vec(wt, ws, nb, bias, &o, s2, n2, NULL, sh);
        int b1 = 0, b2 = 0, moved = 0;
        for (int k = 0; k < nb; ++k) b1 += s1[k] != s2[k] || n1[k] != n2[k];
        nslm_seedtab_build_weighted(wt, one, 0, 65536);
        uint16_t* s3 = malloc(2 * nb), *n3 = malloc(2 * nb), *s4 = malloc(2 * nb), *n4 = malloc(2 * nb);
        nslm_search_vec(wt, w, nb, bias, &o, s3, n3, NULL, one);
        nslm_search_vec(g_tab, w, nb, bias, &o, s4, n4, NULL, NULL);
        for (int k = 0; k < nb; ++k) { b2 += s3[k] != s4[k] || n3[k] != n4[k]; moved += s1[k] != s4[k]; }
        printf("weighted: ref == vec %s; unit weights == unweighted %s; %d/%d blocks choose a different seed under the weights\n",
               b1 ? "MISMATCH" : "identical", b2 ? "MISMATCH" : "identical", moved, nb);
        bad += b1 + b2;
    }
    assert(bad == 0);
    printf("test_search: PASS\n");
    return 0;
}
