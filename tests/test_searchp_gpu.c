// tests/test_searchp_gpu.c - the P = 3 / 8 Metal search (nslm/searchp.metal) against the scalar C reference
// (nslmp_search_ref), sqrt(h) and full-transform modes: identical seed, coefficients, exponent code and error bits.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "searchp.h"
#include "searchp_gpu.h"

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

int main(void) {
    {
        char err[512];
        NslmPGpu* g = nslmp_gpu_open("out/res/searchp.metallib", err, sizeof err);
        if (!g) { printf("FAIL: %s\n", err); return 1; }
        const int rows = 600, cols = 24, ng = cols / 8, NS = 512, bias = -22;
        unsigned sd = 13;
        float* w = malloc(sizeof(float) * rows * cols), sh[24], *A = calloc((size_t) ng * 64, sizeof(float));
        for (int i = 0; i < rows * cols; ++i) w[i] = (float) (frand(&sd) * 0.02 * (1 + (i % 7 == 0) * 3));
        for (int c = 0; c < cols; ++c) sh[c] = (float) (0.3 + (frand(&sd) + 1));
        for (int gi = 0; gi < ng; ++gi)
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j <= i; ++j) A[gi * 64 + i * 8 + j] = i == j ? (float) (1.0 + fabs(frand(&sd))) : (float) (frand(&sd) * 0.5);
        Search4Opts o = {NS, 3, {0, -1, 1}, 1};
        const size_t nb = (size_t) rows * ng;
        uint16_t* gs = malloc(2 * nb);
        uint32_t* gc = malloc(4 * nb);
        uint8_t* ge = malloc(nb);
        float* gr = malloc(4 * nb), *tab = calloc((size_t) 65536 * NSLMP_ENT(8), sizeof(float)), *wg = malloc(sizeof(float) * rows * 8);
        uint8_t* ok = calloc(65536, 1), *ce = malloc(rows);
        uint16_t* cs = malloc(2 * rows);
        uint32_t* cc = malloc(4 * rows);
        float* cr = malloc(4 * rows);
        long bad = 0, badr = 0;
        for (int P = 3; P <= 8; P += 5)
        for (int full = 0; full <= 1; ++full) {
            if (nslmp_gpu_search(g, P, w, rows, cols, full ? NULL : sh, full ? A : NULL, bias, &o, gs, gc, ge, gr, err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
            long b = 0, br = 0;
            for (int gi = 0; gi < ng; ++gi) {
                for (int s = 1; s <= NS; ++s)
                    ok[s] = (uint8_t) nslmp_seed_entry(P, s, full ? NULL : sh + 8 * gi, full ? A + 64 * gi : NULL, tab + (size_t) s * NSLMP_ENT(P));
                for (int r = 0; r < rows; ++r) memcpy(wg + r * 8, w + (size_t) r * cols + gi * 8, 32);
                nslmp_search_ref(P, tab, ok, wg, rows, full ? NULL : sh + 8 * gi, full ? A + 64 * gi : NULL, bias, &o, cs, cc, ce, cr);
                for (int r = 0; r < rows; ++r) {
                    const size_t k = (size_t) r * ng + gi;
                    const int same = gs[k] == cs[r] && gc[k] == cc[r] && ge[k] == ce[r];
                    b += !same;
                    br += memcmp(&gr[k], &cr[r], 4) != 0;
                    if (!same && b < 4) printf("  %s r %d g %d: gpu (%u %08x %u) cpu (%u %08x %u)\n", full ? "A" : "sh", r, gi, gs[k], gc[k], ge[k], cs[r], cc[r], ce[r]);
                }
            }
            printf("searchp P = %d %s GPU vs C (%d x %d, %d seeds): %ld blocks differ, %ld error bits differ, of %zu\n", P, full ? "full A" : "sh", rows, cols, NS, b, br, nb);
            bad += b; badr += br;
        }
        nslmp_gpu_close(g);
        printf(bad || badr ? "FAIL\n" : "PASS\n");
        return bad || badr;
    }
}
