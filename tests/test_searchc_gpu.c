// tests/test_searchc_gpu.c - the any-C seed search on CUDA (nslm/searchc.cu) against the scalar C reference
// (nslmc_search_ref) at (C, P) = (12, 4), (16, 7), (8, 3), with a padded last block: identical seed, coefficients,
// exponent code and error bits.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "searchc_gpu.h"

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

int main(void) {
    char err[512];
    NslmCGpu* g = nslmc_gpu_open(err, sizeof err);
    if (!g && strstr(err, "no CUDA device")) { printf("SKIP: %s\n", err); return 77; }
    if (!g) { printf("FAIL: %s\n", err); return 1; }
    const int rows = 600, cols = 40, NS = 512, bias = -22;   // 40: C = 12 and 16 pad the last block
    static const int lay[3][2] = {{12, 4}, {16, 7}, {8, 3}};
    unsigned sd = 13;
    float* w = malloc(sizeof(float) * rows * cols), sh[40];
    for (int i = 0; i < rows * cols; ++i) w[i] = (float) (frand(&sd) * 0.02 * (1 + (i % 7 == 0) * 3));
    for (int c = 0; c < cols; ++c) sh[c] = (float) (0.3 + (frand(&sd) + 1));
    const Search4Opts o = {NS, 3, {0, -1, 1}, 1};
    long bad = 0, badr = 0;
    for (int li = 0; li < 3; ++li) {
        const int C = lay[li][0], P = lay[li][1], ng = (cols + C - 1) / C;
        const size_t nb = (size_t) rows * ng;
        uint16_t* gs = malloc(2 * nb), *cs = malloc(2 * nb);
        uint32_t* gc = malloc(4 * nb), *cc = malloc(4 * nb);
        uint8_t* ge = malloc(nb), *ce = malloc(nb);
        float* gr = malloc(4 * nb), *cr = malloc(4 * nb), *wb = calloc(nb * C, sizeof(float)), *sb = calloc(nb * C, sizeof(float));
        for (int r = 0; r < rows; ++r)   // block-contiguous, zero-padded copies for the reference
            for (int c = 0; c < cols; ++c) {
                wb[((size_t) r * ng + c / C) * C + c % C] = w[(size_t) r * cols + c];
                sb[((size_t) r * ng + c / C) * C + c % C] = sh[c];
            }
        if (nslmc_gpu_search(g, C, P, w, rows, cols, sh, bias, &o, gs, gc, ge, gr, err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
        nslmc_search_ref(C, P, wb, sb, (int) nb, bias, &o, cs, cc, ce, cr);
        long b = 0, br = 0;
        for (size_t k = 0; k < nb; ++k) {
            const int same = gs[k] == cs[k] && gc[k] == cc[k] && ge[k] == ce[k];
            b += !same;
            br += memcmp(&gr[k], &cr[k], 4) != 0;
            if (!same && b < 4) printf("  C %d P %d block %zu: gpu (%u %08x %u) cpu (%u %08x %u)\n", C, P, k, gs[k], gc[k], ge[k], cs[k], cc[k], ce[k]);
        }
        printf("searchc C = %d P = %d GPU vs C (%d x %d, %d seeds): %ld blocks differ, %ld error bits differ, of %zu\n", C, P, rows, cols, NS, b, br, nb);
        bad += b; badr += br;
        free(gs); free(cs); free(gc); free(cc); free(ge); free(ce); free(gr); free(cr); free(wb); free(sb);
    }
    nslmc_gpu_close(g);
    free(w);
    printf(bad || badr ? "FAIL\n" : "PASS\n");
    return bad || badr;
}
