// tests/test_search_gpu.c - the P = 3 GPU seed search (nslm/search.metal or nslm/search.cu, through search_gpu.h)
// against the scalar C reference (nslm_search_ref): identical seed, nibble word and error bits for every block, over
// all 65535 seeds, unweighted and with extreme sqrt(h) (one channel ~2700x the others: near-singular tables), with
// the lower-bound prune and without it.
//
//   test_search_gpu [out/res/search.metallib]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"
#include "search.h"
#include "search_gpu.h"

static SeedTab g_tab[65536];

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

int main(int argc, char** argv) {
    char err[512];
    NslmGpu* g = nslm_gpu_open(argc > 1 ? argv[1] : "out/res/search.metallib", err, sizeof err);
    if (!g && (strstr(err, "no CUDA device") || strstr(err, "no Metal device"))) { printf("SKIP: %s\n", err); return 77; }   // a machine without a GPU
    if (!g) { printf("FAIL: %s\n", err); return 1; }
    const float shs[3][8] = {{1, 1, 1, 1, 1, 1, 1, 1},
                             {0.1012f, 0.07429f, 0.06921f, 0.06821f, 0.08077f, 0.06578f, 185.2f, 0.06626f},
                             {0.9f, 3.7f, 0.02f, 1.1f, 12.5f, 0.6f, 0.33f, 2.2f}};
    const char* names[3] = {"unweighted", "extreme sqrt(h) (L3 down cg216)", "mixed sqrt(h)"};
    const int rows = 96, bias = -12;
    const SearchOpts o = {65535, 3, {0, -1, 1}, 1};
    float* w = (float*) malloc(sizeof(float) * rows * 8), *xs = (float*) malloc(sizeof(float) * rows * 8);
    uint16_t gs[96], gn[96], cs[96], cn[96];
    float ge[96], ce[96];
    unsigned sd = 5;
    int fail = 0;
    for (int i = 0; i < rows * 8; ++i) w[i] = (float) (frand(&sd) * 0.02 * (1 + (i % 11 == 0) * 4));
    for (int k = 0; k < 3; ++k) {
        const int unweighted = k == 0;
        if (unweighted) nslm_seedtab_build(g_tab);
        else nslm_seedtab_build_weighted(g_tab, shs[k], 1, 65536);
        for (int i = 0; i < rows * 8; ++i) xs[i] = unweighted ? w[i] : w[i] * shs[k][i % 8];
        nslm_search_ref(g_tab, xs, rows, bias, &o, cs, cn, ce, unweighted ? NULL : shs[k]);
        for (int prune = 0; prune <= 1; ++prune) {
            if (nslm_gpu_search(g, w, rows, 8, unweighted ? NULL : shs[k], bias, &o, prune, gs, gn, ge, err, sizeof err)) {
                printf("FAIL: %s\n", err);
                return 1;
            }
            int bad = 0, bade = 0;
            for (int r = 0; r < rows; ++r) {
                bad += gs[r] != cs[r] || gn[r] != cn[r];
                bade += memcmp(&ge[r], &ce[r], 4) != 0;
                if ((gs[r] != cs[r] || gn[r] != cn[r]) && bad < 4)
                    printf("  %s r %d: gpu (%u %04x) cpu (%u %04x)\n", names[k], r, gs[r], gn[r], cs[r], cn[r]);
            }
            printf("%-32s prune %d: %d of %d blocks differ, %d error bits differ\n", names[k], prune, bad, rows, bade);
            fail += bad || bade;
        }
    }
    nslm_gpu_close(g);
    free(w); free(xs);
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail != 0;
}
