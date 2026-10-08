// tests/test_search4_gpu.c - the P = 4 GPU search (nslm/search4.metal or nslm/search4.cu) against the scalar C reference
// (nslm4_search_ref): identical seed, coefficients, exponent code and error bits for every block.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "search4.h"
#include "search4_gpu.h"

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

int main(void) {
    {
        char err[512];
        Nslm4Gpu* g = nslm4_gpu_open("out/res/search4.metallib", err, sizeof err);
        if (!g) { printf("FAIL: %s\n", err); return 1; }
        const int rows = 600, cols = 32, ng = cols / 8, NS = 2048, bias = -22;
        unsigned sd = 11;
        float* w = malloc(sizeof(float) * rows * cols), sh[32];
        for (int i = 0; i < rows * cols; ++i) w[i] = (float) (frand(&sd) * 0.02 * (1 + (i % 7 == 0) * 3));
        for (int c = 0; c < cols; ++c) sh[c] = (float) (0.3 + (frand(&sd) + 1));
        Search4Opts o = {NS, 3, {0, -1, 1}, 1};
        const size_t nb = (size_t) rows * ng;
        uint16_t* gs = malloc(2 * nb), *gc = malloc(2 * nb);
        uint8_t* ge = malloc(nb);
        float* gr = malloc(4 * nb);
        if (nslm4_gpu_search(g, w, rows, cols, sh, bias, &o, gs, gc, ge, gr, err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
        float* tab = calloc((size_t) 65536 * NSLM4_ENT, sizeof(float));
        uint8_t* ok = calloc(65536, 1);
        float* wg = malloc(sizeof(float) * rows * 8);
        uint16_t* cs = malloc(2 * rows), *cc = malloc(2 * rows);
        uint8_t* ce = malloc(rows);
        float* cr = malloc(4 * rows);
        long bad = 0, badr = 0;
        for (int gi = 0; gi < ng; ++gi) {
            for (int s = 1; s <= NS; ++s) ok[s] = (uint8_t) nslm4_seed_entry(s, sh + 8 * gi, tab + (size_t) s * NSLM4_ENT);
            for (int r = 0; r < rows; ++r) memcpy(wg + r * 8, w + (size_t) r * cols + gi * 8, 32);
            nslm4_search_ref(tab, ok, wg, rows, sh + 8 * gi, bias, &o, cs, cc, ce, cr);
            for (int r = 0; r < rows; ++r) {
                const size_t k = (size_t) r * ng + gi;
                const int same = gs[k] == cs[r] && gc[k] == cc[r] && ge[k] == ce[r];
                bad += !same;
                badr += memcmp(&gr[k], &cr[r], 4) != 0;
                if (!same && bad < 4) printf("  block r %d g %d: gpu (%u %04x %u) cpu (%u %04x %u)\n", r, gi, gs[k], gc[k], ge[k], cs[r], cc[r], ce[r]);
            }
        }
        printf("search4 GPU vs C (%d x %d, %d seeds): %ld blocks differ, %ld error bits differ, of %zu\n", rows, cols, NS, bad, badr, nb);
        nslm4_gpu_close(g);
        printf(bad || badr ? "FAIL\n" : "PASS\n");
        return bad || badr;
    }
}
