// tests/test_cpu_seed4.c - the CPU seed mat-vec (nslm/cpu_seed4.h) against the scalar P = 4 spec (lfsr.h, search4.h):
// y_r = sum_b sum_c x_c * isum_c * R32 * 2^e, the decoded block before BF16 rounding, accumulated in double.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cpu_seed4.h"
#include "lfsr.h"
#include "search4.h"

static uint32_t rng = 12345;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static double ref_dot(uint16_t seed, uint16_t coef, int e, const float* x) {
    uint16_t st[32];
    lfsr_states(seed, 32, st);
    double d = 0.0;
    for (int c = 0; c < 8; ++c) {
        int32_t isum = 0;
        for (int p = 0; p < 4; ++p) isum += ((int32_t) st[4 * c + p] - 32768) * nslm4_q(coef, p);
        d += (double) x[c] * (double) isum;
    }
    return d * (double) nslm_scale(e);
}

static int check(int rows, int kb, int bias, int threads) {
    const int n = rows * kb;
    uint16_t* seeds = malloc(n * 2);
    uint16_t* coefs = malloc(n * 2);
    uint8_t* codes = malloc((n + 1) / 2);
    float* x = malloc(kb * 8 * 4);
    float* y = malloc(rows * 4);
    for (int i = 0; i < n; ++i) { seeds[i] = (uint16_t) (1 + rnd() % 65535); coefs[i] = (uint16_t) rnd(); }
    for (int i = 0; i < (n + 1) / 2; ++i) codes[i] = (uint8_t) rnd();
    for (int i = 0; i < kb * 8; ++i) x[i] = (float) ((int) (rnd() % 20001) - 10000) / 10000.0f;
    uint32_t* table = cpu_seed4_table();
    cpu_seed4_matvec_mt(seeds, coefs, codes, bias, table, x, rows, kb, y, threads);
    // the CPU-tiled layout gives the same rows
    float tdiff = 0.0f;
    if (kb % 2 == 0) {
        CpuSeed4Tiled tm;
        cpu_seed4_tile(&tm, seeds, coefs, codes, bias, rows, kb);
        float* yt = malloc(rows * 4);
        cpu_seed4_tiled_matvec_mt(&tm, table, x, yt, threads);
        for (int r = 0; r < rows; ++r) tdiff = fmaxf(tdiff, fabsf(yt[r] - y[r]));
        cpu_seed4_tiled_free(&tm);
        free(yt);
    }
    double worst = 0.0, scale = 0.0;
    double* ref = malloc(rows * sizeof(double));
    for (int r = 0; r < rows; ++r) {
        double d = 0.0;
        for (int b = 0; b < kb; ++b) {
            const int g = r * kb + b;
            const int code = (codes[g >> 1] >> ((g & 1) * 4)) & 15;
            d += ref_dot(seeds[g], coefs[g], bias + code, x + 8 * b);
        }
        ref[r] = d;
        scale = fmax(scale, fabs(d));
    }
    for (int r = 0; r < rows; ++r) worst = fmax(worst, fabs((double) y[r] - ref[r]));
    const double rel = worst / scale;
    const int ok = rel < 2e-6 && tdiff == 0.0f;   // tiled: the same arithmetic in the same order
    printf("rows %d kb %d bias %d threads %d: max |y - ref| / max |ref| = %.2e, tiled max diff %g %s\n", rows, kb,
           bias, threads, rel, tdiff, ok ? "ok" : "FAIL");
    free(seeds); free(coefs); free(codes); free(x); free(y); free(ref);
    return ok;
}

int main(void) {
    uint32_t* t = cpu_seed4_table();
    for (int s = 1; s < 65536; s += 997)
        if (t[s] != lfsr_stream32((uint16_t) s)) { printf("table FAIL at %d\n", s); return 1; }
    int ok = 1;
    ok &= check(16, 4, -20, 1);
    ok &= check(37, 9, -18, 1);      // rows not a multiple of the vector width
    ok &= check(768, 320, -22, 1);   // an expert's gate / up (768 x 2560)
    ok &= check(2560, 96, -22, 4);   // its down (2560 x 768), threaded
    ok &= check(1000, 33, -21, 7);
    printf("test_cpu_seed4: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
