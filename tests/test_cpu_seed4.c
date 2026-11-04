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

// ntok tokens at once (tiled): each token's rows equal a one-token call, bit for bit.
static int check_tokens(int rows, int kb, int ntok, int threads) {
    const int n = rows * kb;
    uint16_t* seeds = malloc(n * 2), *coefs = malloc(n * 2);
    uint8_t* codes = malloc((n + 1) / 2);
    float* x = malloc((size_t) ntok * kb * 8 * 4), *y = malloc((size_t) ntok * rows * 4), *y1 = malloc(rows * 4);
    for (int i = 0; i < n; ++i) { seeds[i] = (uint16_t) (1 + rnd() % 65535); coefs[i] = (uint16_t) rnd(); }
    for (int i = 0; i < (n + 1) / 2; ++i) codes[i] = (uint8_t) rnd();
    for (int i = 0; i < ntok * kb * 8; ++i) x[i] = (float) ((int) (rnd() % 20001) - 10000) / 10000.0f;
    CpuSeed4Tiled tm;
    cpu_seed4_tile(&tm, seeds, coefs, codes, -21, rows, kb);
    const uint32_t* table = cpu_seed4_table();
    cpu_seed4_tiled_matmul_mt(&tm, table, x, ntok, y, threads);
    int same = 1;
    for (int t = 0; t < ntok; ++t) {
        cpu_seed4_tiled_matvec_mt(&tm, table, x + (size_t) t * kb * 8, y1, 1);
        for (int r = 0; r < rows; ++r) same &= y[(size_t) t * rows + r] == y1[r];
    }
    printf("rows %d kb %d tokens %d threads %d: %s\n", rows, kb, ntok, threads, same ? "ok" : "FAIL");
    cpu_seed4_tiled_free(&tm);
    free(seeds); free(coefs); free(codes); free(x); free(y); free(y1);
    return same;
}

// P = 3 (SEED4 nibble words) against the scalar spec (lfsr.h), and ntok tokens against one-token calls.
static int check3(int rows, int kb, int bias, int ntok, int threads) {
    const int n = rows * kb;
    uint16_t* seeds = malloc(n * 2), *nibs = malloc(n * 2);
    float* x = malloc((size_t) ntok * kb * 8 * 4), *y = malloc((size_t) ntok * rows * 4), *y1 = malloc(rows * 4);
    for (int i = 0; i < n; ++i) { seeds[i] = (uint16_t) (1 + rnd() % 65535); nibs[i] = (uint16_t) rnd(); }
    for (int i = 0; i < ntok * kb * 8; ++i) x[i] = (float) ((int) (rnd() % 20001) - 10000) / 10000.0f;
    CpuSeed4Tiled tm;
    cpu_seed3_tile(&tm, seeds, nibs, bias, rows, kb);
    const uint32_t* table = cpu_seed4_table();
    cpu_seed4_tiled_matmul_mt(&tm, table, x, ntok, y, threads);
    double worst = 0.0, scale = 0.0;
    int same = 1;
    for (int t = 0; t < ntok; ++t) {
        cpu_seed4_tiled_matvec_mt(&tm, table, x + (size_t) t * kb * 8, y1, 1);
        for (int r = 0; r < rows; ++r) {
            same &= y[(size_t) t * rows + r] == y1[r];
            double d = 0.0;
            for (int b = 0; b < kb; ++b) {
                const uint16_t nb = nibs[r * kb + b];
                uint16_t st[24];
                lfsr_states(seeds[r * kb + b], 24, st);
                double e = 0.0;
                for (int c = 0; c < 8; ++c) {
                    int32_t isum = 0;
                    for (int p = 0; p < 3; ++p) isum += ((int32_t) st[3 * c + p] - 32768) * nslm_q(nb, p);
                    e += (double) x[(size_t) t * kb * 8 + 8 * b + c] * (double) isum;
                }
                d += e * (double) nslm_scale(bias + nslm_ecode(nb));
            }
            worst = fmax(worst, fabs((double) y[(size_t) t * rows + r] - d));
            scale = fmax(scale, fabs(d));
        }
    }
    const int ok = same && worst / scale < 2e-6;
    printf("P = 3 rows %d kb %d tokens %d threads %d: max |y - ref| / max |ref| = %.2e, tokens %s %s\n", rows, kb, ntok,
           threads, worst / scale, same ? "equal" : "differ", ok ? "ok" : "FAIL");
    cpu_seed4_tiled_free(&tm);
    free(seeds); free(nibs); free(x); free(y); free(y1);
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
    ok &= check_tokens(40, 8, 3, 1);
    ok &= check_tokens(640, 320, 4, 3);   // a Qwen3.8 expert's gate / up (640 x 2560), an MTP verify window
    ok &= check3(37, 10, -19, 1, 1);
    ok &= check3(640, 320, -21, 4, 3);
    printf("test_cpu_seed4: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
