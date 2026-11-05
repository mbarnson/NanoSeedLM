// harness/nslm-cpu-seed-bench.c - CPU P = 4 seed mat-vec throughput (G weights / s) by thread count, on random
// seed matrices shaped like MoVA's routed experts (gate / up: 768 x 2560, down: 2560 x 768) and its decode step; and
// per token for 2 and 4 tokens at once (an MTP verify window), at P = 4 and P = 3.  Prints the load average.
// qwen4: one layer of Qwen3.8-Flash-Next's decode instead (10 experts of 640 x 2560), and the time for its 48 layers.
//
//   nslm-cpu-seed-bench [threads,...] [reps] [qwen4]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpu_seed4.h"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    const char* tl = argc > 1 ? argv[1] : "1,2,4,8,12,16";
    const int reps = argc > 2 ? atoi(argv[2]) : 20;
    // one decode step's routed experts: 8 experts x (gate + up + down) of 768 x 2560 = 47.2 M weights (qwen4: 10 of
    // 640 x 2560 a layer, 49.2 M; 48 layers a token)
    const int qwen4 = argc > 3 && !strcmp(argv[3], "qwen4"), layers = qwen4 ? 48 : 1;
    const int rows = qwen4 ? 10 * 3 * 640 : 8 * 3 * 768, kb = 2560 / 8;
    const size_t n = (size_t) rows * kb;
    uint16_t* seeds = malloc(n * 2);
    uint16_t* coefs = malloc(n * 2);
    uint8_t* codes = malloc(n / 2 + 1);
    float* x = malloc(kb * 8 * 4);
    float* y = malloc(rows * 4);
    uint32_t r = 7;
    for (size_t i = 0; i < n; ++i) {
        r = r * 1664525u + 1013904223u;
        seeds[i] = (uint16_t) (1 + (r >> 8) % 65535);
        coefs[i] = (uint16_t) (r >> 12);
    }
    memset(codes, 0x77, n / 2 + 1);
    for (int i = 0; i < kb * 8; ++i) x[i] = (float) (i % 17) / 17.0f - 0.5f;
    const uint32_t* table = cpu_seed4_table();
    CpuSeed4Tiled tm, t3;
    cpu_seed4_tile(&tm, seeds, coefs, codes, -22, rows, kb);
    cpu_seed3_tile(&t3, seeds, coefs, -22, rows, kb);   // the same words as P = 3 nibble words
    printf("matrix %d x %d (%.1f M weights, %.1f MB of seed data)\n", rows, kb * 8, n * 8 / 1e6, n * 4.5 / 1e6);
    FILE* la = fopen("/proc/loadavg", "r");   // Linux; elsewhere read uptime(1) before the run
    char lb[64] = "n/a";
    if (la) { if (!fgets(lb, sizeof lb, la)) strcpy(lb, "n/a\n"); fclose(la); } else strcat(lb, "\n");
    printf("load average: %s", lb);
    float* xt = malloc((size_t) 4 * kb * 8 * 4), *yt = malloc((size_t) 4 * rows * 4);
    for (int i = 0; i < 4 * kb * 8; ++i) xt[i] = (float) (i % 23) / 23.0f - 0.5f;
    char buf[256];
    strncpy(buf, tl, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    for (char* tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        const int th = atoi(tok);
        cpu_seed4_matvec_mt(seeds, coefs, codes, -22, table, x, rows, kb, y, th);   // warm-up
        double best = 1e9;
        for (int i = 0; i < reps; ++i) {
            const double t0 = now();
            cpu_seed4_matvec_mt(seeds, coefs, codes, -22, table, x, rows, kb, y, th);
            const double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("threads %2d: %.2f ms per step, %.1f G weights/s (best of %d)  y[0] %.4f\n", th, best * 1e3,
               n * 8 / best / 1e9, reps, y[0]);
        best = 1e9;
        for (int i = 0; i < reps; ++i) {
            const double t0 = now();
            cpu_seed4_tiled_matvec_mt(&tm, table, x, y, th);
            const double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("  tiled    %.2f ms per step, %.1f G weights/s", best * 1e3, n * 8 / best / 1e9);
        if (layers > 1) printf(", %.1f ms for %d layers", best * 1e3 * layers, layers);
        printf("\n");
        for (int pp = 4; pp >= 3; --pp)
            for (int nt = pp == 4 ? 2 : 1; nt <= 4; nt *= 2) {
                best = 1e9;
                for (int i = 0; i < reps; ++i) {
                    const double t0 = now();
                    cpu_seed4_tiled_matmul_mt(pp == 4 ? &tm : &t3, table, xt, nt, yt, th);
                    const double dt = now() - t0;
                    if (dt < best) best = dt;
                }
                printf("  P = %d, %d token%s %.2f ms per step, %.1f G weights/s per token\n", pp, nt, nt > 1 ? "s" : "",
                       best * 1e3, n * 8 * nt / best / 1e9);
            }
    }
    return 0;
}
