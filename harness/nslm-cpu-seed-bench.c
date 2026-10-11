// harness/nslm-cpu-seed-bench.c - CPU P = 4 seed mat-vec throughput (G weights / s) by thread count, on random
// seed matrices shaped like MoVA's routed experts (gate / up: 768 x 2560, down: 2560 x 768) and its decode step.
//
//   nslm-cpu-seed-bench [threads,...] [reps]
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
    // one decode step's routed experts: 8 experts x (gate + up + down) of 768 x 2560 = 47.2 M weights
    const int rows = 8 * 3 * 768, kb = 2560 / 8;
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
    CpuSeed4Tiled tm;
    cpu_seed4_tile(&tm, seeds, coefs, codes, -22, rows, kb);
    printf("matrix %d x %d (%.1f M weights, %.1f MB of seed data)\n", rows, kb * 8, n * 8 / 1e6, n * 4.5 / 1e6);
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
        printf("  tiled    %.2f ms per step, %.1f G weights/s\n", best * 1e3, n * 8 / best / 1e9);
    }
    return 0;
}
