// harness/nslm-trellis-bench.c - weights a second of the CUDA trellis encoder (nslm/trellis.cu), plan T2: one expert's
// gate/up (640 x 2560, T = 256) and down (2560 x 640, T = 320) on N(0, 0.02^2) weights with an imatrix-like importance.
//   nslm-trellis-bench [--L 16] [--K 3] [--rows-check N]
// --rows-check N also encodes the first N rows with the C reference (nslm_tq_encode_row) and compares the words.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"
#include "trellis.h"
#include "trellis_gpu.h"

static uint32_t rs = 99;
static double gauss(void) {
    rs = rs * 1664525u + 1013904223u;
    const double u = ((rs >> 8) + 0.5) / 16777216.0;
    rs = rs * 1664525u + 1013904223u;
    const double v = ((rs >> 8) + 0.5) / 16777216.0;
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * v);
}

int main(int argc, char** argv) {
    int L = 16, K = 3, check = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--L") && i + 1 < argc) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows-check") && i + 1 < argc) check = atoi(argv[++i]);
        else { fprintf(stderr, "usage: nslm-trellis-bench [--L 16] [--K 3] [--rows-check N]\n"); return 2; }
    }
    char err[256];
    NslmTqGpu* g = nslm_tq_gpu_open(err, sizeof err);
    if (!g) { fprintf(stderr, "nslm-trellis-bench: %s\n", err); return 1; }
    static const struct { const char* name; int rows, cols, T; } sh[2] = {{"gate/up", 640, 2560, 256}, {"down", 2560, 640, 320}};
    for (int k = 0; k < 2; ++k) {
        const int rows = sh[k].rows, cols = sh[k].cols, T = sh[k].T;
        const size_t nw = nslm_tq_row_words(cols, T, L, K), nb = (size_t) (cols + T - 1) / T;
        float* w = malloc(sizeof(float) * (size_t) rows * cols), *h = malloc(sizeof(float) * cols);
        for (size_t i = 0; i < (size_t) rows * cols; ++i) w[i] = (float) (0.02 * gauss());
        for (int j = 0; j < cols; ++j) h[j] = (float) exp(1.5 * gauss());
        uint32_t* words = malloc(sizeof(uint32_t) * nw * rows);
        uint16_t* sc = malloc(sizeof(uint16_t) * nb * rows);
        double* e = malloc(sizeof(double) * rows);
        const double t0 = now_s();
        if (nslm_tq_gpu_encode(g, w, h, rows, cols, T, L, K, words, sc, e, err, sizeof err)) { fprintf(stderr, "nslm-trellis-bench: %s\n", err); return 1; }
        const double dt = now_s() - t0;
        double num = 0, den = 0;
        for (int r = 0; r < rows; ++r) {
            num += e[r];
            for (int j = 0; j < cols; ++j) den += h[j] * (double) w[(size_t) r * cols + j] * w[(size_t) r * cols + j];
        }
        printf("L %d K %d %-7s %4d x %4d, T %d: %.3f s, %.0f weights a second, weighted relative error %.5f (%.3f bits a weight)\n", L, K,
               sh[k].name, rows, cols, T, dt, (double) rows * cols / dt, num / den, K + (L - K + 16.0) / T);
        if (check > 0) {
            NsTqScratch z;
            nslm_tq_scratch_init(&z, T, L, K);
            uint32_t* cw = malloc(sizeof(uint32_t) * nw);
            uint16_t* cs = malloc(sizeof(uint16_t) * nb);
            int bad = 0;
            const double c0 = now_s();
            for (int r = 0; r < check && r < rows; ++r) {
                nslm_tq_encode_row(w + (size_t) r * cols, h, cols, T, &z, cw, cs, NULL);
                bad += memcmp(cw, words + nw * r, sizeof(uint32_t) * nw) != 0 || memcmp(cs, sc + nb * r, sizeof(uint16_t) * nb) != 0;
            }
            const double cdt = now_s() - c0;
            printf("  C reference, %d rows: %d differ; %.0f weights a second on one core\n", check, bad, (double) check * cols / cdt);
            nslm_tq_scratch_free(&z);
            free(cw); free(cs);
        }
        free(w); free(h); free(words); free(sc); free(e);
    }
    nslm_tq_gpu_close(g);
    return 0;
}
