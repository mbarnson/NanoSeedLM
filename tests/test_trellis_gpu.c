// tests/test_trellis_gpu.c - the trellis encoder on CUDA (nslm/trellis.cu) against the C reference (nslm_tq_encode_row):
// the same words, scales and error, bit for bit, at L = 12 and 16, K = 2 to 4, gate/up (T = 256) and down (T = 320)
// blocks, a short last block, with and without importance weights, and a zero row (scale 0).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trellis.h"
#include "trellis_gpu.h"

static int failures;
#define CHECK(c, ...) do { if (!(c)) { ++failures; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rs = 4242;
static double gauss(void) {   // Box-Muller on an LCG
    rs = rs * 1664525u + 1013904223u;
    const double u = ((rs >> 8) + 0.5) / 16777216.0;
    rs = rs * 1664525u + 1013904223u;
    const double v = ((rs >> 8) + 0.5) / 16777216.0;
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * v);
}

static void one(NslmTqGpu* g, int rows, int cols, int T, int L, int K, int imat, int zero_row) {
    const size_t nw = nslm_tq_row_words(cols, T, L, K), nb = (size_t) (cols + T - 1) / T;
    float* w = malloc(sizeof(float) * (size_t) rows * cols), *h = imat ? malloc(sizeof(float) * cols) : NULL;
    for (size_t i = 0; i < (size_t) rows * cols; ++i) w[i] = (float) (0.02 * gauss());
    if (zero_row) memset(w, 0, sizeof(float) * cols);
    if (h) for (int j = 0; j < cols; ++j) h[j] = (float) exp(1.5 * gauss());   // an imatrix-like spread
    uint32_t* gw = malloc(sizeof(uint32_t) * nw * rows), *cw = malloc(sizeof(uint32_t) * nw * rows);
    uint16_t* gs = malloc(sizeof(uint16_t) * nb * rows), *cs = malloc(sizeof(uint16_t) * nb * rows);
    double* ge = malloc(sizeof(double) * rows), ce;
    char msg[256];
    const int rc = nslm_tq_gpu_encode(g, w, h, rows, cols, T, L, K, gw, gs, ge, msg, sizeof msg);
    CHECK(rc == 0, "L %d K %d T %d: %s", L, K, T, msg);
    NsTqScratch z;
    CHECK(nslm_tq_scratch_init(&z, T, L, K) == 0, "scratch");
    int bw = 0, bs = 0, be = 0;
    double rel = 0, den = 0;
    for (int r = 0; r < rows && rc == 0; ++r) {
        ce = nslm_tq_encode_row(w + (size_t) r * cols, h, cols, T, &z, cw + nw * r, cs + nb * r, NULL);
        bw += memcmp(gw + nw * r, cw + nw * r, sizeof(uint32_t) * nw) != 0;
        bs += memcmp(gs + nb * r, cs + nb * r, sizeof(uint16_t) * nb) != 0;
        be += ge[r] != ce;
        rel += ce;
        for (int j = 0; j < cols; ++j) den += (h ? h[j] : 1.0) * (double) w[(size_t) r * cols + j] * w[(size_t) r * cols + j];
    }
    printf("L %2d K %d T %d, %d x %d%s%s: rows with other words %d, scales %d, error %d (relative error %.5f)\n", L, K, T, rows, cols,
           imat ? ", imatrix" : "", zero_row ? ", a zero row" : "", bw, bs, be, den > 0 ? rel / den : 0.0);
    CHECK(!bw && !bs && !be, "L %d K %d T %d: the GPU differs from the C reference", L, K, T);
    nslm_tq_scratch_free(&z);
    free(w); free(h); free(gw); free(cw); free(gs); free(cs); free(ge);
}

int main(void) {
    char err[256];
    NslmTqGpu* g = nslm_tq_gpu_open(err, sizeof err);
    if (!g && strstr(err, "no CUDA device")) { printf("SKIP: %s\n", err); return 77; }
    if (!g) { printf("FAIL: %s\n", err); return 1; }
    one(g, 6, 600, 256, 12, 3, 0, 1);    // a short last block (88 weights), a zero row
    one(g, 4, 2560, 256, 12, 2, 1, 0);
    one(g, 4, 640, 320, 12, 4, 1, 0);
    one(g, 2, 512, 256, 16, 3, 1, 0);    // format v0: gate/up blocks
    one(g, 2, 640, 320, 16, 4, 1, 0);    // format v0: down blocks
    one(g, 1, 256, 256, 16, 2, 0, 0);    // Q = 16384: 32 items a thread, 64 KB of shared memory
    {   // a trellis the encoder cannot run fails loudly
        float w[16] = {0};
        uint32_t words[64];
        uint16_t sc[4];
        CHECK(nslm_tq_gpu_encode(g, w, NULL, 1, 16, 16, 6, 4, words, sc, NULL, err, sizeof err) != 0, "2 K > L accepted");
    }
    nslm_tq_gpu_close(g);
    printf("test_trellis_gpu: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
