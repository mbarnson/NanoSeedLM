// tests/test_tq_mv_gpu.c - the trellis expert gathers on CUDA (kc_tq_mv_gu, kc_tq_mv_sel in engine/kernels_moe.cu)
// against a C matvec over nslm_tq_decode_row rows: gate + up + SwiGLU and down, 10 and 30 (token, expert) pairs, K = 2
// to 4 bits a weight, L = 12 and 16, T = 256 and 320, a row count that is not a multiple of 8.  Outputs are BF16; they
// match the reference within the matvec's float sum order (a BF16 rounding step or two).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "kernels_cuda.h"
#include "lfsr.h"
#include "trellis.h"

static int failures;
#define CHECK(c, ...) do { if (!(c)) { ++failures; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rs = 2718;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs; }
static float bfr_h(float f) { return nslm_bf2f(nslm_f2bf(f)); }
static float silu_bf_h(float g) { const float s = bfr_h(1.0f / (1.0f + expf(-g))); return bfr_h(g * s); }

typedef struct { uint32_t* words; uint16_t* sc; float* dec; WSlice* tab; void *dw, *ds; } Tq;   // S slices of R x C

static Tq make(int S, int R, int C, int L, int K, int T) {
    Tq t;
    const size_t nw = nslm_tq_row_words(C, T, L, K), nb = (size_t) (C / T);
    t.words = malloc(sizeof(uint32_t) * nw * R * S);
    t.sc = malloc(sizeof(uint16_t) * nb * R * S);
    t.dec = malloc(sizeof(float) * (size_t) C * R * S);
    for (size_t i = 0; i < nw * R * S; ++i) t.words[i] = rnd();   // every bit stream is a valid code
    for (size_t i = 0; i < nb * R * S; ++i) t.sc[i] = nslm_f2bf(0.0002f * (float) (1 + rnd() % 9));
    for (size_t r = 0; r < (size_t) R * S; ++r) nslm_tq_decode_row(t.words + nw * r, t.sc + nb * r, C, T, L, K, t.dec + (size_t) C * r);
    cudaMalloc(&t.dw, sizeof(uint32_t) * nw * R * S);
    cudaMalloc(&t.ds, sizeof(uint16_t) * nb * R * S);
    cudaMemcpy(t.dw, t.words, sizeof(uint32_t) * nw * R * S, cudaMemcpyHostToDevice);
    cudaMemcpy(t.ds, t.sc, sizeof(uint16_t) * nb * R * S, cudaMemcpyHostToDevice);
    WSlice* h = calloc((size_t) S, sizeof(WSlice));
    for (int s = 0; s < S; ++s) { h[s].p[0] = (uint32_t*) t.dw + nw * R * s; h[s].p[1] = (uint16_t*) t.ds + nb * R * s; }
    cudaMalloc((void**) &t.tab, sizeof(WSlice) * S);
    cudaMemcpy(t.tab, h, sizeof(WSlice) * S, cudaMemcpyHostToDevice);
    free(h);
    return t;
}
static void unmake(Tq* t) { free(t->words); free(t->sc); free(t->dec); cudaFree(t->dw); cudaFree(t->ds); cudaFree(t->tab); }
static double dot(const float* w, const float* x, int n) { double s = 0; for (int j = 0; j < n; ++j) s += (double) w[j] * x[j]; return s; }

// |gpu - ref| within rel of |ref| (plus a floor at 1e-3 of the outputs' largest magnitude)
static int compare(const char* what, const float* g, const float* ref, int n, double rel) {
    double mx = 0;
    for (int i = 0; i < n; ++i) mx = fmax(mx, fabs(ref[i]));
    int bad = 0, exact = 0;
    double worst = 0;
    for (int i = 0; i < n; ++i) {
        const double d = fabs((double) g[i] - ref[i]);
        exact += g[i] == ref[i];
        if (d > rel * fabs(ref[i]) + 1e-3 * mx) ++bad;
        if (fabs(ref[i]) > 1e-3 * mx) worst = fmax(worst, d / fabs(ref[i]));
    }
    printf("  %-34s %6d outputs: %5.1f%% equal, worst relative difference %.2e%s\n", what, n, 100.0 * exact / n, worst, bad ? "  <-- FAIL" : "");
    return bad;
}

static void one(int L, int K, int T, int R, int C, int P, int xdiv) {
    const int S = 12;
    Tq g = make(S, R, C, L, K, T), u = make(S, R, C, L, K, T);
    const int ntok = (P + xdiv - 1) / xdiv;
    float* x = malloc(sizeof(float) * (size_t) C * ntok);
    for (int i = 0; i < C * ntok; ++i) x[i] = bfr_h((float) ((int) (rnd() % 2001) - 1000) / 1000.0f);
    int32_t sel[64];
    for (int p = 0; p < P; ++p) sel[p] = (int32_t) ((p * 7 + 3) % S);
    float *dx, *dy;
    int32_t* dsel;
    cudaMalloc((void**) &dx, sizeof(float) * (size_t) C * ntok);
    cudaMalloc((void**) &dy, sizeof(float) * (size_t) R * P);
    cudaMalloc((void**) &dsel, sizeof(int32_t) * P);
    cudaMemcpy(dx, x, sizeof(float) * (size_t) C * ntok, cudaMemcpyHostToDevice);
    cudaMemcpy(dsel, sel, sizeof(int32_t) * P, cudaMemcpyHostToDevice);
    float* y = malloc(sizeof(float) * (size_t) R * P), *ref = malloc(sizeof(float) * (size_t) R * P);
    char nm[96];
    // gate + up + SwiGLU
    CHECK(kc_tq_mv_gu(0, L, K, T, g.tab, u.tab, C, R, dx, C, dy, R, dsel, P, xdiv) == 0, "kc_tq_mv_gu L %d K %d T %d", L, K, T);
    cudaMemcpy(y, dy, sizeof(float) * (size_t) R * P, cudaMemcpyDeviceToHost);
    for (int p = 0; p < P; ++p)
        for (int r = 0; r < R; ++r) {
            const float* xr = x + (size_t) (p / xdiv) * C;
            const float gs = (float) dot(g.dec + ((size_t) sel[p] * R + r) * C, xr, C), us = (float) dot(u.dec + ((size_t) sel[p] * R + r) * C, xr, C);
            ref[(size_t) p * R + r] = bfr_h(silu_bf_h(bfr_h(gs)) * bfr_h(us));
        }
    snprintf(nm, sizeof nm, "gate+up L %d K %d T %d %dx%d P %d", L, K, T, R, C, P);
    CHECK(!compare(nm, y, ref, R * P, 1.0 / 32), "%s", nm);
    // one projection (down's kernel, on the gate slices)
    CHECK(kc_tq_mv_sel(0, L, K, T, g.tab, C, R, dx, C, dy, R, dsel, P, xdiv) == 0, "kc_tq_mv_sel L %d K %d T %d", L, K, T);
    cudaMemcpy(y, dy, sizeof(float) * (size_t) R * P, cudaMemcpyDeviceToHost);
    for (int p = 0; p < P; ++p)
        for (int r = 0; r < R; ++r) ref[(size_t) p * R + r] = bfr_h((float) dot(g.dec + ((size_t) sel[p] * R + r) * C, x + (size_t) (p / xdiv) * C, C));
    snprintf(nm, sizeof nm, "gather L %d K %d T %d %dx%d P %d", L, K, T, R, C, P);
    CHECK(!compare(nm, y, ref, R * P, 1.0 / 64), "%s", nm);
    const cudaError_t e = cudaDeviceSynchronize();
    CHECK(e == cudaSuccess, "CUDA: %s", cudaGetErrorString(e));
    cudaFree(dx); cudaFree(dy); cudaFree(dsel);
    free(x); free(y); free(ref);
    unmake(&g); unmake(&u);
}

int main(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { printf("SKIP: no CUDA device\n"); return 77; }
    for (int K = 2; K <= 4; ++K) {
        one(16, K, 256, 76, 2560, 10, 10);   // gate/up's shape at 1 token (76 rows: not a multiple of 8)
        one(16, K, 320, 200, 640, 30, 10);   // down's shape at 3 tokens
    }
    one(12, 3, 256, 40, 512, 30, 10);
    one(12, 4, 320, 40, 640, 10, 1);
    {   // a trellis without a kernel fails loudly
        CHECK(kc_tq_mv_sel(0, 16, 5, 256, NULL, 2560, 8, NULL, 2560, NULL, 8, NULL, 1, 1) != 0, "K = 5 accepted");
        CHECK(kc_tq_mv_sel(0, 16, 3, 256, NULL, 2500, 8, NULL, 2500, NULL, 8, NULL, 1, 1) != 0, "T not dividing K accepted");
    }
    printf("test_tq_mv_gpu: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
