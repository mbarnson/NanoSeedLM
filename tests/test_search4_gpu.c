// tests/test_search4_gpu.c - the P = 4 GPU search (nslm/search4.metal or nslm/search4.cu) against the scalar C reference
// (nslm4_search_ref): identical seed, coefficients, exponent code and error bits for every block.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gptq4.h"
#include "linalg.h"
#include "search.h"
#include "search4.h"
#include "search4_gpu.h"

static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

// nslm4_gptq's search through the GPU, or through the C reference (seed tables per column group)
typedef struct { Nslm4Gpu* g; const Search4Opts* o; } GpuCtx;
static int search_gpu(void* ctx, const float* w, int rows, int cols, const float* A, int bias, uint16_t* seed, uint16_t* coef,
                      uint8_t* ecode, char* err, int errlen) {
    GpuCtx* c = (GpuCtx*) ctx;
    float* er = malloc(4 * (size_t) rows * (cols / 8));
    const int rc = nslm4_gpu_search_a(c->g, w, rows, cols, A, bias, c->o, seed, coef, ecode, er, err, errlen);
    free(er);
    return rc;
}
static int search_ref(void* ctx, const float* w, int rows, int cols, const float* A, int bias, uint16_t* seed, uint16_t* coef,
                      uint8_t* ecode, char* err, int errlen) {
    const Search4Opts* o = ((GpuCtx*) ctx)->o;
    const int ng = cols / 8;
    float* tab = calloc((size_t) (o->n_seeds + 1) * NSLM4_ENT, sizeof(float)), *wg = malloc(sizeof(float) * rows * 8), *er = malloc(4 * (size_t) rows);
    uint8_t* ok = calloc((size_t) o->n_seeds + 1, 1), *ce = malloc((size_t) rows);
    uint16_t* cs = malloc(2 * (size_t) rows), *cc = malloc(2 * (size_t) rows);
    for (int gi = 0; gi < ng; ++gi) {
        for (int s = 1; s <= o->n_seeds; ++s) ok[s] = (uint8_t) nslm4_seed_entry_a(s, A + gi * 64, tab + (size_t) s * NSLM4_ENT);
        for (int r = 0; r < rows; ++r) memcpy(wg + r * 8, w + (size_t) r * cols + gi * 8, 32);
        nslm4_search_ref_a(tab, ok, wg, rows, A + gi * 64, bias, o, cs, cc, ce, er);
        for (int r = 0; r < rows; ++r) { seed[(size_t) r * ng + gi] = cs[r]; coef[(size_t) r * ng + gi] = cc[r]; ecode[(size_t) r * ng + gi] = ce[r]; }
    }
    free(tab); free(wg); free(er); free(ok); free(ce); free(cs); free(cc);
    (void) err; (void) errlen;
    return 0;
}
// sum over slices of |(W - Q) X^T|^2 / |W X^T|^2 with H = X^T X / n: tr((W - Q) H (W - Q)^T) / tr(W H W^T)
static double out_err(const float* W, const uint16_t* sd, const uint16_t* cf, const uint8_t* ec, int bias, int R, int C, const double* H) {
    double num = 0, den = 0;
    double* d = malloc(sizeof(double) * C);
    for (int r = 0; r < R; ++r) {
        for (int b = 0; b < C / 8; ++b) {
            uint16_t bf[8];
            const size_t k = (size_t) r * (C / 8) + b;
            nslm4_decode_block(sd[k], cf[k], bias + ec[k], bf);
            for (int c = 0; c < 8; ++c) d[b * 8 + c] = W[(size_t) r * C + b * 8 + c] - nslm4_bf2f(bf[c]);
        }
        for (int i = 0; i < C; ++i)
            for (int j = 0; j < C; ++j) { num += d[i] * H[i * C + j] * d[j]; den += W[(size_t) r * C + i] * H[i * C + j] * W[(size_t) r * C + j]; }
    }
    free(d);
    return num / den;
}

#define CHECK0(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); return 1; } } while (0)
int main(void) {
    {
        char err[512];
        Nslm4Gpu* g = nslm4_gpu_open("out/res/search4.metallib", err, sizeof err);
        if (!g && (strstr(err, "no CUDA device") || strstr(err, "no Metal device"))) { printf("SKIP: %s\n", err); return 77; }   // a machine without a GPU
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
        // full transform: a lower-triangular A per column group (nslm4_gpu_search_a vs nslm4_search_ref_a)
        float* A = calloc((size_t) ng * 64, sizeof(float));
        for (int gi = 0; gi < ng; ++gi)
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j <= i; ++j) A[gi * 64 + i * 8 + j] = i == j ? (float) (1.0 + fabs(frand(&sd))) : (float) (frand(&sd) * 0.5);
        if (nslm4_gpu_search_a(g, w, rows, cols, A, bias, &o, gs, gc, ge, gr, err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
        long bada = 0, badra = 0;
        for (int gi = 0; gi < ng; ++gi) {
            for (int s = 1; s <= NS; ++s) ok[s] = (uint8_t) nslm4_seed_entry_a(s, A + gi * 64, tab + (size_t) s * NSLM4_ENT);
            for (int r = 0; r < rows; ++r) memcpy(wg + r * 8, w + (size_t) r * cols + gi * 8, 32);
            nslm4_search_ref_a(tab, ok, wg, rows, A + gi * 64, bias, &o, cs, cc, ce, cr);
            for (int r = 0; r < rows; ++r) {
                const size_t k = (size_t) r * ng + gi;
                bada += !(gs[k] == cs[r] && gc[k] == cc[r] && ge[k] == ce[r]);
                badra += memcmp(&gr[k], &cr[r], 4) != 0;
            }
        }
        printf("search4 full A GPU vs C: %ld blocks differ, %ld error bits differ, of %zu\n", bada, badra, nb);
        bad += bada; badr += badra;
        {   // nslm4_gptq (nslm/gptq4.h): three slices (two share an exponent bias), each with correlated inputs; the GPU
            // and the C reference searches give the same seeds, and GPTQ's output error is below the sqrt(diag H) search's
            enum { NSL = 3, R = 96, C = 48, N = 400, NS2 = 512 };
            const Search4Opts o2 = {NS2, 3, {0, -1, 1}, 1};
            float* W[NSL], *Wg[NSL], *Wc[NSL];
            double* H[NSL], *U[NSL];
            int bs[NSL];
            uint16_t* sg[NSL], *cg[NSL], *sc[NSL], *ccr[NSL], *sa[NSL], *ca[NSL];
            uint8_t* eg[NSL], *ecr[NSL], *ea[NSL];
            unsigned s2 = 21;
            for (int k = 0; k < NSL; ++k) {
                W[k] = malloc(sizeof(float) * R * C); Wg[k] = malloc(sizeof(float) * R * C); Wc[k] = malloc(sizeof(float) * R * C);
                for (int i = 0; i < R * C; ++i) W[k][i] = (float) (frand(&s2) * 0.02 * (k == 1 ? 9 : 1));
                memcpy(Wg[k], W[k], sizeof(float) * R * C);
                memcpy(Wc[k], W[k], sizeof(float) * R * C);
                int64_t cl = 0;
                bs[k] = nslm_choose_bias(W[k], (int64_t) R * C / 8, &cl);
                double* X = malloc(sizeof(double) * N * C), *M = malloc(sizeof(double) * C * C);   // x = M z: correlated
                for (int i = 0; i < C * C; ++i) M[i] = frand(&s2) * (i % (C + 1) == 0 ? 3 : 0.7);
                for (int n = 0; n < N; ++n) {
                    double z[C];
                    for (int c = 0; c < C; ++c) z[c] = frand(&s2);
                    for (int i = 0; i < C; ++i) { double a = 0; for (int c = 0; c < C; ++c) a += M[i * C + c] * z[c]; X[n * C + i] = a; }
                }
                H[k] = calloc((size_t) C * C, sizeof(double)); U[k] = malloc(sizeof(double) * C * C);
                for (int n = 0; n < N; ++n)
                    for (int i = 0; i < C; ++i)
                        for (int j = 0; j < C; ++j) H[k][i * C + j] += X[n * C + i] * X[n * C + j] / N;
                CHECK0(nslm_gptq_factor(H[k], C, 0.01, U[k], NULL) == 0, "gptq factor");
                free(X); free(M);
                sg[k] = malloc(R * C / 4); cg[k] = malloc(R * C / 4); eg[k] = malloc(R * C / 8);
                sc[k] = malloc(R * C / 4); ccr[k] = malloc(R * C / 4); ecr[k] = malloc(R * C / 8);
                sa[k] = malloc(R * C / 4); ca[k] = malloc(R * C / 4); ea[k] = malloc(R * C / 8);
            }
            if (bs[0] != bs[2] || bs[0] == bs[1]) printf("  (biases %d %d %d: the grouping is not exercised)\n", bs[0], bs[1], bs[2]);
            GpuCtx gc2 = {g, &o2};
            int rc = nslm4_gptq(search_gpu, &gc2, NSL, R, C, Wg, (const double* const*) U, bs, sg, cg, eg, err, sizeof err);
            rc |= nslm4_gptq(search_ref, &gc2, NSL, R, C, Wc, (const double* const*) U, bs, sc, ccr, ecr, err, sizeof err);
            long badg = 0;
            double eq = 0, ea2 = 0;
            for (int k = 0; k < NSL && !rc; ++k) {
                for (int i = 0; i < R * C / 8; ++i) badg += sg[k][i] != sc[k][i] || cg[k][i] != ccr[k][i] || eg[k][i] != ecr[k][i];
                float shk[C], *er2 = malloc(4 * R * C / 8);
                for (int c = 0; c < C; ++c) shk[c] = (float) sqrt(H[k][c * C + c]);
                rc |= nslm4_gpu_search(g, W[k], R, C, shk, bs[k], &o2, sa[k], ca[k], ea[k], er2, err, sizeof err);
                free(er2);
                eq += out_err(W[k], sg[k], cg[k], eg[k], bs[k], R, C, H[k]);
                ea2 += out_err(W[k], sa[k], ca[k], ea[k], bs[k], R, C, H[k]);
            }
            printf("gptq4: biases %d %d %d; GPU vs C %ld blocks differ; relative output error GPTQ %.5f, sqrt(diag H) %.5f\n", bs[0], bs[1],
                   bs[2], badg, eq / NSL, ea2 / NSL);
            if (rc) printf("FAIL: gptq4: %s\n", err);
            bad += rc != 0 || badg != 0 || !(eq < 0.8 * ea2);
        }
        nslm4_gpu_close(g);
        printf(bad || badr ? "FAIL\n" : "PASS\n");
        return bad || badr;
    }
}
