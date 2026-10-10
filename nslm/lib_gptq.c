// nslm/lib_gptq.c - nslm_gptq (gptq.h): GPTQ-style error feedback over a seed search of any block codec.
#include "gptq.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"
#include "search4.h"

// W[r][c1:] -= sum_q E[r][q] U[c0 + q][c1:] for rows r0 .. r1 of every slice
typedef struct { int ns, R, C, c0, ne, c1, r0, r1; float* const* W; const float* E; const double* const* U; } FbJob;
static void* fb_rows(void* arg) {
    const FbJob* j = (const FbJob*) arg;
    for (int s = 0; s < j->ns; ++s)
        for (int r = j->r0; r < j->r1; ++r) {
            float* wr = j->W[s] + (size_t) r * j->C;
            const float* er = j->E + ((size_t) s * j->R + r) * 128;
            for (int q = 0; q < j->ne; ++q) {
                const float eq = er[q];
                const double* ur = j->U[s] + (size_t) (j->c0 + q) * j->C;
                for (int c = j->c1; c < j->C; ++c) wr[c] -= eq * (float) ur[c];
            }
        }
    return NULL;
}
static void feedback(int threads, int ns, int R, int C, int c0, int ne, int c1, float* const* W, const float* E, const double* const* U) {
    enum { NT = 32 };
    const int nt = threads < 1 ? 1 : threads > NT ? NT : threads, per = (R + nt - 1) / nt;
    FbJob jb[NT];
    pthread_t th[NT];
    for (int i = 0; i < nt; ++i) jb[i] = (FbJob) {ns, R, C, c0, ne, c1, i * per < R ? i * per : R, (i + 1) * per < R ? (i + 1) * per : R, W, E, U};
    if (nt == 1) { fb_rows(&jb[0]); return; }
    for (int i = 0; i < nt; ++i) pthread_create(&th[i], NULL, fb_rows, &jb[i]);
    for (int i = 0; i < nt; ++i) pthread_join(th[i], NULL);
}

int nslm_gptq(const NslmGptq* q, int ns, int R, int C, float* const* W, const double* const* U, const int* bias,
              uint16_t* const* seed, uint32_t* const* coef, uint8_t* const* ecode, char* err, int errlen) {
    if (ns < 1 || R < 1 || C < 8 || C % 8) { snprintf(err, (size_t) errlen, "gptq: bad shape"); return -1; }
    const int ng = C / 8;
    int* ord = malloc(sizeof(int) * (size_t) ns);   // the slices by bias (stable): runs of one bias search together
    for (int i = 0; i < ns; ++i) ord[i] = i;
    for (int i = 1; i < ns; ++i)
        for (int k = i; k > 0 && bias[ord[k - 1]] > bias[ord[k]]; --k) { const int t = ord[k]; ord[k] = ord[k - 1]; ord[k - 1] = t; }
    float* wb = malloc(sizeof(float) * (size_t) R * 8 * ns), *A = malloc(sizeof(float) * 64 * (size_t) ns);
    float* E = malloc(sizeof(float) * (size_t) ns * R * 128);   // per slice, this batch's errors E = (W_b - Q_b) T [R][128]
    double* T = malloc(sizeof(double) * 64 * (size_t) ns);
    uint16_t* sb = malloc(2 * (size_t) R * ns);
    uint32_t* cb = malloc(4 * (size_t) R * ns);
    uint8_t* eb = malloc((size_t) R * ns);
    int rc = 0;
    for (int bb = 0; bb < ng && !rc; bb += 16) {   // batches of 16 column groups (lazy feedback)
        const int b1 = bb + 16 < ng ? bb + 16 : ng, c0 = bb * 8, c1 = b1 * 8;
        for (int b = bb; b < b1 && !rc; ++b) {
            const int g0 = b * 8;
            for (int k0 = 0; k0 < ns && !rc;) {
                int k1 = k0 + 1;
                while (k1 < ns && bias[ord[k1]] == bias[ord[k0]]) ++k1;
                const int n = k1 - k0;
                for (int i = 0; i < n; ++i) {
                    const int s = ord[k0 + i];
                    double* t = T + (size_t) s * 64;
                    nslm_upper8_inverse(U[s] + (size_t) g0 * C + g0, C, t);
                    for (int r = 0; r < 8; ++r)
                        for (int c = 0; c < 8; ++c) A[(size_t) i * 64 + r * 8 + c] = (float) t[c * 8 + r];   // A = T^T
                    for (int r = 0; r < R; ++r) memcpy(wb + ((size_t) r * n + i) * 8, W[s] + (size_t) r * C + g0, sizeof(float) * 8);
                }
                if ((rc = q->search(q->ctx, wb, R, 8 * n, A, bias[ord[k0]], sb, cb, eb, err, errlen))) break;
                for (int i = 0; i < n; ++i) {
                    const int s = ord[k0 + i];
                    const double* t = T + (size_t) s * 64;
                    for (int r = 0; r < R; ++r) {
                        const size_t k = (size_t) r * n + i, o = (size_t) r * ng + b;
                        seed[s][o] = sb[k]; coef[s][o] = cb[k]; ecode[s][o] = eb[k];
                        uint16_t bf[8];
                        q->decode(sb[k], cb[k], bias[s] + eb[k], bf);
                        double d[8];
                        for (int c = 0; c < 8; ++c) d[c] = (double) wb[k * 8 + c] - nslm4_bf2f(bf[c]);
                        float* er = E + ((size_t) s * R + r) * 128 + (g0 - c0);
                        for (int j = 0; j < 8; ++j) {
                            double a = 0;
                            for (int c = 0; c <= j; ++c) a += d[c] * t[c * 8 + j];
                            er[j] = (float) a;
                        }
                        float* wr = W[s] + (size_t) r * C;   // this batch's later groups take this group's error now
                        for (int j = 0; j < 8; ++j) {
                            const float eq = er[j];
                            const double* ur = U[s] + (size_t) (g0 + j) * C;
                            for (int c = g0 + 8; c < c1; ++c) wr[c] -= eq * (float) ur[c];
                        }
                    }
                }
                k0 = k1;
            }
        }
        if (!rc && c1 < C) feedback(q->threads, ns, R, C, c0, c1 - c0, c1, W, E, U);   // the later batches
    }
    free(ord); free(wb); free(A); free(E); free(T); free(sb); free(cb); free(eb);
    return rc ? -1 : 0;
}
