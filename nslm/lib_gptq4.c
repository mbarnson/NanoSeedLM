// nslm/lib_gptq4.c - nslm4_gptq (gptq4.h): GPTQ-style error feedback over the P = 4 seed search.
#include "gptq4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"
#include "search4.h"

int nslm4_gptq(Nslm4SearchA search, void* ctx, int ns, int R, int C, float* const* W, const double* const* U, const int* bias,
               uint16_t* const* seed, uint16_t* const* coef, uint8_t* const* ecode, char* err, int errlen) {
    if (ns < 1 || R < 1 || C < 8 || C % 8) { snprintf(err, (size_t) errlen, "gptq: bad shape"); return -1; }
    const int ng = C / 8;
    int* ord = malloc(sizeof(int) * (size_t) ns);   // the slices by bias (stable): runs of one bias search together
    for (int i = 0; i < ns; ++i) ord[i] = i;
    for (int i = 1; i < ns; ++i)
        for (int k = i; k > 0 && bias[ord[k - 1]] > bias[ord[k]]; --k) { const int t = ord[k]; ord[k] = ord[k - 1]; ord[k - 1] = t; }
    float* wb = malloc(sizeof(float) * (size_t) R * 8 * ns), *A = malloc(sizeof(float) * 64 * (size_t) ns);
    float* E = malloc(sizeof(float) * (size_t) ns * R * 128);   // per slice, this batch's errors E = (W_b - Q_b) T [R][128]
    double* T = malloc(sizeof(double) * 64 * (size_t) ns);
    uint16_t* sb = malloc(2 * (size_t) R * ns), *cb = malloc(2 * (size_t) R * ns);
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
                if ((rc = search(ctx, wb, R, 8 * n, A, bias[ord[k0]], sb, cb, eb, err, errlen))) break;
                for (int i = 0; i < n; ++i) {
                    const int s = ord[k0 + i];
                    const double* t = T + (size_t) s * 64;
                    for (int r = 0; r < R; ++r) {
                        const size_t k = (size_t) r * n + i, o = (size_t) r * ng + b;
                        seed[s][o] = sb[k]; coef[s][o] = cb[k]; ecode[s][o] = eb[k];
                        uint16_t bf[8];
                        nslm4_decode_block(sb[k], cb[k], bias[s] + eb[k], bf);
                        double d[8];
                        for (int c = 0; c < 8; ++c) d[c] = (double) wb[k * 8 + c] - nslm4_bf2f(bf[c]);
                        float* er = E + ((size_t) s * R + r) * 128 + (g0 - c0);
                        for (int q = 0; q < 8; ++q) {
                            double a = 0;
                            for (int c = 0; c <= q; ++c) a += d[c] * t[c * 8 + q];
                            er[q] = (float) a;
                        }
                        float* wr = W[s] + (size_t) r * C;   // this batch's later groups take this group's error now
                        for (int q = 0; q < 8; ++q) {
                            const float eq = er[q];
                            const double* ur = U[s] + (size_t) (g0 + q) * C;
                            for (int c = g0 + 8; c < c1; ++c) wr[c] -= eq * (float) ur[c];
                        }
                    }
                }
                k0 = k1;
            }
        }
        for (int s = 0; s < ns && !rc && c1 < C; ++s)   // the later batches take this batch's errors
            for (int r = 0; r < R; ++r) {
                float* wr = W[s] + (size_t) r * C;
                const float* er = E + ((size_t) s * R + r) * 128;
                for (int q = 0; q < c1 - c0; ++q) {
                    const float eq = er[q];
                    const double* ur = U[s] + (size_t) (c0 + q) * C;
                    for (int c = c1; c < C; ++c) wr[c] -= eq * (float) ur[c];
                }
            }
    }
    free(ord); free(wb); free(A); free(E); free(T); free(sb); free(cb); free(eb);
    return rc ? -1 : 0;
}
