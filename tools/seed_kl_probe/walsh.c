// walsh.c - probe: is the P3 block error, at fixed (q, e), a short sum of Walsh characters (-1)^<b,s> of the seed?
// For real blocks: brute-force the best (s, q, e) (least squares, e0 and e0 +- 1, no refit), then expand
// err(s) = sum_c (x_c - sc sum_p q_p S_p(s)[c])^2 in the signs c_j = (-1)^x_j of the stream bits x_j = <a_j, s> (GF(2)),
// evaluate the expansion for all seeds by one fast Walsh-Hadamard transform, and compare with direct evaluation.
//   walsh gu.f32 NBLK
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"

#define N 65536
static double V[N], D[N];
static uint16_t A[40];   // a_j: stream bit j as a GF(2) linear form of the 16 seed bits

static void fwht(double* v) {
    for (int h = 1; h < N; h <<= 1)
        for (int i = 0; i < N; i += 2 * h)
            for (int j = i; j < i + h; ++j) { const double a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
}

int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    const long nw = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    float* w = malloc(nw * 4);
    if (fread(w, 4, nw, f) != (size_t) nw) return 1;
    const int nblk = atoi(argv[2]);
    for (int j = 0; j < 16; ++j) A[j] = (uint16_t) (1u << j);
    for (int j = 16; j < 40; ++j) {
        uint16_t a = 0;
        for (int i = 0; i < 16; ++i) a |= (uint16_t) (((lfsr_stream32((uint16_t) (1u << i)) >> (j - 16)) & 1u) << i);
        A[j] = a;
    }
    // check linearity on random seeds
    for (int t = 1; t < 65536; t += 977) {
        const uint32_t g = lfsr_stream32((uint16_t) t);
        for (int j = 16; j < 40; ++j)
            if ((int) ((g >> (j - 16)) & 1u) != __builtin_parity(A[j] & t)) { printf("linearity fails\n"); return 1; }
    }
    static double S[N][24];
    for (int s = 1; s < N; ++s) {
        uint16_t st[24];
        lfsr_states((uint16_t) s, 24, st);
        for (int k = 0; k < 24; ++k) S[s][k] = (double) st[k] - 32768.0;
    }
    double worst = 0;
    int minterms = N, maxterms = 0;
    for (int bi = 0; bi < nblk; ++bi) {
        const float* x = w + 8 * (long) bi * (nw / 8 / nblk);
        // brute force (s, q, e)
        double best = INFINITY;
        int bs = 1, be = 0, bq[3] = {0, 0, 0};
        for (int s = 1; s < N; ++s) {
            double G[3][3] = {{0}}, b[3] = {0};
            for (int c = 0; c < 8; ++c)
                for (int p = 0; p < 3; ++p) {
                    const double u = S[s][3 * c + p] * NSLM_R32;
                    b[p] += u * x[c];
                    for (int r = 0; r < 3; ++r) G[p][r] += u * S[s][3 * c + r] * NSLM_R32;
                }
            // solve 3x3 by Cramer
            const double det = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1]) - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0]) +
                               G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);
            if (fabs(det) < 1e-12) continue;
            double t[3];
            for (int k = 0; k < 3; ++k) {
                double M[3][3];
                memcpy(M, G, sizeof M);
                for (int r = 0; r < 3; ++r) M[r][k] = b[r];
                t[k] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                        M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) / det;
            }
            const double m = fmax(fabs(t[0]), fmax(fabs(t[1]), fabs(t[2])));
            if (m == 0) continue;
            const int e0 = (int) floor(log2(m)) - 2;
            for (int de = -1; de <= 1; ++de) {
                const int e = e0 + de;
                int q[3];
                for (int p = 0; p < 3; ++p) { double r = nearbyint(t[p] / ldexp(1, e)); q[p] = (int) fmin(7, fmax(-8, r)); }
                double er = 0;
                for (int c = 0; c < 8; ++c) {
                    double v = 0;
                    for (int p = 0; p < 3; ++p) v += S[s][3 * c + p] * q[p];
                    const double d = x[c] - v * NSLM_R32 * ldexp(1, e);
                    er += d * d;
                }
                if (er < best) { best = er; bs = s; be = e; memcpy(bq, q, sizeof bq); }
            }
        }
        // direct err(s) at the winner's (q, e), and the character expansion
        const double sc = NSLM_R32 * ldexp(1, be);
        memset(V, 0, sizeof V);
        for (int c = 0; c < 8; ++c) {
            double al[40] = {0};
            double beta = x[c] + 0.5 * sc * (bq[0] + bq[1] + bq[2]);
            for (int p = 0; p < 3; ++p) {
                const int k = 3 * c + p + 1;   // state index: bits k .. k + 15
                for (int i = 0; i < 16; ++i) al[k + i] += bq[p] * ldexp(1, i - 1);
            }
            // al[j] indexes stream bit j (state k bit i = stream bit k + i); k + i ranges 1 .. 39
            V[0] += beta * beta;
            for (int j = 0; j < 40; ++j) {
                if (al[j] == 0) continue;
                V[0] += sc * sc * al[j] * al[j];
                V[A[j]] += 2 * beta * sc * al[j];
                for (int k2 = j + 1; k2 < 40; ++k2)
                    if (al[k2] != 0) V[A[j] ^ A[k2]] += 2 * sc * sc * al[j] * al[k2];
            }
        }
        int terms = 0;
        for (int b2 = 0; b2 < N; ++b2) terms += V[b2] != 0;
        memcpy(D, V, sizeof D);
        fwht(D);   // D[s] = sum_b V[b] (-1)^<b,s>
        double md = 0, scale = 0, mn = INFINITY;
        int arg = 0;
        for (int s = 1; s < N; ++s) {
            double er = 0;
            for (int c = 0; c < 8; ++c) {
                double v = 0;
                for (int p = 0; p < 3; ++p) v += S[s][3 * c + p] * bq[p];
                const double d = x[c] - v * sc;
                er += d * d;
            }
            md = fmax(md, fabs(er - D[s]));
            scale = fmax(scale, fabs(er));
            if (D[s] < mn) { mn = D[s]; arg = s; }
        }
        worst = fmax(worst, md / scale);
        if (terms < minterms) minterms = terms;
        if (terms > maxterms) maxterms = terms;
        if (bi < 4)
            printf("block %d: best seed %5d err %.3e | expansion: %d terms, max |direct - expansion| / max err %.1e, argmin %5d\n", bi, bs,
                   best, terms, md / scale, arg);
    }
    printf("%d blocks: terms %d .. %d of 65536, worst relative mismatch %.1e\n", nblk, minterms, maxterms, worst);
    return 0;
}
