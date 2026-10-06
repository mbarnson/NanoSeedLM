// nslm/bits_probe.c - nslm-bits-probe (experiment tool): unweighted relative reconstruction error
// sum |w - w^|^2 / sum |w|^2 of SeedLM blocks at P = 3 (4.0 bpw), P = 3 with 5-bit coefficients (4.375 bpw) and P = 4
// (4.5 bpw), vs MLX affine Q4 g64 (4.5 bpw), on random 8-weight blocks of real MoVA routed-expert matrices.
//
//   nslm-bits-probe MODEL_DIR
//
// Seed search as the format: full 65535-seed budget, least squares, shared exponent e = floor(log2 max|t|) - (qbits-2),
// q = clamp(rne(t / 2^e)), errors from the Gram matrix; exponents e-1, e, e+1 tried.
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "lfsr.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"

#define NB 3072
typedef struct { int P, qmin, qmax, qbits; double U[65536][32]; } Fam;
static Fam* F;
static float* W;           // [NB][8]
static double* best;       // [NB]
static int g_nt = 16;

static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

static void* run(void* arg) {
    const int id = (int) (long) arg, P = F->P;
    for (int b = id; b < NB; b += g_nt) {
        const float* w = W + b * 8;
        double wn = 0;
        for (int c = 0; c < 8; ++c) wn += (double) w[c] * w[c];
        double bestv = wn;   // q = 0
        for (int s = 1; s < 65536; ++s) {
            const double* U = F->U[s];   // [c][p] row-major, centred states
            double G[16] = {0}, bv[4] = {0};
            for (int c = 0; c < 8; ++c)
                for (int p = 0; p < P; ++p) {
                    bv[p] += U[c * P + p] * w[c];
                    for (int r = 0; r < P; ++r) G[p * P + r] += U[c * P + p] * U[c * P + r];
                }
            // solve G t = bv (Gauss, P <= 4)
            double A[4][5];
            for (int i = 0; i < P; ++i) { for (int j = 0; j < P; ++j) A[i][j] = G[i * P + j]; A[i][P] = bv[i]; }
            int sing = 0;
            for (int i = 0; i < P && !sing; ++i) {
                int pv = i;
                for (int k = i + 1; k < P; ++k) if (fabs(A[k][i]) > fabs(A[pv][i])) pv = k;
                if (fabs(A[pv][i]) < 1e-12) { sing = 1; break; }
                if (pv != i) for (int j = 0; j <= P; ++j) { double tmp = A[i][j]; A[i][j] = A[pv][j]; A[pv][j] = tmp; }
                for (int k = 0; k < P; ++k) if (k != i) { const double f = A[k][i] / A[i][i]; for (int j = i; j <= P; ++j) A[k][j] -= f * A[i][j]; }
            }
            if (sing) continue;
            double t[4], tm = 0;
            for (int i = 0; i < P; ++i) { t[i] = A[i][P] / A[i][i]; tm = fmax(tm, fabs(t[i])); }
            if (tm == 0) continue;
            const int E = (int) floor(log2(tm)) - (F->qbits - 2);
            for (int de = -1; de <= 1; ++de) {
                const double sc = ldexp(1.0, E + de);
                double q[4];
                for (int i = 0; i < P; ++i) { double r = nearbyint(t[i] / sc); q[i] = r < F->qmin ? F->qmin : (r > F->qmax ? F->qmax : r); }
                double qb = 0, qgq = 0;
                for (int i = 0; i < P; ++i) { qb += q[i] * bv[i]; for (int j = 0; j < P; ++j) qgq += q[i] * G[i * P + j] * q[j]; }
                const double err = wn - 2 * sc * qb + sc * sc * qgq;
                if (err < bestv) bestv = err;
            }
        }
        best[b] = bestv;
    }
    return NULL;
}

static double seeds(int P, int qbits) {
    F->P = P; F->qbits = qbits; F->qmin = -(1 << (qbits - 1)); F->qmax = (1 << (qbits - 1)) - 1;
    uint16_t st[32];
    for (int s = 1; s < 65536; ++s) {
        lfsr_states((uint16_t) s, 8 * P, st);
        for (int c = 0; c < 8; ++c) for (int p = 0; p < P; ++p) F->U[s][c * P + p] = (double) st[P * c + p] - 32768.0;
    }
    pthread_t th[64];
    for (int i = 0; i < g_nt; ++i) pthread_create(&th[i], NULL, run, (void*) (long) i);
    for (int i = 0; i < g_nt; ++i) pthread_join(th[i], NULL);
    double e = 0, n = 0;
    for (int b = 0; b < NB; ++b) { e += best[b]; for (int c = 0; c < 8; ++c) n += (double) W[b * 8 + c] * W[b * 8 + c]; }
    return e / n;
}

int main(int argc, char** argv) {
    const char* model = argv[1];
    char err[512];
    MovaCfg cfg;
    if (mova_cfg_load(&cfg, model, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    MovaCkpt* ck = mova_ckpt_open(model, err, sizeof err);
    MovaTensor* all = NULL;
    const int na = mova_tensors(&cfg, &all);
    // tensors: (layer, proj) pairs, a few experts each
    const struct { int layer; const char* proj; } pick[] = {{8, "gate_proj"}, {20, "up_proj"}, {33, "down_proj"}, {44, "gate_proj"}};
    F = malloc(sizeof *F);
    W = malloc(sizeof(float) * NB * 8);
    best = malloc(sizeof(double) * NB);
    // affine errors over the same blocks' full rows: collect whole rows for affine (groups need 64 / 128 contiguous)
    double ae64 = 0, ae128 = 0, an = 0;
    int nb = 0;
    unsigned rs = 12345;
    for (int pi = 0; pi < 4; ++pi) {
        const MovaTensor* t = NULL;
        for (int i = 0; i < na; ++i) if (all[i].kind == MOVA_K_EXPERTS && all[i].layer == pick[pi].layer && !strcmp(all[i].proj, pick[pi].proj)) t = &all[i];
        if (!t) { fprintf(stderr, "no tensor\n"); return 1; }
        char nm[160];
        for (int ex = 0; ex < 4; ++ex) {
            const int e = (ex * 29 + pi * 7) % t->slices;
            const uint16_t* w = mova_ckpt_bf16(ck, mova_slice_name(t, e, nm, sizeof nm), t->rows, t->cols, err, sizeof err);
            if (!w) { fprintf(stderr, "%s\n", err); return 1; }
            for (int k = 0; k < NB / 16; ++k) {
                rs = rs * 1103515245u + 12345u;
                const int r = (int) ((rs >> 8) % (unsigned) t->rows);
                rs = rs * 1103515245u + 12345u;
                const int g = (int) ((rs >> 8) % (unsigned) (t->cols / 128));   // 128-wide group; seed block = first 8
                const uint16_t* row = w + (size_t) r * t->cols + (size_t) g * 128;
                for (int c = 0; c < 8; ++c) W[nb * 8 + c] = bf2f(row[c]);
                // affine over the whole 128 group (g128) and its first 64 (g64); error counted on the first 8 only
                uint32_t q[64]; uint16_t sc[2], bi[2], d[128];
                nslm_affine_quantize(row, 1, 128, 4, q, sc, bi);
                nslm_affine_dequantize(q, sc, bi, 1, 128, 4, d);
                for (int c = 0; c < 8; ++c) { const double x = bf2f(row[c]), y = bf2f(d[c]); ae128 += (x - y) * (x - y); }
                nslm_affine_quantize(row, 2, 64, 4, q, sc, bi);
                nslm_affine_dequantize(q, sc, bi, 2, 64, 4, d);
                for (int c = 0; c < 8; ++c) { const double x = bf2f(row[c]), y = bf2f(d[c]); ae64 += (x - y) * (x - y); an += x * x; }
                ++nb;
            }
        }
    }
    printf("blocks %d from 16 expert matrices (layers 8 / 20 / 33 / 44)\n", nb);
    (void) ae128;
    printf("affine Q4 g64  (4.50 bpw): rel err %.5f\n", ae64 / an);
    printf("seeds P=3 q4   (4.00 bpw): rel err %.5f\n", seeds(3, 4));
    printf("seeds P=3 q5   (4.375 bpw): rel err %.5f\n", seeds(3, 5));
    printf("seeds P=4 q4   (4.50 bpw): rel err %.5f\n", seeds(4, 4));
    (void) argc;
    return 0;
}
