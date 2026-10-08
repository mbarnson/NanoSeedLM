// nslm/dense.c - nslm-dense: seed search over MoVA's dense 2-D tensors (attention, shared / dense MLPs, LM head,
// embedding), one file per tensor for nslm-mova-pack --seeds: P = 4 (.blk4, nslm-moe's format with one slice) or
// P = 8 (.blk8: the same layout with 32-bit coefficient words; search8.h).
//
//   nslm-dense --model DIR --xtx DIR --out DIR --mode aw|gptq [--codec p4|p8] [--parts amhe] [--workers 4]
//              [--seeds 65535] [--damp 0.01] [--res out/res] [--only SUBSTR]
//
// --xtx: tools/mova_capture_dense.py's L<l>_<site>.bin (site 0 attention input, 1 o_proj input, 2 MLP input, 3 the MLP
// down_proj input; L<n_layers>_0 the LM head input).  aw: the search weighted by h = diag(H) / rows (as nslm-moe).
// gptq: column groups in order, each searched with A = T^T (T the inverse of the group's 8 x 8 block of
// U = chol(H^-1), nslm_gptq_factor), its error (W_B - Q_B) T fed forward: W[:, after] -= E U[B, after].  The
// embedding has no input and is searched unweighted.  Parts: a attention, m shared / dense MLPs, h LM head, e embedding.
// A finished .blk4 is skipped (resumable).
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "linalg.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"
#include "search.h"
#include "search4.h"
#include "search4_gpu.h"
#include "search8.h"
#include "search8_gpu.h"

typedef struct {
    char magic[8];
    uint32_t rows, cols, n_experts;
    uint32_t n_seeds, n_exp, refit;
    int32_t exp_delta[3];
    float n0;
    double seconds;
} MoeBlkHeader;   // nslm/moe.c

static int has_flag(int argc, char** argv, const char* n) { for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], n)) return 1; return 0; }
static const char* opt(int argc, char** argv, const char* n, const char* d) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], n)) return argv[i + 1];
    return d;
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static MovaCkpt* g_ck;
static const char* g_xtx, *g_out, *g_res;
static int g_gptq, g_layers, g_p8;
static double g_damp;
static Search4Opts g_o;

static int site_of(const MovaTensor* t, int* layer) {
    *layer = t->layer;
    if (t->kind == MOVA_K_HEAD) { *layer = g_layers; return 0; }
    if (t->kind == MOVA_K_EMBED) return -1;
    const char* n = t->name;
    if (strstr(n, "o_proj")) return 1;
    if (strstr(n, "self_attn")) return 0;   // q / k / v / gate
    return strstr(n, "down_proj") ? 3 : 2;
}

static double* read_h(int l, int s, int dim) {   // H / rows, double
    char p[2048];
    snprintf(p, sizeof p, "%s/L%d_%d.bin", g_xtx, l, s);
    FILE* f = fopen(p, "rb");
    char mg[8];
    int32_t d = 0;
    int64_t rows = 0;
    if (!f || fread(mg, 1, 8, f) != 8 || memcmp(mg, "NSLMXTX1", 8) || fread(&d, 4, 1, f) != 1 || fread(&rows, 8, 1, f) != 1 || d != dim) {
        fprintf(stderr, "bad %s\n", p);
        if (f) fclose(f);
        return NULL;
    }
    float* hf = malloc(sizeof(float) * (size_t) d * d);
    const size_t got = fread(hf, 4, (size_t) d * d, f);
    fclose(f);
    if (got != (size_t) d * d) { fprintf(stderr, "short %s\n", p); free(hf); return NULL; }
    double* h = malloc(sizeof(double) * (size_t) d * d);
    for (size_t k = 0; k < (size_t) d * d; ++k) h[k] = (double) hf[k] / (double) rows;
    free(hf);
    return h;
}

// W[r][c1:] -= sum_k E[r][k] U[c0 + k][c1:], rows split across threads
typedef struct { float* W; const float* E; const double* U; int cols, c0, ne, c1, r0, r1; } FbJob;
static void* fb_worker(void* arg) {
    FbJob* j = (FbJob*) arg;
    for (int r = j->r0; r < j->r1; ++r) {
        float* wr = j->W + (size_t) r * j->cols;
        const float* er = j->E + (size_t) r * 128;
        for (int k = 0; k < j->ne; ++k) {
            const float ek = er[k];
            const double* ur = j->U + (size_t) (j->c0 + k) * j->cols;
            for (int c = j->c1; c < j->cols; ++c) wr[c] -= ek * (float) ur[c];
        }
    }
    return NULL;
}
static void feedback(float* W, const float* E, const double* U, int R, int cols, int c0, int ne, int c1) {
    enum { NT = 12 };
    pthread_t th[NT];
    FbJob jb[NT];
    const int per = (R + NT - 1) / NT;
    for (int i = 0; i < NT; ++i) {
        jb[i] = (FbJob) {W, E, U, cols, c0, ne, c1, i * per < R ? i * per : R, (i + 1) * per < R ? (i + 1) * per : R};
        pthread_create(&th[i], NULL, fb_worker, &jb[i]);
    }
    for (int i = 0; i < NT; ++i) pthread_join(th[i], NULL);
}

static int fwrite16(const uint32_t* c, size_t n, FILE* f) {   // P = 4: 16-bit coefficient words on disk
    uint16_t* h = malloc(2 * n);
    for (size_t k = 0; k < n; ++k) h[k] = (uint16_t) c[k];
    const int bad = fwrite(h, 2, n, f) != n;
    free(h);
    return bad;
}

// One tensor's search through either codec: rows x cols (A: ng x 64 or NULL, then sh), coefficients as 32-bit words.
static int gpu_search(Nslm4Gpu* g4, Nslm8Gpu* g8, const float* w, int rows, int cols, const float* sh, const float* A, int bias,
                      uint16_t* seed, uint32_t* coef, uint8_t* ec, char* err, int errlen) {
    if (g_p8) return nslm8_gpu_search(g8, w, rows, cols, sh, A, bias, &g_o, seed, coef, ec, NULL, err, errlen);
    const size_t nb = (size_t) rows * cols / 8;
    uint16_t* c16 = malloc(2 * nb);
    const int rc = A ? nslm4_gpu_search_a(g4, w, rows, cols, A, bias, &g_o, seed, c16, ec, NULL, err, errlen)
                     : nslm4_gpu_search(g4, w, rows, cols, sh, bias, &g_o, seed, c16, ec, NULL, err, errlen);
    for (size_t k = 0; k < nb; ++k) coef[k] = c16[k];
    free(c16);
    return rc;
}
static void decode_block(uint16_t s, uint32_t c, int e, uint16_t bf[8]) {
    if (g_p8) nslm8_decode_block(s, c, e, bf); else nslm4_decode_block(s, (uint16_t) c, e, bf);
}

static int search_tensor(Nslm4Gpu* g4, Nslm8Gpu* g8, const MovaTensor* t) {
    char path[2048], base[160], err[512];
    snprintf(base, sizeof base, "%.*s", (int) (strlen(t->name) - 7), t->name);   // without ".weight"
    snprintf(path, sizeof path, "%s/%s.%s", g_out, base, g_p8 ? "blk8" : "blk4");
    struct stat st;
    if (!stat(path, &st)) { printf("%-52s done\n", base); return 0; }
    const int R = t->rows, C = t->cols, ng = C / 8;
    const size_t nb = (size_t) R * ng;
    const uint16_t* wb = mova_ckpt_bf16(g_ck, t->name, R, C, err, sizeof err);
    if (!wb) { fprintf(stderr, "%s\n", err); return -1; }
    float* W = malloc(sizeof(float) * (size_t) R * C), *W0 = malloc(sizeof(float) * (size_t) R * C);
    for (size_t k = 0; k < (size_t) R * C; ++k) W[k] = W0[k] = nslm4_bf2f(wb[k]);
    int64_t clamped = 0;
    const int32_t bias = nslm_choose_bias(W0, (int64_t) nb, &clamped);
    uint16_t* seed = malloc(2 * nb);
    uint32_t* coef = malloc(4 * nb);
    uint8_t* ec = malloc(nb);
    int layer = 0;
    const int site = site_of(t, &layer);
    double* h = site >= 0 ? read_h(layer, site, C) : NULL;
    if (site >= 0 && !h) return -1;
    const double t0 = now_s();
    int rc = 0;
    if (!g_gptq || !h) {   // aw, or the unweighted embedding
        float* sh = NULL;
        if (h) {
            sh = malloc(sizeof(float) * C);
            for (int c = 0; c < C; ++c) { const double v = h[(size_t) c * C + c]; sh[c] = sqrtf(v > 1e-12 ? (float) v : 1e-12f); }
        }
        rc = gpu_search(g4, g8, W0, R, C, sh, NULL, bias, seed, coef, ec, err, sizeof err);
        free(sh);
    } else {
        double* U = malloc(sizeof(double) * (size_t) C * C);
        double used = 0;
        if (nslm_gptq_factor(h, C, g_damp, U, &used)) { fprintf(stderr, "%s: H + damp not positive definite\n", base); return -1; }
        float* w8 = malloc(sizeof(float) * (size_t) R * 8), *E = malloc(sizeof(float) * (size_t) R * 128);
        uint16_t* sg = malloc(2 * (size_t) R);
        uint32_t* cg = malloc(4 * (size_t) R);
        uint8_t* eg = malloc((size_t) R);
        for (int bb = 0; bb < ng && !rc; bb += 16) {   // 128-column batches (lazy feedback)
            const int b1 = bb + 16 < ng ? bb + 16 : ng, c0 = bb * 8, c1 = b1 * 8;
            for (int b = bb; b < b1 && !rc; ++b) {
                const int cb = b * 8;
                double T[64];
                float A[64];
                nslm_upper8_inverse(U + (size_t) cb * C + cb, C, T);
                for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) A[i * 8 + j] = (float) T[j * 8 + i];
                for (int r = 0; r < R; ++r) memcpy(w8 + (size_t) r * 8, W + (size_t) r * C + cb, sizeof(float) * 8);
                if ((rc = gpu_search(g4, g8, w8, R, 8, NULL, A, bias, sg, cg, eg, err, sizeof err))) break;
                for (int r = 0; r < R; ++r) {
                    seed[(size_t) r * ng + b] = sg[r];
                    coef[(size_t) r * ng + b] = cg[r];
                    ec[(size_t) r * ng + b] = eg[r];
                    uint16_t bf[8];
                    decode_block(sg[r], cg[r], bias + eg[r], bf);
                    double d[8];
                    for (int c = 0; c < 8; ++c) d[c] = (double) w8[(size_t) r * 8 + c] - nslm4_bf2f(bf[c]);
                    float* er = E + (size_t) r * 128 + (cb - c0);
                    for (int k = 0; k < 8; ++k) {   // E = (W_B - Q_B) T
                        double a = 0;
                        for (int c = 0; c <= k; ++c) a += d[c] * T[c * 8 + k];
                        er[k] = (float) a;
                    }
                }
                // within the batch: the next groups of this batch take this group's error
                if (cb + 8 < c1) {
                    for (int r = 0; r < R; ++r) {
                        float* wr = W + (size_t) r * C;
                        const float* er = E + (size_t) r * 128 + (cb - c0);
                        for (int k = 0; k < 8; ++k) {
                            const float ek = er[k];
                            const double* ur = U + (size_t) (cb + k) * C;
                            for (int c = cb + 8; c < c1; ++c) wr[c] -= ek * (float) ur[c];
                        }
                    }
                }
            }
            if (!rc && c1 < C) feedback(W, E, U, R, C, c0, c1 - c0, c1);
        }
        free(U); free(w8); free(E); free(sg); free(cg); free(eg);
    }
    if (rc) { fprintf(stderr, "%s: %s\n", base, err); return -1; }
    double se = 0, sw = 0, wse = 0, wsw = 0;
    for (size_t k = 0; k < nb; ++k) {
        uint16_t bf[8];
        decode_block(seed[k], coef[k], bias + ec[k], bf);
        const int c0 = (int) (k % (size_t) ng) * 8;
        for (int c = 0; c < 8; ++c) {
            const double x = W0[k * 8 + c], d = x - nslm4_bf2f(bf[c]), hc = h ? h[(size_t) (c0 + c) * C + c0 + c] : 1.0;
            se += d * d; sw += x * x; wse += hc * d * d; wsw += hc * x * x;
        }
    }
    MoeBlkHeader hd;
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, g_p8 ? "NSLMBLK8" : "NSLMBLK4", 8);
    hd.rows = (uint32_t) R; hd.cols = (uint32_t) C; hd.n_experts = 1;
    hd.n_seeds = (uint32_t) g_o.n_seeds; hd.n_exp = (uint32_t) g_o.n_exp; hd.refit = (uint32_t) g_o.refit;
    memcpy(hd.exp_delta, g_o.exp_delta, sizeof hd.exp_delta);
    hd.seconds = now_s() - t0;
    const float rel = (float) sqrt(se / sw), wrel = (float) sqrt(wse / wsw);
    char tmp[2100];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f || fwrite(&hd, sizeof hd, 1, f) != 1 || fwrite(&bias, 4, 1, f) != 1 || fwrite(&rel, 4, 1, f) != 1 || fwrite(&wrel, 4, 1, f) != 1 ||
        fwrite(seed, 2, nb, f) != nb || (g_p8 ? fwrite(coef, 4, nb, f) != nb : fwrite16(coef, nb, f)) || fwrite(ec, 1, nb, f) != nb || fclose(f) ||
        rename(tmp, path)) {
        fprintf(stderr, "cannot write %s\n", path);
        return -1;
    }
    printf("%-52s %6dx%-5d %s  rel_err %.5f  weighted %.5f  %7.1f s\n", base, R, C, !h ? "plain" : g_gptq ? "gptq " : "aw   ", rel, wrel, hd.seconds);
    free(W); free(W0); free(seed); free(coef); free(ec); free(h);
    return 0;
}

typedef struct { const MovaTensor* list; int n; atomic_int next, fail; } Jobs;
static void* worker(void* arg) {
    Jobs* j = (Jobs*) arg;
    char err[512], lib[2048];
    snprintf(lib, sizeof lib, "%s/%s.metallib", g_res, g_p8 ? "search8" : "search4");
    Nslm4Gpu* g4 = g_p8 ? NULL : nslm4_gpu_open(lib, err, sizeof err);
    Nslm8Gpu* g8 = g_p8 ? nslm8_gpu_open(lib, err, sizeof err) : NULL;
    if (!g4 && !g8) { fprintf(stderr, "%s\n", err); atomic_store(&j->fail, 1); return NULL; }
    for (;;) {
        const int i = atomic_fetch_add(&j->next, 1);
        if (i >= j->n || atomic_load(&j->fail)) break;
        if (search_tensor(g4, g8, &j->list[i])) atomic_store(&j->fail, 1);
    }
    nslm4_gpu_close(g4);
    nslm8_gpu_close(g8);
    return NULL;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char* model = opt(argc, argv, "--model", NULL), *mode = opt(argc, argv, "--mode", NULL);
    g_xtx = opt(argc, argv, "--xtx", NULL); g_out = opt(argc, argv, "--out", NULL); g_res = opt(argc, argv, "--res", "out/res");
    const char* parts = opt(argc, argv, "--parts", "amhe"), *only = opt(argc, argv, "--only", NULL);
    const int workers = atoi(opt(argc, argv, "--workers", "4"));
    if (!model || !g_out || !mode || (strcmp(mode, "aw") && strcmp(mode, "gptq")) || (!g_xtx && strcmp(parts, "e")) || workers < 1 || workers > 32 ||
        has_flag(argc, argv, "-h")) {
        fprintf(stderr, "usage: nslm-dense --model DIR --xtx DIR --out DIR --mode aw|gptq [--codec p4|p8] [--parts amhe] [--workers 4] [--seeds 65535] "
                        "[--damp 0.01] [--res out/res] [--only SUBSTR]\n");
        return 2;
    }
    g_gptq = !strcmp(mode, "gptq");
    const char* codec = opt(argc, argv, "--codec", "p4");
    if (strcmp(codec, "p4") && strcmp(codec, "p8")) { fprintf(stderr, "--codec p4|p8\n"); return 2; }
    g_p8 = !strcmp(codec, "p8");
    g_damp = atof(opt(argc, argv, "--damp", "0.01"));
    g_o = (Search4Opts) {atoi(opt(argc, argv, "--seeds", "65535")), 3, {0, -1, 1}, 1};
    char err[512];
    MovaCfg cfg;
    if (mova_cfg_load(&cfg, model, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 2; }
    g_layers = cfg.n_layer;
    if (!(g_ck = mova_ckpt_open(model, err, sizeof err))) { fprintf(stderr, "%s\n", err); return 2; }
    mkdir(g_out, 0755);
    MovaTensor* all = NULL, *sel = NULL;
    const int na = mova_tensors(&cfg, &all);
    sel = malloc(sizeof *sel * (size_t) na);
    int n = 0;
    for (int i = 0; i < na; ++i) {
        const MovaTensor* t = &all[i];
        const int part = t->kind == MOVA_K_HEAD ? 'h' : t->kind == MOVA_K_EMBED ? 'e' : t->kind == MOVA_K_LINEAR ? (strstr(t->name, "self_attn") ? 'a' : 'm') : 0;
        if (part && strchr(parts, part) && (!only || strstr(t->name, only))) sel[n++] = *t;
    }
    for (int i = 0; i < n; ++i)   // largest first, so the LM head and embedding do not finish last
        for (int k = i + 1; k < n; ++k)
            if ((double) sel[k].rows * sel[k].cols > (double) sel[i].rows * sel[i].cols) { MovaTensor x = sel[i]; sel[i] = sel[k]; sel[k] = x; }
    printf("%d tensors, codec %s, mode %s, %d seeds, %d workers\n", n, codec, mode, g_o.n_seeds, workers);
    Jobs j = {sel, n, 0, 0};
    pthread_t th[32];
    for (int i = 0; i < workers; ++i) pthread_create(&th[i], NULL, worker, &j);
    for (int i = 0; i < workers; ++i) pthread_join(th[i], NULL);
    free(all); free(sel);
    mova_ckpt_close(g_ck);
    return atomic_load(&j.fail) ? 1 : 0;
}
