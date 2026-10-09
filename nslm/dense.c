// nslm/dense.c - nslm-dense: seed search over MoVA's dense 2-D tensors (attention, shared / dense MLPs, LM head,
// embedding), one file per tensor for nslm-mova-pack --seeds: P = 4 (.blk4, nslm-moe's format with one slice),
// P = 8 (.blk8: the same layout with 32-bit coefficient words; searchp.h) or P = 3 (.blk, nslm-moe's SEED4 format: the
// exponent code in each nibble word, no code stream).
//
//   nslm-dense --model DIR --xtx DIR --out DIR --mode aw|gptq [--codec p3|p4|p8] [--parts amhe] [--workers 4]
//              [--seeds 65535] [--damp 0.01] [--res out/res] [--only SUBSTR] [--shard I/N]
//
// --xtx: tools/mova_capture_dense.py's L<l>_<site>.bin (site 0 attention input, 1 o_proj input, 2 MLP input, 3 the MLP
// down_proj input; L<n_layers>_0 the LM head input).  aw: the search weighted by h = diag(H) / rows (as nslm-moe).
// gptq: column groups in order, each searched with A = T^T (T the inverse of the group's 8 x 8 block of
// U = chol(H^-1), nslm_gptq_factor), its error (W_B - Q_B) T fed forward: W[:, after] -= E U[B, after].  The
// embedding has no input and is searched unweighted.  Parts: a attention, m shared / dense MLPs, h LM head, e embedding,
// r routers (mlp.gate, v_router: for measuring seeded routers; not in the default parts).
// A finished .blk4 is skipped (resumable).
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "gptq.h"
#include "linalg.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"
#include "search.h"
#include "search4.h"
#include "search4_gpu.h"
#include "searchp.h"
#include "searchp_gpu.h"

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
static int g_gptq, g_layers, g_P;   // g_P: 3, 4 or 8
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

static int fwrite16(const uint32_t* c, size_t n, FILE* f) {   // P = 4: 16-bit coefficient words on disk
    uint16_t* h = malloc(2 * n);
    for (size_t k = 0; k < n; ++k) h[k] = (uint16_t) c[k];
    const int bad = fwrite(h, 2, n, f) != n;
    free(h);
    return bad;
}

// One tensor's search through either codec: rows x cols (A: ng x 64 or NULL, then sh), coefficients as 32-bit words.
static int gpu_search(Nslm4Gpu* g4, NslmPGpu* gp, const float* w, int rows, int cols, const float* sh, const float* A, int bias,
                      uint16_t* seed, uint32_t* coef, uint8_t* ec, char* err, int errlen) {
    if (g_P != 4) return nslmp_gpu_search(gp, g_P, w, rows, cols, sh, A, bias, &g_o, seed, coef, ec, NULL, err, errlen);
    const size_t nb = (size_t) rows * cols / 8;
    uint16_t* c16 = malloc(2 * nb);
    const int rc = A ? nslm4_gpu_search_a(g4, w, rows, cols, A, bias, &g_o, seed, c16, ec, NULL, err, errlen)
                     : nslm4_gpu_search(g4, w, rows, cols, sh, bias, &g_o, seed, c16, ec, NULL, err, errlen);
    for (size_t k = 0; k < nb; ++k) coef[k] = c16[k];
    free(c16);
    return rc;
}
static void decode_block(uint16_t s, uint32_t c, int e, uint16_t bf[8]) {
    if (g_P != 4) nslmp_decode_block(g_P, s, c, e, bf); else nslm4_decode_block(s, (uint16_t) c, e, bf);
}
typedef struct { Nslm4Gpu* g4; NslmPGpu* gp; } Gpus;
static int gptq_search(void* ctx, const float* w, int rows, int cols, const float* A, int bias, uint16_t* seed, uint32_t* coef,
                       uint8_t* ec, char* err, int errlen) {
    const Gpus* g = (const Gpus*) ctx;
    return gpu_search(g->g4, g->gp, w, rows, cols, NULL, A, bias, seed, coef, ec, err, errlen);
}

static int search_tensor(Nslm4Gpu* g4, NslmPGpu* gp, const MovaTensor* t) {
    char path[2048], base[160], err[512];
    snprintf(base, sizeof base, "%.*s", (int) (strlen(t->name) - 7), t->name);   // without ".weight"
    snprintf(path, sizeof path, "%s/%s.%s", g_out, base, g_P == 8 ? "blk8" : g_P == 4 ? "blk4" : "blk");
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
        rc = gpu_search(g4, gp, W0, R, C, sh, NULL, bias, seed, coef, ec, err, sizeof err);
        free(sh);
    } else {
        double* U = malloc(sizeof(double) * (size_t) C * C);
        double used = 0;
        if (nslm_gptq_factor(h, C, g_damp, U, &used)) { fprintf(stderr, "%s: H + damp not positive definite\n", base); return -1; }
        Gpus gs = {g4, gp};
        const NslmGptq q = {gptq_search, decode_block, &gs, 12};
        const double* Uc = U;
        rc = nslm_gptq(&q, 1, R, C, &W, &Uc, &bias, &seed, &coef, &ec, err, sizeof err);
        free(U);
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
    memcpy(hd.magic, g_P == 8 ? "NSLMBLK8" : g_P == 4 ? "NSLMBLK4" : "NSLMBLKM", 8);
    hd.rows = (uint32_t) R; hd.cols = (uint32_t) C; hd.n_experts = 1;
    hd.n_seeds = (uint32_t) g_o.n_seeds; hd.n_exp = (uint32_t) g_o.n_exp; hd.refit = (uint32_t) g_o.refit;
    memcpy(hd.exp_delta, g_o.exp_delta, sizeof hd.exp_delta);
    hd.seconds = now_s() - t0;
    const float rel = (float) sqrt(se / sw), wrel = (float) sqrt(wse / wsw);
    if (g_P == 3)   // SEED4 nibble words: the exponent code, then the 3 coefficients
        for (size_t k = 0; k < nb; ++k) coef[k] = ec[k] | coef[k] << 4;
    char tmp[2100];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f || fwrite(&hd, sizeof hd, 1, f) != 1 || fwrite(&bias, 4, 1, f) != 1 || fwrite(&rel, 4, 1, f) != 1 || fwrite(&wrel, 4, 1, f) != 1 ||
        fwrite(seed, 2, nb, f) != nb || (g_P == 8 ? fwrite(coef, 4, nb, f) != nb : fwrite16(coef, nb, f)) || (g_P != 3 && fwrite(ec, 1, nb, f) != nb) || fclose(f) ||
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
    snprintf(lib, sizeof lib, "%s/%s.metallib", g_res, g_P == 4 ? "search4" : "searchp");
    Nslm4Gpu* g4 = g_P != 4 ? NULL : nslm4_gpu_open(lib, err, sizeof err);
    NslmPGpu* gp = g_P != 4 ? nslmp_gpu_open(lib, err, sizeof err) : NULL;
    if (!g4 && !gp) { fprintf(stderr, "%s\n", err); atomic_store(&j->fail, 1); return NULL; }
    for (;;) {
        const int i = atomic_fetch_add(&j->next, 1);
        if (i >= j->n || atomic_load(&j->fail)) break;
        if (search_tensor(g4, gp, &j->list[i])) atomic_store(&j->fail, 1);
    }
    nslm4_gpu_close(g4);
    nslmp_gpu_close(gp);
    return NULL;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // _IOLBF with size 0 fails fast in MSVC's CRT
    const char* model = opt(argc, argv, "--model", NULL), *mode = opt(argc, argv, "--mode", NULL);
    g_xtx = opt(argc, argv, "--xtx", NULL); g_out = opt(argc, argv, "--out", NULL); g_res = opt(argc, argv, "--res", "out/res");
    const char* parts = opt(argc, argv, "--parts", "amhe"), *only = opt(argc, argv, "--only", NULL);
    const int workers = atoi(opt(argc, argv, "--workers", "4"));
    if (!model || !g_out || !mode || (strcmp(mode, "aw") && strcmp(mode, "gptq")) || (!g_xtx && strcmp(parts, "e")) || workers < 1 || workers > 32 ||
        has_flag(argc, argv, "-h")) {
        fprintf(stderr, "usage: nslm-dense --model DIR --xtx DIR --out DIR --mode aw|gptq [--codec p3|p4|p8] [--parts amhe] [--workers 4] [--seeds 65535] "
                        "[--damp 0.01] [--res out/res] [--only SUBSTR] [--shard I/N]\n");
        return 2;
    }
    g_gptq = !strcmp(mode, "gptq");
    const char* codec = opt(argc, argv, "--codec", "p4");
    if (strcmp(codec, "p3") && strcmp(codec, "p4") && strcmp(codec, "p8")) { fprintf(stderr, "--codec p3|p4|p8\n"); return 2; }
    g_P = codec[1] - '0';
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
        const int part = t->kind == MOVA_K_HEAD ? 'h' : t->kind == MOVA_K_EMBED ? 'e' : t->kind == MOVA_K_ROUTER ? 'r'
                       : t->kind == MOVA_K_LINEAR ? (strstr(t->name, "self_attn") ? 'a' : 'm') : 0;
        if (part && strchr(parts, part) && (!only || strstr(t->name, only))) sel[n++] = *t;
    }
    for (int i = 0; i < n; ++i)   // largest first, so the LM head and embedding do not finish last
        for (int k = i + 1; k < n; ++k)
            if ((double) sel[k].rows * sel[k].cols > (double) sel[i].rows * sel[i].cols) { MovaTensor x = sel[i]; sel[i] = sel[k]; sel[k] = x; }
    int si = 0, sn = 1;   // --shard I/N: tensors I, I + N, ... of the largest-first list (one process per GPU)
    if (opt(argc, argv, "--shard", NULL) && (sscanf(opt(argc, argv, "--shard", NULL), "%d/%d", &si, &sn) != 2 || sn < 1 || si < 0 || si >= sn)) {
        fprintf(stderr, "--shard I/N with 0 <= I < N\n");
        return 2;
    }
    int m = 0;
    for (int i = si; i < n; i += sn) sel[m++] = sel[i];
    n = m;
    printf("%d tensors, codec %s, mode %s, %d seeds, %d workers\n", n, codec, mode, g_o.n_seeds, workers);
    Jobs j = {sel, n, 0, 0};
    pthread_t th[32];
    for (int i = 0; i < workers; ++i) pthread_create(&th[i], NULL, worker, &j);
    for (int i = 0; i < workers; ++i) pthread_join(th[i], NULL);
    free(all); free(sel);
    mova_ckpt_close(g_ck);
    return atomic_load(&j.fail) ? 1 : 0;
}
