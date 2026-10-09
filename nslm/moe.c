// nslm/moe.c - nslm-moe: activation-weighted SeedLM seed search over MoVA's routed and value expert matrices (one
// block file per (layer, projection)), and their expansion to BF16 safetensors.
//
//   nslm-moe --model DIR --act FILE --out DIR [--scope gu|gud|d|v|dv|all|mla] [--workers 4] [--n0 64] [--layers A-B]
//            [--res DIR] [--seeds 65535] [--weighting plain|w2] [--experts A-B] [--no-prune] [--p4]
//       writes OUT/L{l}_{proj}.blk (.blk4 with --p4), proj = gate_proj|up_proj|down_proj|v_experts, so all scopes can
//       share one directory; --res holds search.metallib / search4.metallib (default out/res); --weighting w2 weights
//       routed tokens by routing weight^2; --experts A-B profiles those experts only and writes nothing
//       (NSLM_MOE_VERBOSE=1: per-expert times)
//   nslm-moe --model DIR --act FILE --check L,PROJ,EXPERT,K    GPU vs the CPU search on K column groups of one expert
//   --scope mla --p4 --xtx DIR [--damp 0.01]: the MLA projections by GPTQ (nslm/gptq4.h) over their inputs' X^T X
//       (nslm-mova-mlacapture --xtx DIR: L{l}_{x,v,q,o}.xtx), H dampened by damp * mean(diag H); the heads of a per-head
//       map with one exponent bias search together
//   nslm-moe --expand --blk DIR --out DIR [--scope gu] [--p4]   OUT/L{l}_{proj}.safetensors: 'w' [E][rows][cols] BF16
//
// Each expert matrix (768 x 2560 gate/up, 2560 x 768 down, 1024 x 2560 value experts) is searched as its own tensor by
// the Metal search (search_gpu.h; search4_gpu.h with --p4): full seed budget, exponents {e, e-1, e+1}, the 3^P refit,
// its own exponent bias, weighted by h = mean squared input over the calibration tokens routed to that expert, shrunk
// toward the layer mean with n0 prior tokens (nslm_moe_blend; tools/mova_capture.py writes the sums).  Workers each
// own a GPU queue and take (layer, projection) jobs; results do not depend on the worker count.  A finished file is
// skipped (resumable).
//
// .blk (magic NSLMBLKM): MoeBlkHeader, int32 bias[E], float rel_err[E] (unweighted), float wrel_err[E] (h-weighted),
// uint16 seed[E][nb], uint16 nib[E][nb]  (nb = rows * cols / 8, row-major block order).
// .blk4 (magic NSLMBLK4, search4.h): as .blk, but nib = the 4 coefficients, then uint8 ecode[E][nb] (e - bias).
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "format.h"
#include "gptq4.h"
#include "linalg.h"
#include "moe.h"
#include "search.h"
#include "search_gpu.h"
#include "search4.h"
#include "search4_gpu.h"

typedef struct {
    char magic[8];   // "NSLMBLKM"
    uint32_t rows, cols, n_experts;
    uint32_t n_seeds, n_exp, refit;
    int32_t exp_delta[3];
    float n0;
    double seconds;
} MoeBlkHeader;

static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static int has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return 1;
    return 0;
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// ---- the projections ------------------------------------------------------------------------------------------------

// MLA projections (--scope mla, a TransMLA conversion; shapes from the --act file): kv_a_x [r][d], kv_a_v [r][kvd]
// (MoVA layers), k_rope_proj [128][d]; the per-head maps q_rope_mix [H][128][128], q_lat [H][r][128], v_up [H][128][r]
// are searched head by head, as the experts of one file.
enum { P_GATE, P_UP, P_DOWN, P_V, P_KAX, P_KAV, P_KR, P_QM, P_QL, P_VU, NPROJ };
static const char* kProj[NPROJ] = {"gate_proj", "up_proj", "down_proj", "v_experts", "kv_a_x", "kv_a_v", "k_rope_proj",
                                   "q_rope_mix", "q_lat", "v_up"};
static struct { int on, n_layer, d, kvd, qd, H; int64_t rows; int r[64]; double *x[64], *v[64], *q[64], *o[64]; } g_mla;
static int proj_rows(int l, int p) {
    return p == P_DOWN ? 2560 : p == P_V ? 1024 : p < P_KAX ? 768 : p == P_KAX || p == P_KAV || p == P_QL ? g_mla.r[l] : 128;
}
static int proj_cols(int l, int p) {
    return p == P_DOWN ? 768 : p < P_KAX ? 2560 : p == P_KAX || p == P_KR ? g_mla.d : p == P_KAV ? g_mla.kvd
           : p == P_VU ? g_mla.r[l] : 128;
}
static int proj_experts(int p) { return p == P_V ? 64 : p < P_KAX ? 100 : p >= P_QM ? g_mla.H : 1; }
static void tensor_name(char* s, int n, int l, int p, int e) {
    if (p >= P_KAX) snprintf(s, (size_t) n, "model.layers.%d.self_attn.mla.%s", l, kProj[p]);
    else if (p == P_V) snprintf(s, (size_t) n, "model.layers.%d.self_attn.v_experts.%d.weight", l, e);
    else snprintf(s, (size_t) n, "model.layers.%d.mlp.experts.%d.%s.weight", l, e, kProj[p]);
}
static int scope_mask(const char* s) {
    if (!strcmp(s, "gu")) return 1 << P_GATE | 1 << P_UP;
    if (!strcmp(s, "gud")) return 1 << P_GATE | 1 << P_UP | 1 << P_DOWN;
    if (!strcmp(s, "d")) return 1 << P_DOWN;
    if (!strcmp(s, "v")) return 1 << P_V;
    if (!strcmp(s, "dv")) return 1 << P_DOWN | 1 << P_V;
    if (!strcmp(s, "all")) return 1 << P_GATE | 1 << P_UP | 1 << P_DOWN | 1 << P_V;
    if (!strcmp(s, "mla")) return 1 << P_KAX | 1 << P_KAV | 1 << P_KR | 1 << P_QM | 1 << P_QL | 1 << P_VU;
    return 0;
}

// ---- the checkpoint: shards opened on demand ------------------------------------------------------------------------

static char* g_index;
static const char* g_model;
static struct { char name[128]; StFile st; int open; } g_shard[64];   // the shard file name (as file below)
static int g_nshard;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static char* read_all(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = (char*) malloc((size_t) n + 1);
    if (fread(b, 1, (size_t) n, f) != (size_t) n) { fclose(f); free(b); return NULL; }
    b[n] = 0;
    fclose(f);
    return b;
}

// BF16 tensor -> f32 (rows * cols; slice e of a 3-D tensor [slices][rows][cols] when slices > 1).  NULL on error.
static float* load_tensor(const char* name, int rows, int cols, int slices, int ex) {
    char file[128], err[512], path[2048];
    if (nslm_moe_index_lookup(g_index, name, file, sizeof file)) { fprintf(stderr, "%s: not in the index\n", name); return NULL; }
    pthread_mutex_lock(&g_mu);
    int k = 0;
    while (k < g_nshard && strcmp(g_shard[k].name, file)) ++k;
    if (k == g_nshard) {
        snprintf(path, sizeof path, "%s/%s", g_model, file);
        if (g_nshard == 64 || st_open_file(&g_shard[k].st, path, err, sizeof err)) {
            pthread_mutex_unlock(&g_mu);
            fprintf(stderr, "%s\n", err);
            return NULL;
        }
        snprintf(g_shard[k].name, sizeof g_shard[k].name, "%s", file);
        g_shard[k].open = 1;
        ++g_nshard;
    }
    const StFile* st = &g_shard[k].st;
    pthread_mutex_unlock(&g_mu);
    const StEntry* e = st_find(st, name);
    const StEntry* t = e;
    if (!t || strcmp(t->dtype, "BF16") || (slices > 1 ? t->ndim != 3 || t->shape[0] != slices || t->shape[1] != rows || t->shape[2] != cols
                                                      : t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols)) {
        fprintf(stderr, "%s: missing or unexpected shape\n", name);
        return NULL;
    }
    const uint16_t* b = (const uint16_t*) st_data(st, t) + (size_t) (slices > 1 ? ex : 0) * rows * cols;
    float* w = (float*) malloc(sizeof(float) * (size_t) rows * cols);
    for (size_t i = 0; i < (size_t) rows * cols; ++i) w[i] = nslm_bf2f(b[i]);
    return w;
}

// ---- calibration statistics (tools/mova_capture.py, "NSLMMOE1") -----------------------------------------------------

#define E_MAX 100
typedef struct {
    int64_t count[E_MAX];
    double* x2_all;   // [D]
    double* x2;       // [E][D]
    double* w2x2;     // [E][D]
    double* down;     // [E][F]
    int64_t vcount[64];
    double* v_all;    // [D]
    double* v;        // [EV][D]
} LayerAct;
static LayerAct g_act[48];
static int g_l0, g_nl;
static int64_t g_tokens;

// The MLA projections' statistics (harness/nslm-mova-mlacapture.c, "NSLMMLA1")
static int read_act_mla(FILE* f) {
    int32_t hd[5];
    if (fread(hd, 4, 5, f) != 5 || fread(&g_mla.rows, 8, 1, f) != 1 || hd[0] > 64 || g_mla.rows < 1) return -1;
    g_mla.n_layer = hd[0]; g_mla.d = hd[1]; g_mla.kvd = hd[2]; g_mla.qd = hd[3]; g_mla.H = hd[4];
    for (int l = 0; l < g_mla.n_layer; ++l) {
        int32_t r;
        if (fread(&r, 4, 1, f) != 1 || r < 8 || r > 4096) return -1;
        g_mla.r[l] = r;
        const size_t n = (size_t) (g_mla.d + g_mla.kvd + g_mla.qd + g_mla.H * r);
        double* b = (double*) malloc(sizeof(double) * n);
        if (fread(b, sizeof(double), n, f) != n) return -1;
        g_mla.x[l] = b; g_mla.v[l] = b + g_mla.d; g_mla.q[l] = b + g_mla.d + g_mla.kvd; g_mla.o[l] = b + g_mla.d + g_mla.kvd + g_mla.qd;
    }
    g_mla.on = 1;
    g_l0 = 0; g_nl = g_mla.n_layer; g_tokens = g_mla.rows;
    return 0;
}
static int read_act(const char* path) {
    FILE* f = fopen(path, "rb");
    char mg[8];
    int32_t hd[7];
    if (!f || fread(mg, 1, 8, f) != 8) return -1;
    if (!memcmp(mg, "NSLMMLA1", 8)) { const int rc = read_act_mla(f); fclose(f); return rc; }
    if (memcmp(mg, "NSLMMOE1", 8) || fread(hd, 4, 7, f) != 7 || fread(&g_tokens, 8, 1, f) != 1)
        return -1;
    g_l0 = hd[0]; g_nl = hd[1];
    const int E = hd[2], D = hd[3], F = hd[4], EV = hd[5];
    if (E != 100 || D != 2560 || F != 768 || EV != 64 || g_l0 + g_nl > 48) return -1;
    for (int i = 0; i < g_nl; ++i) {
        LayerAct* a = &g_act[g_l0 + i];
        a->x2_all = malloc(8 * (size_t) D); a->x2 = malloc(8 * (size_t) E * D); a->w2x2 = malloc(8 * (size_t) E * D);
        a->down = malloc(8 * (size_t) E * F); a->v_all = malloc(8 * (size_t) D); a->v = malloc(8 * (size_t) EV * D);
        if (fread(a->count, 8, (size_t) E, f) != (size_t) E || fread(a->x2_all, 8, (size_t) D, f) != (size_t) D ||
            fread(a->x2, 8, (size_t) E * D, f) != (size_t) E * D || fread(a->w2x2, 8, (size_t) E * D, f) != (size_t) E * D ||
            fread(a->down, 8, (size_t) E * F, f) != (size_t) E * F || fread(a->vcount, 8, (size_t) EV, f) != (size_t) EV ||
            fread(a->v_all, 8, (size_t) D, f) != (size_t) D || fread(a->v, 8, (size_t) EV * D, f) != (size_t) EV * D)
            return -1;
    }
    fclose(f);
    return 0;
}

// h (cols floats) of expert e of projection p in layer l, with the shrinkage prior.
static int g_w2;   // --weighting w2: weight each routed token by its routing weight squared
static void act_h(int l, int p, int e, double n0, float* h) {
    if (p >= P_KAX) {   // MLA: each input column's mean square over the calibration rows (every row reaches every layer)
        const int n = proj_cols(l, p);
        const double* sum = p == P_KAX || p == P_KR ? g_mla.x[l] : p == P_KAV ? g_mla.v[l]
                          : p == P_VU ? g_mla.o[l] + (size_t) e * g_mla.r[l] : g_mla.q[l] + (size_t) e * 128;
        for (int c = 0; c < n; ++c) h[c] = (float) (sum[c] / (double) g_mla.rows);
        return;
    }
    const LayerAct* a = &g_act[l];
    double prior[2560];
    if ((p == P_GATE || p == P_UP) && g_w2) {
        // sum w^2 x^2 over the expert's tokens; the prior is the layer mean scaled by the expert's mean w^2
        // (r = sum w^2 x^2 / sum x^2 over its channels), so h stays a per-token mean as in the plain weighting
        double sw = 0, sx = 0;
        for (int c = 0; c < 2560; ++c) { sw += a->w2x2[(size_t) e * 2560 + c]; sx += a->x2[(size_t) e * 2560 + c]; }
        const double r = sx > 0 ? sw / sx : 1.0;
        for (int c = 0; c < 2560; ++c) prior[c] = r * a->x2_all[c] / (double) g_tokens;
        nslm_moe_blend(a->w2x2 + (size_t) e * 2560, a->count[e], prior, n0, 2560, h);
    } else if (p == P_GATE || p == P_UP) {
        for (int c = 0; c < 2560; ++c) prior[c] = a->x2_all[c] / (double) g_tokens;
        nslm_moe_blend(a->x2 + (size_t) e * 2560, a->count[e], prior, n0, 2560, h);
    } else if (p == P_DOWN) {
        int64_t n = 0;
        for (int k = 0; k < 100; ++k) n += a->count[k];
        for (int c = 0; c < 768; ++c) {
            double s = 0;
            for (int k = 0; k < 100; ++k) s += a->down[(size_t) k * 768 + c];
            prior[c] = s / (double) (n > 0 ? n : 1);
        }
        nslm_moe_blend(a->down + (size_t) e * 768, a->count[e], prior, n0, 768, h);
    } else {
        for (int c = 0; c < 2560; ++c) prior[c] = a->v_all[c] / (double) g_tokens;
        nslm_moe_blend(a->v + (size_t) e * 2560, a->vcount[e], prior, n0, 2560, h);
    }
}

// ---- search jobs ----------------------------------------------------------------------------------------------------

typedef struct { int l, p; } JobItem;
typedef struct {
    JobItem* items;
    int n;
    atomic_int next;
    const char* out;
    const char* metallib;
    const SearchOpts* o;
    double n0;
    int prune;
    atomic_int failed;
    int e0, e1;      // --experts A-B: profile those experts only (nothing is written)
} Jobs;

static int g_p4;   // --p4: 4.5-bit blocks (nslm/search4.h), written as .blk4
static char g_lib4[1024];
static const char* g_xtx;   // --xtx DIR: GPTQ for the MLA projections
static double g_damp = 0.01;

// H / rows (double) of slice e of the X^T X file of projection p's input (nslm-mova-mlacapture --xtx), dim x dim
static double* read_xtx(int l, int p, int e, int dim, int slices) {
    char path[2048], mg[8];
    snprintf(path, sizeof path, "%s/L%d_%c.xtx", g_xtx, l, p == P_KAX || p == P_KR ? 'x' : p == P_KAV ? 'v' : p == P_VU ? 'o' : 'q');
    FILE* f = fopen(path, "rb");
    int32_t dm[2] = {0, 0};
    int64_t rows = 0;
    if (!f || fread(mg, 1, 8, f) != 8 || memcmp(mg, "NSLMXTX2", 8) || fread(dm, 4, 2, f) != 2 || fread(&rows, 8, 1, f) != 1 ||
        dm[0] != dim || dm[1] != (slices > 1 ? slices : 1) || rows < 1 ||
        fseek(f, (long) ((size_t) e * dim * dim * sizeof(float)), SEEK_CUR)) {
        fprintf(stderr, "%s: missing, or not %d x %d x %d\n", path, slices, dim, dim);
        if (f) fclose(f);
        return NULL;
    }
    float* hf = (float*) malloc(sizeof(float) * (size_t) dim * dim);
    double* h = (double*) malloc(sizeof(double) * (size_t) dim * dim);
    const int ok = fread(hf, sizeof(float), (size_t) dim * dim, f) == (size_t) dim * dim;
    fclose(f);
    for (size_t k = 0; ok && k < (size_t) dim * dim; ++k) h[k] = (double) hf[k] / (double) rows;
    free(hf);
    if (!ok) { fprintf(stderr, "%s: short\n", path); free(h); return NULL; }
    return h;
}
typedef struct { Nslm4Gpu* g; Search4Opts o; } GptqCtx;
static int gptq_search(void* ctx, const float* w, int rows, int cols, const float* A, int bias, uint16_t* seed, uint16_t* coef,
                       uint8_t* ecode, char* err, int errlen) {
    GptqCtx* c = (GptqCtx*) ctx;
    float* er = (float*) malloc(sizeof(float) * (size_t) rows * (cols / 8));
    const int rc = nslm4_gpu_search_a(c->g, w, rows, cols, A, bias, &c->o, seed, coef, ecode, er, err, errlen);
    free(er);
    return rc;
}
// GPTQ over every slice of MLA projection p of layer l (--xtx): seeds, coefficients, exponent codes and biases per slice
static int gptq_job(Nslm4Gpu* gpu4, const Search4Opts* o4, int l, int p, int rows, int cols, int E, size_t nb, int32_t* bias,
                    uint16_t* seed, uint16_t* nib, uint8_t* ec, float** w0, char* err, int errlen) {
    float** W = (float**) calloc((size_t) E, sizeof(float*));
    double** U = (double**) calloc((size_t) E, sizeof(double*));
    uint16_t** sp = (uint16_t**) malloc(sizeof(void*) * (size_t) E), **cp = (uint16_t**) malloc(sizeof(void*) * (size_t) E);
    uint8_t** ep = (uint8_t**) malloc(sizeof(void*) * (size_t) E);
    int* bs = (int*) malloc(sizeof(int) * (size_t) E);
    const int shared = p == P_KAX || p == P_KAV || p == P_KR;   // one input for the whole matrix
    int rc = 0;
    for (int e = 0; e < E && !rc; ++e) {
        W[e] = (float*) malloc(sizeof(float) * (size_t) rows * cols);
        memcpy(W[e], w0[e], sizeof(float) * (size_t) rows * cols);
        int64_t clamped = 0;
        bias[e] = bs[e] = nslm_choose_bias(w0[e], (int64_t) nb, &clamped);
        sp[e] = seed + (size_t) e * nb; cp[e] = nib + (size_t) e * nb; ep[e] = ec + (size_t) e * nb;
        double* h = read_xtx(l, p, shared ? 0 : e, cols, shared ? 1 : E);
        U[e] = (double*) malloc(sizeof(double) * (size_t) cols * cols);
        if (!h) { snprintf(err, (size_t) errlen, "no X^T X"); rc = -1; break; }
        if (nslm_gptq_factor(h, cols, g_damp, U[e], NULL)) { snprintf(err, (size_t) errlen, "H + damp not positive definite"); rc = -1; }
        free(h);
    }
    GptqCtx ctx = {gpu4, *o4};
    if (!rc) rc = nslm4_gptq(gptq_search, &ctx, E, rows, cols, W, (const double* const*) U, bs, sp, cp, ep, err, errlen);
    for (int e = 0; e < E; ++e) { free(W[e]); free(U[e]); }
    free(W); free(U); free(sp); free(cp); free(ep); free(bs);
    return rc;
}

static int blk_done(const char* path, const SearchOpts* o, double n0, MoeBlkHeader* h) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    const int ok = fread(h, sizeof *h, 1, f) == 1 && !memcmp(h->magic, g_p4 ? "NSLMBLK4" : "NSLMBLKM", 8) && h->n_seeds == (uint32_t) o->n_seeds &&
                   h->n_exp == (uint32_t) o->n_exp && h->refit == (uint32_t) o->refit && h->n0 == (float) n0;
    fclose(f);
    return ok;
}

static void* job_worker(void* arg) {
    Jobs* j = (Jobs*) arg;
    char err[512], path[2048], name[128];
    NslmGpu* gpu = g_p4 ? NULL : nslm_gpu_open(j->metallib, err, sizeof err);
    Nslm4Gpu* gpu4 = g_p4 ? nslm4_gpu_open(g_lib4, err, sizeof err) : NULL;
    if (!gpu && !gpu4) { fprintf(stderr, "gpu: %s\n", err); atomic_store(&j->failed, 1); return NULL; }
    for (;;) {
        const int k = atomic_fetch_add(&j->next, 1);
        if (k >= j->n || atomic_load(&j->failed)) break;
        const int l = j->items[k].l, p = j->items[k].p;
        const int rows = proj_rows(l, p), cols = proj_cols(l, p), E = proj_experts(p);
        const size_t nb = (size_t) rows * cols / NSLM_C;
        snprintf(path, sizeof path, "%s/L%d_%s.%s", j->out, l, kProj[p], g_p4 ? "blk4" : "blk");
        MoeBlkHeader h;
        if (blk_done(path, j->o, j->n0, &h)) { printf("L%-2d %-9s done (%.1f s)\n", l, kProj[p], h.seconds); continue; }
        memset(&h, 0, sizeof h);
        memcpy(h.magic, g_p4 ? "NSLMBLK4" : "NSLMBLKM", 8);
        h.rows = (uint32_t) rows; h.cols = (uint32_t) cols; h.n_experts = (uint32_t) E;
        h.n_seeds = (uint32_t) j->o->n_seeds; h.n_exp = (uint32_t) j->o->n_exp; h.refit = (uint32_t) j->o->refit;
        memcpy(h.exp_delta, j->o->exp_delta, sizeof h.exp_delta);
        h.n0 = (float) j->n0;
        int32_t* bias = (int32_t*) malloc(4 * (size_t) E);
        float* rel = (float*) malloc(4 * (size_t) E), *wrel = (float*) malloc(4 * (size_t) E);
        uint16_t* seed = (uint16_t*) malloc(2 * nb * (size_t) E), *nib = (uint16_t*) malloc(2 * nb * (size_t) E);
        uint8_t* ec = g_p4 ? (uint8_t*) malloc(nb * (size_t) E) : NULL;   // P = 4: exponent codes
        float* er = (float*) malloc(4 * nb), *hh = (float*) malloc(4 * (size_t) cols), *sh = (float*) malloc(4 * (size_t) cols);
        const double t0 = now_s();
        double gsearch = 0;
        if (g_xtx && p >= P_KAX) {   // GPTQ: every slice at once, then the same error report as below
            float** w0 = (float**) calloc((size_t) E, sizeof(float*));
            for (int e = 0; e < E && !atomic_load(&j->failed); ++e) {
                tensor_name(name, sizeof name, l, p, e);
                if (!(w0[e] = load_tensor(name, rows, cols, E > 1 ? E : 1, e))) atomic_store(&j->failed, 1);
            }
            const Search4Opts o4 = {j->o->n_seeds, j->o->n_exp, {j->o->exp_delta[0], j->o->exp_delta[1], j->o->exp_delta[2]}, j->o->refit};
            if (!atomic_load(&j->failed) && gptq_job(gpu4, &o4, l, p, rows, cols, E, nb, bias, seed, nib, ec, w0, err, sizeof err)) {
                fprintf(stderr, "L%d %s: %s\n", l, kProj[p], err);
                atomic_store(&j->failed, 1);
            }
            gsearch = now_s() - t0;
            for (int e = 0; e < E && !atomic_load(&j->failed); ++e) {
                act_h(l, p, e, j->n0, hh);
                double se = 0, sw = 0, wse = 0, wsw = 0;
                for (size_t b = 0; b < nb; ++b) {
                    uint16_t bf[NSLM_C];
                    nslm4_decode_block(seed[(size_t) e * nb + b], nib[(size_t) e * nb + b], bias[e] + ec[(size_t) e * nb + b], bf);
                    const int c0 = (int) (b % (size_t) (cols / NSLM_C)) * NSLM_C;
                    for (int c = 0; c < NSLM_C; ++c) {
                        const double x = w0[e][b * NSLM_C + c], d = x - nslm_bf2f(bf[c]);
                        se += d * d; sw += x * x;
                        wse += hh[c0 + c] * d * d; wsw += hh[c0 + c] * x * x;
                    }
                }
                rel[e] = (float) sqrt(se / sw);
                wrel[e] = (float) sqrt(wse / wsw);
            }
            for (int e = 0; e < E; ++e) free(w0[e]);
            free(w0);
        }
        for (int e = 0; e < E && !(g_xtx && p >= P_KAX); ++e) {
            if (j->e1 >= 0 && (e < j->e0 || e > j->e1)) continue;
            tensor_name(name, sizeof name, l, p, e);
            float* w = load_tensor(name, rows, cols, E > 1 && p >= P_KAX ? E : 1, e);
            if (!w) { atomic_store(&j->failed, 1); break; }
            int64_t clamped = 0;
            bias[e] = nslm_choose_bias(w, (int64_t) nb, &clamped);
            act_h(l, p, e, j->n0, hh);
            for (int c = 0; c < cols; ++c) sh[c] = sqrtf(hh[c] > 1e-12f ? hh[c] : 1e-12f);
            const double ts = now_s();
            const Search4Opts o4 = {j->o->n_seeds, j->o->n_exp, {j->o->exp_delta[0], j->o->exp_delta[1], j->o->exp_delta[2]}, j->o->refit};
            if (g_p4 ? nslm4_gpu_search(gpu4, w, rows, cols, sh, bias[e], &o4, seed + (size_t) e * nb, nib + (size_t) e * nb,
                                        ec + (size_t) e * nb, er, err, sizeof err)
                     : nslm_gpu_search(gpu, w, rows, cols, sh, bias[e], j->o, j->prune, seed + (size_t) e * nb, nib + (size_t) e * nb,
                                       er, err, sizeof err)) {
                fprintf(stderr, "%s: %s\n", name, err);
                atomic_store(&j->failed, 1);
                free(w);
                break;
            }
            gsearch += now_s() - ts;
            if (getenv("NSLM_MOE_VERBOSE")) {   // per-expert profile
                int64_t n = 0;
                double hmin = 1e30, hmax = 0;
                for (int c = 0; c < cols; ++c) { hmin = hh[c] < hmin ? hh[c] : hmin; hmax = hh[c] > hmax ? hh[c] : hmax; }
                n = p == P_V ? g_act[l].vcount[e] : g_act[l].count[e];
                printf("  L%d %s expert %3d: %7.2f s  routed %7lld  h min %.3e max %.3e  bias %d clamped %lld\n", l, kProj[p], e,
                       now_s() - ts, (long long) n, hmin, hmax, bias[e], (long long) clamped);
            }
            double se = 0, sw = 0, wse = 0, wsw = 0;
            for (size_t b = 0; b < nb; ++b) {
                uint16_t bf[NSLM_C];
                if (g_p4) nslm4_decode_block(seed[(size_t) e * nb + b], nib[(size_t) e * nb + b], bias[e] + ec[(size_t) e * nb + b], bf);
                else nslm_decode_block(seed[(size_t) e * nb + b], nib[(size_t) e * nb + b], bias[e], bf);
                const int c0 = (int) (b % (size_t) (cols / NSLM_C)) * NSLM_C;
                for (int c = 0; c < NSLM_C; ++c) {
                    const double x = w[b * NSLM_C + c], d = x - nslm_bf2f(bf[c]);
                    se += d * d; sw += x * x;
                    wse += hh[c0 + c] * d * d; wsw += hh[c0 + c] * x * x;
                }
            }
            rel[e] = (float) sqrt(se / sw);
            wrel[e] = (float) sqrt(wse / wsw);
            free(w);
        }
        if (atomic_load(&j->failed) || j->e1 >= 0) {
            free(bias); free(rel); free(wrel); free(seed); free(nib); free(ec); free(er); free(hh); free(sh);
            if (j->e1 >= 0 && !atomic_load(&j->failed)) continue;
            break;
        }
        h.seconds = now_s() - t0;
        char tmp[2100];
        snprintf(tmp, sizeof tmp, "%s.tmp", path);
        FILE* f = fopen(tmp, "wb");
        int ok = f && fwrite(&h, sizeof h, 1, f) == 1 && fwrite(bias, 4, (size_t) E, f) == (size_t) E && fwrite(rel, 4, (size_t) E, f) == (size_t) E &&
                 fwrite(wrel, 4, (size_t) E, f) == (size_t) E && fwrite(seed, 2, nb * (size_t) E, f) == nb * (size_t) E &&
                 fwrite(nib, 2, nb * (size_t) E, f) == nb * (size_t) E && (!ec || fwrite(ec, 1, nb * (size_t) E, f) == nb * (size_t) E);
        if (f && fclose(f)) ok = 0;
        if (ok && rename(tmp, path)) ok = 0;
        if (!ok) {
            fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno));
            remove(tmp);
            atomic_store(&j->failed, 1);
        }
        double mr = 0, mw = 0;
        for (int e = 0; e < E; ++e) { mr += rel[e]; mw += wrel[e]; }
        printf("L%-2d %-9s %3d x %4dx%-4d  rel_err %.5f  weighted %.5f  %6.1f s (gpu %6.1f s)  %.3e block-seeds/s\n", l, kProj[p], E,
               rows, cols, mr / E, mw / E, h.seconds, gsearch, (double) nb * E * j->o->n_seeds / gsearch);
        free(bias); free(rel); free(wrel); free(seed); free(nib); free(ec); free(er); free(hh); free(sh);
    }
    if (gpu) nslm_gpu_close(gpu);
    if (gpu4) nslm4_gpu_close(gpu4);
    return NULL;
}

// ---- the CPU check: K whole column groups of one expert, activation-weighted, against the GPU -----------------------

typedef struct { SeedTab* tab; const float* sh; int s0, s1; } TabJob;
static void* tab_worker(void* arg) {
    TabJob* t = (TabJob*) arg;
    nslm_seedtab_build_weighted(t->tab, t->sh, t->s0, t->s1);
    return NULL;
}

static int check(const char* spec, const SearchOpts* o, double n0, const char* metallib) {
    int l = 0, e = 0, k = 0;
    char pn[32];
    if (sscanf(spec, "%d,%31[^,],%d,%d", &l, pn, &e, &k) != 4) { fprintf(stderr, "--check L,PROJ,EXPERT,K\n"); return 2; }
    int p = 0;
    while (p < NPROJ && strcmp(kProj[p], pn)) ++p;
    if (p == NPROJ) { fprintf(stderr, "unknown projection %s\n", pn); return 2; }
    const int rows = proj_rows(l, p), cols = proj_cols(l, p), ng = cols / NSLM_C;
    char name[128], err[512];
    tensor_name(name, sizeof name, l, p, e);
    float* w = load_tensor(name, rows, cols, 1, 0);
    if (!w) return 2;
    int64_t clamped = 0;
    const int bias = nslm_choose_bias(w, (int64_t) rows * cols / NSLM_C, &clamped);
    float* hh = (float*) malloc(4 * (size_t) cols), *sh = (float*) malloc(4 * (size_t) cols);
    act_h(l, p, e, n0, hh);
    for (int c = 0; c < cols; ++c) sh[c] = sqrtf(hh[c] > 1e-12f ? hh[c] : 1e-12f);
    const size_t nb = (size_t) rows * cols / NSLM_C;
    uint16_t* gs = (uint16_t*) malloc(2 * nb), *gn = (uint16_t*) malloc(2 * nb);
    float* ge = (float*) malloc(4 * nb);
    NslmGpu* gpu = nslm_gpu_open(metallib, err, sizeof err);
    if (!gpu || nslm_gpu_search(gpu, w, rows, cols, sh, bias, o, 1, gs, gn, ge, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 2; }
    static SeedTab wt[65536];
    float* wg = (float*) malloc(4 * (size_t) rows * NSLM_C), *eg = (float*) malloc(4 * (size_t) rows);
    uint16_t* sg = (uint16_t*) malloc(2 * (size_t) rows), *ngb = (uint16_t*) malloc(2 * (size_t) rows);
    int64_t ncmp = 0, bad = 0;
    const double t0 = now_s();
    for (int i = 0; i < k; ++i) {
        const int g = (int) ((int64_t) i * ng / k + ng / (2 * k));
        TabJob tj[16];
        pthread_t th[16];
        for (int t = 0; t < 16; ++t) {
            tj[t] = (TabJob){wt, sh + g * NSLM_C, t * 65536 / 16, (t + 1) * 65536 / 16};
            pthread_create(&th[t], NULL, tab_worker, &tj[t]);
        }
        for (int t = 0; t < 16; ++t) pthread_join(th[t], NULL);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < NSLM_C; ++c) wg[r * NSLM_C + c] = w[(size_t) r * cols + g * NSLM_C + c] * sh[g * NSLM_C + c];
        nslm_search_vec(wt, wg, rows, bias, o, sg, ngb, eg, sh + g * NSLM_C);
        for (int r = 0; r < rows; ++r) {
            ++ncmp;
            const size_t kk = (size_t) r * ng + g;
            if (sg[r] != gs[kk] || ngb[r] != gn[kk]) {
                if (++bad <= 20) printf("  mismatch group %d row %d: cpu %u/%04x gpu %u/%04x\n", g, r, sg[r], ngb[r], gs[kk], gn[kk]);
            }
        }
    }
    printf("check %s: %lld blocks in %d column groups, %lld mismatches (%.4f%%), cpu %.1f s\n", name, (long long) ncmp, k,
           (long long) bad, 100.0 * (double) bad / (double) ncmp, now_s() - t0);
    nslm_gpu_close(gpu);
    return bad ? 1 : 0;
}

// ---- expansion ------------------------------------------------------------------------------------------------------

static int expand(const char* blk, const char* out, int mask) {
    mkdir(out, 0755);
    char path[2048], err[512];
    for (int l = 0; l < 48; ++l)
        for (int p = 0; p < NPROJ; ++p) {
            if (!(mask >> p & 1)) continue;
            snprintf(path, sizeof path, "%s/L%d_%s.%s", blk, l, kProj[p], g_p4 ? "blk4" : "blk");
            FILE* f = fopen(path, "rb");
            if (!f) continue;
            MoeBlkHeader h;
            if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, g_p4 ? "NSLMBLK4" : "NSLMBLKM", 8)) { fprintf(stderr, "bad %s\n", path); return 2; }
            const size_t E = h.n_experts, nb = (size_t) h.rows * h.cols / NSLM_C;
            int32_t* bias = (int32_t*) malloc(4 * E);
            float* tmp = (float*) malloc(8 * E);
            uint16_t* seed = (uint16_t*) malloc(2 * nb * E), *nib = (uint16_t*) malloc(2 * nb * E);
            uint16_t* w = (uint16_t*) malloc(2 * nb * NSLM_C * E);
            uint8_t* ec = g_p4 ? (uint8_t*) malloc(nb * E) : NULL;
            if (fread(bias, 4, E, f) != E || fread(tmp, 4, 2 * E, f) != 2 * E || fread(seed, 2, nb * E, f) != nb * E ||
                fread(nib, 2, nb * E, f) != nb * E || (ec && fread(ec, 1, nb * E, f) != nb * E)) { fprintf(stderr, "short %s\n", path); return 2; }
            fclose(f);
            for (size_t e = 0; e < E; ++e)
                for (size_t b = 0; b < nb; ++b) {
                    if (ec) nslm4_decode_block(seed[e * nb + b], nib[e * nb + b], bias[e] + ec[e * nb + b], w + (e * nb + b) * NSLM_C);
                    else nslm_decode_block(seed[e * nb + b], nib[e * nb + b], bias[e], w + (e * nb + b) * NSLM_C);
                }
            free(ec);
            snprintf(path, sizeof path, "%s/L%d_%s.safetensors", out, l, p == P_V ? "v_experts" : kProj[p]);
            if (nslm_st_write_bf16_3d(path, "w", (int) E, (int) h.rows, (int) h.cols, w, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 2; }
            printf("expanded L%d %s\n", l, kProj[p]);
            free(bias); free(tmp); free(seed); free(nib); free(w);
        }
    return 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // every line out at once (MSVC rejects _IOLBF with size 0: a fail-fast at start)
    const int mask = scope_mask(opt(argc, argv, "--scope", "gu"));
    if (!mask) { fprintf(stderr, "--scope gu|gud|d|v|dv|all|mla\n"); return 2; }
    g_p4 = has_flag(argc, argv, "--p4");
    if (has_flag(argc, argv, "--expand")) {
        const char* blk = opt(argc, argv, "--blk", NULL), *out = opt(argc, argv, "--out", NULL);
        if (!blk || !out) { fprintf(stderr, "--expand --blk DIR --out DIR\n"); return 2; }
        return expand(blk, out, mask);
    }
    g_model = opt(argc, argv, "--model", NULL);
    const char* actp = opt(argc, argv, "--act", NULL);
    if (!g_model || !actp) {
        fprintf(stderr, "usage: nslm-moe --model DIR --act FILE (--out DIR | --check L,PROJ,E,K) [--scope gu] [--workers 4] [--n0 64]\n");
        return 2;
    }
    char path[2048];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", g_model);
    if (!(g_index = read_all(path))) { fprintf(stderr, "cannot read %s\n", path); return 2; }
    if (read_act(actp)) { fprintf(stderr, "bad --act %s\n", actp); return 2; }
    SearchOpts o = {atoi(opt(argc, argv, "--seeds", "65535")), 3, {0, -1, 1}, 1};
    const double n0 = atof(opt(argc, argv, "--n0", "64"));
    g_w2 = !strcmp(opt(argc, argv, "--weighting", "plain"), "w2");
    char lib[1024];
    snprintf(lib, sizeof lib, "%s/search.metallib", opt(argc, argv, "--res", "out/res"));
    g_p4 = has_flag(argc, argv, "--p4");
    snprintf(g_lib4, sizeof g_lib4, "%s/search4.metallib", opt(argc, argv, "--res", "out/res"));
    printf("calibration: %lld tokens, layers %d-%d; weighting %s; prior n0 %.0f; budget %d seeds; exponents e+0 e-1 e+1; refit 1\n",
           (long long) g_tokens, g_l0, g_l0 + g_nl - 1, g_w2 ? "w2" : "plain", n0, o.n_seeds);
    if (opt(argc, argv, "--check", NULL)) return check(opt(argc, argv, "--check", NULL), &o, n0, lib);

    const char* out = opt(argc, argv, "--out", NULL);
    if (!out) { fprintf(stderr, "--out DIR\n"); return 2; }
    mkdir(out, 0755);
    int la = g_l0, lb = g_l0 + g_nl - 1;
    if (opt(argc, argv, "--layers", NULL)) sscanf(opt(argc, argv, "--layers", NULL), "%d-%d", &la, &lb);
    if ((mask >> P_KAX & 1) != g_mla.on) { fprintf(stderr, "--scope mla goes with an MLA capture (nslm-mova-mlacapture), and only it\n"); return 2; }
    g_xtx = opt(argc, argv, "--xtx", NULL);
    g_damp = atof(opt(argc, argv, "--damp", "0.01"));
    if (g_xtx && (!g_mla.on || !g_p4)) { fprintf(stderr, "--xtx goes with --scope mla --p4\n"); return 2; }
    if (g_xtx) printf("GPTQ over X^T X from %s, damp %.3g\n", g_xtx, g_damp);
    static JobItem items[64 * NPROJ];
    int n = 0;
    for (int l = la; l <= lb; ++l)
        for (int p = 0; p < NPROJ; ++p) {
            if (!(mask >> p & 1)) continue;
            char nm[128], file[128];
            tensor_name(nm, sizeof nm, l, p, 0);
            if (p >= P_KAX && nslm_moe_index_lookup(g_index, nm, file, sizeof file)) continue;   // kv_a_v: MoVA layers only
            items[n++] = (JobItem){l, p};
        }
    const int workers = atoi(opt(argc, argv, "--workers", "4"));
    Jobs j = {items, n, 0, out, lib, &o, n0, !has_flag(argc, argv, "--no-prune"), 0, 0, -1};
    if (opt(argc, argv, "--experts", NULL)) sscanf(opt(argc, argv, "--experts", NULL), "%d-%d", &j.e0, &j.e1);
    const double t0 = now_s();
    pthread_t th[16];
    for (int i = 0; i < workers && i < 16; ++i) pthread_create(&th[i], NULL, job_worker, &j);
    for (int i = 0; i < workers && i < 16; ++i) pthread_join(th[i], NULL);
    if (atomic_load(&j.failed)) return 1;
    printf("moe_total_s=%.1f (%d jobs, %d workers)\n", now_s() - t0, n, workers);
    return 0;
}
