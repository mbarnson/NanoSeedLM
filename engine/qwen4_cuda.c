// engine/qwen4_cuda.c - the qwen4 CUDA engine (engine/qwen4_eng.h): the steps of nslm/qwen4_ref.h on the matrix
// kernels of engine/kernels_cuda.h and the qwen4 kernels of engine/kernels_qwen4.h.
#include "qwen4_eng.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "kernels_cuda.h"
#include "kernels_qwen4.h"
#include "lfsr.h"
#include "model_st.h"
#include "qwen4_ops.h"
#include "rows.h"

#define Q4E_NEV 8192   // Q4E_PROFILE events of one forward

#define MAXT Q4E_MAX_ROWS
#define XR Q4E_PASS_ROWS

typedef struct {   // a weight: its format and shape, slice 0, and a device table of all its slices
    int fmt, slices, rows, cols;
    WSlice w;
    const WSlice* tab;
} W;

typedef struct {
    W hc_down[2], hc_up[2], hc_inj[2];
    float* hc_norm[2];
    W qkv, z, b, a, out;                        // Gated DeltaNet
    Q4GdnDev gw;
    float *conv, *S;
    W q, k, v, o, ix;                           // QSA attention
    float *qn, *kn, *iqn, *ikn, *K, *V, *raw, *blk;
    W gate, sg, sh_g, sh_u, sh_d, eg, eu, ed;   // MoE
    W key, val;                                 // PLE
    float *nk, *nq, *nc, *pw, *hist;
} Layer;

struct Q4Eng {
    Qwen4Cfg c;
    NsModel nm;
    NsRows rows;
    Qwen4Ngram ng[QWEN4_MAX_LAYERS];
    PleTable pt;
    const PleTable* table;   // the GGUF PLE table, or NULL (the folder's)
    Layer L[QWEN4_MAX_LAYERS];
    W embed, head, mix_down, mix_up;
    float* mix_norm;
    uint32_t *g24, *g32;
    cudaStream_t s;
    int max_pos, pos, host_experts, cache, stream;
    CachePool pool;   // the expert cache (cache = 1)
    uint64_t ulen[3][4];   // bytes of each expert stream in a unit
    uint8_t* stage;        // stream = 1: pinned staging of the units one admit brings
    int32_t* hjobs;        // stream = 1: the admit's jobs, read back (pinned)
    const NsTensor* et[QWEN4_MAX_LAYERS][3];   // the layers' gate, up and down experts
    int32_t* ids;   // the context (host), for the PLE n-grams
    float *Hall;   // the hidden streams of a pass, XR rows; H: the current chunk's
    float *x, *H, *normed, *lo, *up, *inj, *br, *qp, *kk, *vv, *o, *ixo, *sc, *emb, *emb_h, *keys, *qs, *vals, *gated;
    float *lg, *wt, *A, *Dx, *sh, *sgv, *logits, *y, *cv, *qkvb, *zb, *bb, *ab;
    int32_t *ind, *sel, *done, *nsel, *zero, *perm, *ntiles;
    // A chunk's input: its slot of hin (pinned) = {pos, ids[T]}, copied to din on the device first.  The kernels read the position
    // from din, so a CUDA graph of T decode rows serves every position.  emb_h (PLE rows) and hlog (logits) are pinned too.
    int32_t *hin, *din;
    float* hlog;
    int nb, nple, cap, no_graph;   // QSA block slots; PLE layers; capturing a graph; NSLM_NO_GRAPH
    int ple_threads;               // Q4E_PLE_THREADS: threads for the PLE rows of 64 rows or more
    double host_ms[2];             // Q4E_PROFILE: host time of the PLE prefetch and the PLE rows
    cudaGraphExec_t graph[2][MV_MAXT + 1];   // [with logits][rows]
    FILE* routes;     // Q4E_ROUTES=FILE: each forward's int32 T, then its routed experts [n_layer][T][top_k] (tools/qwen4_cache_sim.py)
    int32_t *droute, *hroute;
    float *Gb, *Ub;   // prompt rows: the gate and up outputs of the grouped GEMMs
    MmTile* tiles;
    int prof, nev;                     // Q4E_PROFILE=1: CUDA events between the stages of a forward
    cudaEvent_t ev[Q4E_NEV];   // a pass of 8,192 rows marks about 2,700 stages
    int evstage[Q4E_NEV];
    double stage_ms[16];
    void** dev;   // allocations, freed at close
    int ndev, cdev;
    void** host;
    int nhost, chost;
    char err[256];
};

// the stages of a forward, for Q4E_PROFILE
enum { ST_EMBED, ST_PLE, ST_HC, ST_PROJ, ST_ATTN, ST_GDN, ST_ROUTE, ST_EXPERT, ST_HEAD, ST_N };   // PROJ: attention and DeltaNet projections
static const char* const kStage[ST_N] = {"embed", "ple", "hyper-connections", "projections", "attention",
                                         "gated deltanet", "router + cache", "experts", "head"};

static int fail(Q4Eng* e, const char* fmt, ...) {
    if (!e->err[0]) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(e->err, sizeof e->err, fmt, ap);
        va_end(ap);
    }
    return -1;
}
static void track(void*** v, int* n, int* cap, void* p) {
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 256; *v = (void**) realloc(*v, sizeof(void*) * (size_t) *cap); }
    (*v)[(*n)++] = p;
}
// a zeroed device array, or a device copy of p
static void* dmem(Q4Eng* e, const void* p, size_t n) {
    void* d = NULL;
    if (cudaMalloc(&d, n ? n : 16) != cudaSuccess) { fail(e, "out of VRAM (%zu bytes)", n); return NULL; }
    track(&e->dev, &e->ndev, &e->cdev, d);
    if (p) cudaMemcpy(d, p, n, cudaMemcpyHostToDevice);
    else cudaMemset(d, 0, n ? n : 16);
    return d;
}
// n bytes of pinned host memory, freed at close
static void* pin(Q4Eng* e, size_t n) {
    void* h = NULL;
    if (cudaHostAlloc(&h, n ? n : 16, cudaHostAllocDefault) != cudaSuccess) { fail(e, "cannot pin %zu bytes of host memory", n); return NULL; }
    track(&e->host, &e->nhost, &e->chost, h);
    return h;
}
// a copy of p in mapped host memory; its device address
static void* hmem(Q4Eng* e, const void* p, size_t n) {
    void *h = NULL, *d = NULL;
    if (cudaHostAlloc(&h, n ? n : 16, cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
        fail(e, "cannot map %zu bytes of host memory", n);
        return NULL;
    }
    track(&e->host, &e->nhost, &e->chost, h);
    memcpy(h, p, n);
    return d;
}
#define NAME(buf, fmt) do { va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap); } while (0)

// The tensor named by fmt in VRAM, or (host) in mapped host memory
static void load_w(Q4Eng* e, W* w, int host, const char* fmt, ...) {
    char name[192];
    NAME(name, fmt);
    const NsTensor* t = ns_find(&e->nm, name);
    if (!t || t->enc == NS_SEED6P8) { fail(e, "%s: %s", name, t ? "P = 8 seeds not supported" : "missing"); return; }
    if ((t->enc == NS_Q4 || t->enc == NS_Q8) && t->group != 32 && t->group != 64) { fail(e, "%s: affine group %d not supported", name, t->group); return; }
    const int seed = t->enc == NS_SEED4 || t->enc == NS_SEED4P4;
    const uint8_t* p[4] = {NULL, NULL, NULL, NULL};
    for (int s = 0; s < 4; ++s)
        if (t->s[s].len && !(seed && s == 2)) p[s] = (const uint8_t*) (host ? hmem(e, t->s[s].p, t->s[s].len) : dmem(e, t->s[s].p, t->s[s].len));
    WSlice* tab = (WSlice*) calloc((size_t) t->slices, sizeof(WSlice));
    for (int i = 0; i < t->slices; ++i) {
        for (int s = 0; s < 4; ++s) if (p[s]) tab[i].p[s] = p[s] + t->s[s].len / (uint64_t) t->slices * (uint64_t) i;
        if (seed) memcpy(&tab[i].eb, t->s[2].p + 4 * (size_t) i, 4);
        if (t->enc == NS_Q4 || t->enc == NS_Q8) tab[i].gs = t->group == 32 ? 5 : 6;
    }
    *w = (W) {t->enc, t->slices, t->rows, t->cols, tab[0], (const WSlice*) dmem(e, tab, sizeof(WSlice) * (size_t) t->slices)};
    free(tab);
}
// A BF16 vector of n values (any shape) as f32 in VRAM
static float* load_v(Q4Eng* e, int64_t n, const char* fmt, ...) {
    char name[192];
    NAME(name, fmt);
    const NsTensor* t = ns_find(&e->nm, name);
    if (!t || t->enc != NS_BF16 || (int64_t) t->slices * t->rows * t->cols != n) { fail(e, "%s: missing, or not BF16 with %lld values", name, (long long) n); return NULL; }
    float* v = (float*) malloc(sizeof(float) * (size_t) n);
    for (int64_t i = 0; i < n; ++i) { const uint32_t u = (uint32_t) ((const uint16_t*) t->s[0].p)[i] << 16; memcpy(&v[i], &u, 4); }
    float* d = (float*) dmem(e, v, sizeof(float) * (size_t) n);
    free(v);
    return d;
}

static const uint32_t* G(const Q4Eng* e, const W* w) { return w->fmt == MF_SEED4P4 ? e->g32 : e->g24; }
// y[T][rows] = W x[T][cols]: the matvec for up to MV_MAXT rows, the tensor-core GEMM (BF16 inputs) for more
static void mv(Q4Eng* e, const W* w, const float* x, float* y, int T) {
    if (T <= MV_MAXT) kc_mv(e->s, w->fmt, w->w, w->cols, w->rows, x, w->cols, y, w->rows, T, 0, G(e, w));
    else kc_mm(e->s, w->fmt, w->w, w->cols, w->rows, x, w->cols, y, w->rows, T, 0, G(e, w));
}

static void load_layer(Q4Eng* e, int l) {
    const Qwen4Cfg* c = &e->c;
    Layer* L = &e->L[l];
    const int HD = c->hc * c->d, cd = 2 * c->lin_hk * c->lin_dk + c->lin_hv * c->lin_dv;
    const char* p = "language_model.model.layers";
    for (int b = 0; b < 2; ++b) {
        const char* hc = b ? "mlp_hyper_connection" : "attn_hyper_connection";
        L->hc_norm[b] = load_v(e, HD, "%s.%d.%s.hc_norm.weight", p, l, hc);
        load_w(e, &L->hc_down[b], 0, "%s.%d.%s.input_mix_weight_down.weight", p, l, hc);
        load_w(e, &L->hc_up[b], 0, "%s.%d.%s.input_mix_weight_up.weight", p, l, hc);
        load_w(e, &L->hc_inj[b], 0, "%s.%d.%s.block_inject_weight.weight", p, l, hc);
    }
    if (c->attn[l]) {
        const int nb = e->max_pos / c->ix_ratio + 1;
        load_w(e, &L->q, 0, "%s.%d.self_attn.q_proj.weight", p, l);
        load_w(e, &L->k, 0, "%s.%d.self_attn.k_proj.weight", p, l);
        load_w(e, &L->v, 0, "%s.%d.self_attn.v_proj.weight", p, l);
        load_w(e, &L->o, 0, "%s.%d.self_attn.o_proj.weight", p, l);
        load_w(e, &L->ix, 0, "%s.%d.self_attn.indexer.index_qk_proj.weight", p, l);
        L->qn = load_v(e, c->head_dim, "%s.%d.self_attn.q_norm.weight", p, l);
        L->kn = load_v(e, c->head_dim, "%s.%d.self_attn.k_norm.weight", p, l);
        L->iqn = load_v(e, c->ix_dim, "%s.%d.self_attn.indexer.q_layernorm.weight", p, l);
        L->ikn = load_v(e, c->ix_dim, "%s.%d.self_attn.indexer.k_layernorm.weight", p, l);
        L->K = (float*) dmem(e, NULL, sizeof(float) * (size_t) e->max_pos * c->n_kv * c->head_dim);
        L->V = (float*) dmem(e, NULL, sizeof(float) * (size_t) e->max_pos * c->n_kv * c->head_dim);
        L->raw = (float*) dmem(e, NULL, sizeof(float) * (size_t) e->max_pos * c->ix_dim);
        L->blk = (float*) dmem(e, NULL, sizeof(float) * (size_t) nb * c->ix_dim);
    } else {
        load_w(e, &L->qkv, 0, "%s.%d.linear_attn.in_proj_qkv.weight", p, l);
        load_w(e, &L->z, 0, "%s.%d.linear_attn.in_proj_z.weight", p, l);
        load_w(e, &L->b, 0, "%s.%d.linear_attn.in_proj_b.weight", p, l);
        load_w(e, &L->a, 0, "%s.%d.linear_attn.in_proj_a.weight", p, l);
        load_w(e, &L->out, 0, "%s.%d.linear_attn.out_proj.weight", p, l);
        L->gw.conv_w = load_v(e, (int64_t) cd * c->lin_conv, "%s.%d.linear_attn.conv1d.weight", p, l);
        L->gw.A_log = load_v(e, c->lin_hv, "%s.%d.linear_attn.A_log", p, l);
        L->gw.dt_bias = load_v(e, c->lin_hv, "%s.%d.linear_attn.dt_bias", p, l);
        L->gw.norm_w = load_v(e, c->lin_dv, "%s.%d.linear_attn.norm.weight", p, l);
        L->conv = (float*) dmem(e, NULL, sizeof(float) * qwen4_gdn_conv_len(c));
        L->S = (float*) dmem(e, NULL, sizeof(float) * qwen4_gdn_state_len(c));
    }
    load_w(e, &L->gate, 0, "%s.%d.mlp.gate.weight", p, l);
    load_w(e, &L->sg, 0, "%s.%d.mlp.shared_expert_gate.weight", p, l);
    load_w(e, &L->sh_g, 0, "%s.%d.mlp.shared_expert.gate_proj.weight", p, l);
    load_w(e, &L->sh_u, 0, "%s.%d.mlp.shared_expert.up_proj.weight", p, l);
    load_w(e, &L->sh_d, 0, "%s.%d.mlp.shared_expert.down_proj.weight", p, l);
    W* ex[3] = {&L->eg, &L->eu, &L->ed};
    const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int i = 0; i < 3; ++i) {
        char name[160];
        snprintf(name, sizeof name, "%s.%d.mlp.switch_mlp.%s.weight", p, l, proj[i]);
        const NsTensor* t = e->et[l][i] = ns_find(&e->nm, name);
        if (!e->cache) load_w(e, ex[i], e->host_experts, "%s", name);
        else if (!t || t->enc == NS_SEED6P8 || t->slices != c->n_exp) fail(e, "%s: missing, P = 8, or not %d experts", name, c->n_exp);
        else { memset(ex[i], 0, sizeof *ex[i]); ex[i]->fmt = t->enc; ex[i]->slices = t->slices; ex[i]->rows = t->rows; ex[i]->cols = t->cols; }
    }
    if (c->ple[l] >= 0) {
        load_w(e, &L->key, 0, "%s.%d.ple.key_proj.weight", p, l);
        load_w(e, &L->val, 0, "%s.%d.ple.value_proj.weight", p, l);
        L->nk = load_v(e, HD, "%s.%d.ple.norm_key.weight", p, l);
        L->nq = load_v(e, HD, "%s.%d.ple.norm_query.weight", p, l);
        L->nc = load_v(e, HD, "%s.%d.ple.norm_conv.weight", p, l);
        L->pw = load_v(e, (int64_t) HD * c->ple_conv, "%s.%d.ple.conv1d.weight", p, l);
        L->hist = (float*) dmem(e, NULL, sizeof(float) * qwen4_ple_hist_len(c));
    }
}

// Unit u (layer u / n_exp, expert u % n_exp) assembled from the folder's mapping into dst, in the pool's layout
static const uint8_t* unit_bytes(Q4Eng* e, int u, uint8_t* dst) {
    const int E = e->c.n_exp, l = u / E, x = u % E;
    for (int i = 0; i < 3; ++i)
        for (int s = 0; s < 4; ++s)
            if (e->ulen[i][s]) memcpy(dst + e->pool.off[i][s], e->et[l][i]->s[s].p + e->ulen[i][s] * (uint64_t) x, e->ulen[i][s]);
    return dst;
}

// The expert cache (engine/kernels_cuda.h CachePool): unit l * n_exp + x holds expert x's gate, up and down slices of
// layer l, in a pinned host arena; `slots` of them in VRAM, filled round-robin over the layers, then LRU.
static void build_cache(Q4Eng* e, double gb) {
    const Qwen4Cfg* c = &e->c;
    CachePool* p = &e->pool;
    const int E = c->n_exp, units = c->n_layer * E;
    if (E > CACHE_MAX_PER_LAYER) { fail(e, "expert cache: %d experts per layer, at most %d", E, CACHE_MAX_PER_LAYER); return; }
    memset(p, 0, sizeof *p);
    p->per_layer = E;
    p->ntens = 3;
    uint64_t ub = 0, len[3][4] = {{0}};
    for (int i = 0; i < 3; ++i)
        for (int s = 0; s < 4; ++s) {
            const NsTensor* t = e->et[0][i];
            const int seed = t->enc == NS_SEED4 || t->enc == NS_SEED4P4;
            len[i][s] = seed && s == 2 ? 0 : t->s[s].len / (uint64_t) E;
            p->off[i][s] = len[i][s] ? (uint32_t) ub : 0xFFFFFFFFu;
            ub += (len[i][s] + 255) & ~255ull;
        }
    p->unit_bytes = ub;
    const double want = gb * 1e9 / (double) ub;
    p->slots = want > units ? units : want < E ? E : (int) want;
    memcpy(e->ulen, len, sizeof len);
    const int nstage = E < MAXT * c->top_k ? E : MAXT * c->top_k;   // the most units one admit brings
    uint8_t* host = NULL;
    void* dh = NULL;
    if (e->stream) {   // no arena: units are assembled from the folder's mapping
        if (cudaHostAlloc((void**) &e->stage, ub * (uint64_t) nstage, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**) &e->hjobs, 8 * (size_t) E + 16, cudaHostAllocDefault) != cudaSuccess) { fail(e, "cannot pin the staging buffer"); return; }
        track(&e->host, &e->nhost, &e->chost, e->stage);
        track(&e->host, &e->nhost, &e->chost, e->hjobs);
    } else if (cudaHostAlloc((void**) &host, ub * (uint64_t) units, cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer(&dh, host, 0) != cudaSuccess) {
        fail(e, "cannot pin %.1f GB of host memory for the experts (--stream-experts reads them from the files)", ub * (double) units / 1e9);
        return;
    } else track(&e->host, &e->nhost, &e->chost, host);
    WSlice* tab = (WSlice*) calloc((size_t) units * 3, sizeof(WSlice));
    for (int l = 0; l < c->n_layer; ++l)
        for (int i = 0; i < 3; ++i) {
            const NsTensor* t = e->et[l][i];
            if (!t || t->enc != e->et[0][i]->enc || t->s[0].len != e->et[0][i]->s[0].len) { fail(e, "layer %d: experts differ from layer 0's", l); free(tab); return; }
            if ((t->enc == NS_Q4 || t->enc == NS_Q8) && t->group != 32 && t->group != 64) { fail(e, "%s: affine group %d not supported", t->name, t->group); free(tab); return; }
            for (int x = 0; x < E; ++x) {
                WSlice* w = &tab[((size_t) l * 3 + i) * E + x];
                for (int s = 0; s < 4 && host; ++s)
                    if (len[i][s]) memcpy(host + ((uint64_t) l * E + x) * ub + p->off[i][s], t->s[s].p + len[i][s] * (uint64_t) x, len[i][s]);
                if (t->enc == NS_SEED4 || t->enc == NS_SEED4P4) memcpy(&w->eb, t->s[2].p + 4 * (size_t) x, 4);
                if (t->enc == NS_Q4 || t->enc == NS_Q8) w->gs = t->group == 32 ? 5 : 6;
            }
        }
    p->host = (const uint8_t*) dh;
    p->vram = (uint8_t*) dmem(e, NULL, ub * (uint64_t) p->slots);
    int32_t* us = (int32_t*) malloc(4 * (size_t) units);
    int32_t* su = (int32_t*) malloc(4 * (size_t) p->slots);
    for (int u = 0; u < units; ++u) us[u] = -1;
    for (int s = 0, x = 0; x < E && s < p->slots; ++x)   // round-robin: every layer starts with the same share
        for (int l = 0; l < c->n_layer && s < p->slots; ++l, ++s) {
            const int u = l * E + x;
            us[u] = s;
            su[s] = u;
            if (p->vram) cudaMemcpy(p->vram + (uint64_t) s * ub, host ? host + (uint64_t) u * ub : unit_bytes(e, u, e->stage), ub, cudaMemcpyHostToDevice);
            for (int i = 0; i < 3; ++i)
                for (int st = 0; st < 4; ++st)
                    if (len[i][st] && p->vram) tab[((size_t) l * 3 + i) * E + x].p[st] = p->vram + (uint64_t) s * ub + p->off[i][st];
        }
    const uint32_t tick0[4] = {CACHE_TICK_BASE, 0, 0, 0};
    p->unit_slot = (int32_t*) dmem(e, us, 4 * (size_t) units);
    p->slot_unit = (int32_t*) dmem(e, su, 4 * (size_t) p->slots);
    p->slot_last = (uint32_t*) dmem(e, NULL, 4 * (size_t) p->slots);
    p->tick = (uint32_t*) dmem(e, tick0, 16);
    p->stats = (uint32_t*) dmem(e, NULL, 16);
    p->jobs = (int32_t*) dmem(e, NULL, 8 * (size_t) E);
    p->njobs = (int32_t*) dmem(e, NULL, 16);
    p->tab = (WSlice*) dmem(e, tab, sizeof(WSlice) * (size_t) units * 3);
    for (int l = 0; l < c->n_layer && p->tab; ++l) {
        e->L[l].eg.tab = p->tab + ((size_t) l * 3 + 0) * E;
        e->L[l].eu.tab = p->tab + ((size_t) l * 3 + 1) * E;
        e->L[l].ed.tab = p->tab + ((size_t) l * 3 + 2) * E;
    }
    free(us); free(su); free(tab);
}

void q4e_cache_stats(Q4Eng* e, uint64_t* hits, uint64_t* misses, int* slots) {
    uint32_t st[2] = {0, 0};
    if (e->cache) cudaMemcpy(st, e->pool.stats, 8, cudaMemcpyDeviceToHost);
    *hits = st[0];
    *misses = st[1];
    *slots = e->cache ? e->pool.slots : 0;
}

Q4Eng* q4e_open(const char* model_dir, const Q4EOpts* o, char* err, int errlen) {
    Q4Eng* e = (Q4Eng*) calloc(1, sizeof *e);
    if (qwen4_cfg_load(&e->c, model_dir, err, errlen) || ns_open(&e->nm, model_dir, err, errlen)) { free(e); return NULL; }
    const Qwen4Cfg* c = &e->c;
    if (c->n_exp > BUCKET_MAX_EXPERTS) { snprintf(err, (size_t) errlen, "%d experts, at most %d", c->n_exp, BUCKET_MAX_EXPERTS); ns_close(&e->nm); free(e); return NULL; }
    e->max_pos = o && o->max_pos > 0 ? o->max_pos : 4096;
    e->host_experts = !(o && o->vram_experts);
    e->cache = e->host_experts && o && o->cache_gb > 0;
    e->stream = e->cache && o->stream_experts;
    if (o && o->stream_experts && !e->cache) { snprintf(err, (size_t) errlen, "--stream-experts needs --expert-cache-gb"); ns_close(&e->nm); free(e); return NULL; }
    if (o && o->ple_gguf) {
        if (ple_open(&e->pt, o->ple_gguf, "per_layer_token_embd.weight", err, errlen)) { ns_close(&e->nm); free(e); return NULL; }
        e->table = &e->pt;
    }
    ns_rows_init(&e->rows);
    cudaStreamCreate(&e->s);
    e->g24 = (uint32_t*) dmem(e, e->rows.g24, sizeof e->rows.g24);
    e->g32 = (uint32_t*) dmem(e, e->rows.g32, sizeof e->rows.g32);
    load_w(e, &e->embed, 0, "language_model.model.embed_tokens.weight");
    load_w(e, &e->head, 0, "language_model.lm_head.weight");
    load_w(e, &e->mix_down, 0, "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight");
    load_w(e, &e->mix_up, 0, "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight");
    e->mix_norm = load_v(e, (int64_t) c->hc * c->d, "language_model.model.hyper_connection_mixer.hc_norm.weight");
    for (int l = 0; l < c->n_layer && !e->err[0]; ++l) {
        if (c->ple[l] >= 0) qwen4_ngram_init(c, c->ple[l], &e->ng[c->ple[l]]);
        if (c->ple[l] >= 0 && e->table && (c->ple[l] > 0 || e->table->rows != e->ng[0].rows))
            fail(e, "the PLE GGUF has %lld rows, the model's table %lld (and one PLE layer)", (long long) e->table->rows, (long long) e->ng[0].rows);
        load_layer(e, l);
    }
    if (e->cache && !e->err[0]) build_cache(e, o->cache_gb);
    const size_t T = MAXT, HD = (size_t) c->hc * c->d, cd = (size_t) 2 * c->lin_hk * c->lin_dk + (size_t) c->lin_hv * c->lin_dv;
    const size_t vd = (size_t) c->lin_hv * c->lin_dv, iw = (size_t) (c->ix_heads + 1) * c->ix_dim, kf = (size_t) c->top_k * c->ff_exp;
    const size_t nb = (size_t) e->max_pos / c->ix_ratio + 1;
#define F(n) (float*) dmem(e, NULL, sizeof(float) * (n))
#define I(n) (int32_t*) dmem(e, NULL, sizeof(int32_t) * (n))
    e->x = F(T * c->d); e->Hall = e->H = F((size_t) XR * HD); e->normed = F(T * HD); e->lo = F(T * c->hc_rank); e->up = F(T * HD); e->inj = F(T * c->hc);
    e->br = F(T * c->d); e->qp = F(T * c->n_head * 2 * c->head_dim); e->kk = F(T * c->n_kv * c->head_dim);
    e->vv = F(T * c->n_kv * c->head_dim); e->o = F(T * c->n_head * c->head_dim); e->ixo = F(T * iw); e->sc = F(T * nb);
    e->emb = F(T * c->ple_dim); e->keys = F(T * HD); e->qs = F(T * HD); e->vals = F(T * c->d); e->gated = F(T * HD);
    const size_t ka = kf > (size_t) c->ff_shared ? kf : (size_t) c->ff_shared;
    e->lg = F(T * c->n_exp); e->wt = F(T * c->top_k); e->A = F(T * ka); e->Gb = F(T * ka); e->Ub = F(T * ka);
    e->Dx = F(T * c->top_k * c->d); e->sh = F(T * c->d); e->sgv = F(T); e->logits = F((size_t) Q4E_LOGIT_ROWS * c->vocab); e->y = F(T * vd);
    e->cv = F(T * cd); e->qkvb = F(T * cd); e->zb = F(T * vd); e->bb = F(T * c->lin_hv); e->ab = F(T * c->lin_hv);
    e->din = I(T + 1); e->ind = I(T * c->top_k); e->sel = I(T * (c->ix_budget / c->ix_ratio)); e->done = I(T); e->nsel = I(T); e->zero = I(T);
    e->perm = I(T * c->top_k); e->ntiles = I(2 * (size_t) c->n_exp + 3);
    e->tiles = (MmTile*) dmem(e, NULL, sizeof(MmTile) * (T * c->top_k / MMT_BN + (size_t) c->n_exp));
#undef F
#undef I
    for (int l = 0; l < c->n_layer; ++l) if (c->ple[l] + 1 > e->nple) e->nple = c->ple[l] + 1;
    e->nb = (int) nb;
    e->hin = (int32_t*) pin(e, sizeof(int32_t) * (XR / MAXT) * (T + 1));   // a slot of T + 1 for each chunk of a pass
    e->emb_h = (float*) pin(e, sizeof(float) * XR * (size_t) c->ple_dim * (size_t) (e->nple ? e->nple : 1));
    e->hlog = (float*) pin(e, sizeof(float) * (size_t) Q4E_LOGIT_ROWS * c->vocab);
    e->no_graph = getenv("NSLM_NO_GRAPH") != NULL;
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    e->ple_threads = getenv("Q4E_PLE_THREADS") ? atoi(getenv("Q4E_PLE_THREADS")) : ncpu > 16 ? 16 : ncpu > 1 ? (int) ncpu : 1;
    if (e->ple_threads < 1) e->ple_threads = 1;
    if (e->ple_threads > 16) e->ple_threads = 16;
    e->prof = getenv("Q4E_PROFILE") && atoi(getenv("Q4E_PROFILE")) > 0;
    if (getenv("Q4E_ROUTES") && !(e->routes = fopen(getenv("Q4E_ROUTES"), "wb"))) fail(e, "cannot write %s", getenv("Q4E_ROUTES"));
    if (e->routes) {
        e->droute = (int32_t*) dmem(e, NULL, sizeof(int32_t) * (size_t) c->n_layer * T * c->top_k);
        e->hroute = (int32_t*) pin(e, sizeof(int32_t) * (size_t) c->n_layer * T * c->top_k);
    }
    for (int i = 0; e->prof && i < Q4E_NEV; ++i) cudaEventCreate(&e->ev[i]);
    e->ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) e->max_pos);
    if (!e->err[0] && cudaDeviceSynchronize() != cudaSuccess) fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    if (e->err[0]) { snprintf(err, (size_t) errlen, "%s", e->err); q4e_close(e); return NULL; }
    return e;
}

void q4e_close(Q4Eng* e) {
    if (!e) return;
    cudaDeviceSynchronize();
    if (e->prof) {
        double total = 0;
        for (int i = 0; i < ST_N; ++i) total += e->stage_ms[i];
        fprintf(stderr, "Q4E_PROFILE: %.0f ms on the GPU stream\n", total);
        for (int i = 0; i < ST_N; ++i)
            if (e->stage_ms[i] > 0) fprintf(stderr, "  %-18s %9.1f ms  %5.1f%%\n", kStage[i], e->stage_ms[i], 100 * e->stage_ms[i] / (total > 0 ? total : 1));
        fprintf(stderr, "Q4E_PROFILE: host PLE prefetch %.0f ms, PLE rows %.0f ms (%d threads)\n", e->host_ms[0], e->host_ms[1], e->ple_threads);
        for (int i = 0; i < Q4E_NEV; ++i) cudaEventDestroy(e->ev[i]);
    }
    for (int i = 0; i < e->ndev; ++i) cudaFree(e->dev[i]);
    for (int i = 0; i < e->nhost; ++i) cudaFreeHost(e->host[i]);
    for (int w = 0; w < 2; ++w) for (int t = 0; t <= MV_MAXT; ++t) if (e->graph[w][t]) cudaGraphExecDestroy(e->graph[w][t]);
    free(e->dev); free(e->host); free(e->ids);
    if (e->routes) fclose(e->routes);
    if (e->s) cudaStreamDestroy(e->s);
    if (e->table) ple_close(&e->pt);
    ns_close(&e->nm);
    free(e);
}
const Qwen4Cfg* q4e_cfg(const Q4Eng* e) { return &e->c; }
int q4e_pos(const Q4Eng* e) { return e->pos; }
const char* q4e_error(const Q4Eng* e) { return e->err; }

void q4e_reset(Q4Eng* e) {
    const Qwen4Cfg* c = &e->c;
    for (int l = 0; l < c->n_layer; ++l) {
        Layer* L = &e->L[l];
        if (L->conv) cudaMemsetAsync(L->conv, 0, sizeof(float) * qwen4_gdn_conv_len(c), e->s);
        if (L->S) cudaMemsetAsync(L->S, 0, sizeof(float) * qwen4_gdn_state_len(c), e->s);
        if (L->hist) cudaMemsetAsync(L->hist, 0, sizeof(float) * qwen4_ple_hist_len(c), e->s);
    }
    cudaStreamSynchronize(e->s);
    e->pos = 0;
}

// ---- the stage profiler (Q4E_PROFILE=1): an event after each stage, the time to the previous event added to it -------

static void mark(Q4Eng* e, int stage) {
    if (!e->prof || e->nev >= Q4E_NEV) return;
    cudaEventRecord(e->ev[e->nev], e->s);
    e->evstage[e->nev++] = stage;
}
static void prof_collect(Q4Eng* e) {   // after the stream's sync
    for (int i = 1; i < e->nev; ++i) {
        float ms = 0;
        cudaEventElapsedTime(&ms, e->ev[i - 1], e->ev[i]);
        e->stage_ms[e->evstage[i]] += ms;
    }
    e->nev = 0;
}

// ---- the forward of up to MAXT rows ----------------------------------------------------------------------------------

// The streams of H mixed into e->x; with inj, the weights that add the branch back (nslm/qwen4_ops.h)
static void hc_mix(Q4Eng* e, const float* norm, const W* down, const W* upw, const W* injw, int T) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d;
    kq4_norm1p(e->s, e->H, HD, c->d, norm, c->eps, e->normed, T);
    mv(e, down, e->normed, e->lo, T);
    kq4_hc_act(e->s, c->hc, e->lo, T * c->hc_rank);
    mv(e, upw, e->lo, e->up, T);
    kq4_hc_mix(e->s, c->hc, c->d, e->normed, e->up, e->x, T);
    if (injw) { mv(e, injw, e->normed, e->inj, T); kq4_hc_gate(e->s, c->hc, e->inj, T * c->hc); }
}

static double host_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e3 * (double) ts.tv_sec + 1e-6 * (double) ts.tv_nsec;
}

typedef struct { Q4Eng* e; int t0, t1, prefetch, bad; } PleJob;   // bad: 1 + the failing layer
static void* ple_worker(void* arg) {
    PleJob* j = (PleJob*) arg;
    Q4Eng* e = j->e;
    const Qwen4Cfg* c = &e->c;
    for (int l = 0; l < c->n_layer && !j->bad; ++l)
        for (int t = j->t0; c->ple[l] >= 0 && t < j->t1; ++t) {
            if (j->prefetch) qwen4_ple_prefetch(&e->nm, e->table, c, &e->ng[c->ple[l]], l, e->ids, e->pos + t);
            else if (qwen4_ple_embed(&e->nm, &e->rows, e->table, c, &e->ng[c->ple[l]], l, e->ids, e->pos + t,
                                     e->emb_h + ((size_t) c->ple[l] * XR + t) * c->ple_dim)) { j->bad = l + 1; break; }
        }
    return NULL;
}

// The PLE rows [pos, pos + T) of each PLE layer: their pages asked of the OS (prefetch), or into emb_h. Host work,
// before the rows' device work; calls of 64 rows or more split the rows over ple_threads threads.
static int ple_run(Q4Eng* e, int T, int prefetch) {
    PleJob j[16];
    pthread_t th[16];
    const int nw = T < 64 ? 1 : e->ple_threads;
    const double t0 = e->prof ? host_ms() : 0;
    for (int w = 0; w < nw; ++w) {
        j[w] = (PleJob) {e, (int) ((int64_t) T * w / nw), (int) ((int64_t) T * (w + 1) / nw), prefetch, 0};
        if (nw > 1) pthread_create(&th[w], NULL, ple_worker, &j[w]);
    }
    if (nw == 1) ple_worker(&j[0]);
    for (int w = 0; nw > 1 && w < nw; ++w) pthread_join(th[w], NULL);
    if (e->prof) e->host_ms[prefetch ? 0 : 1] += host_ms() - t0;
    for (int w = 0; w < nw; ++w)
        if (j[w].bad) return fail(e, "layer %d: a PLE table shard is missing or does not hold a row", j[w].bad - 1);
    return 0;
}
static int ple_rows(Q4Eng* e, int T) { return ple_run(e, T, 0); }

static void ple(Q4Eng* e, int l, int T, int row0) {   // row0: the chunk's first row in emb_h
    const Qwen4Cfg* c = &e->c;
    Layer* L = &e->L[l];
    const int HD = c->hc * c->d;
    cudaMemcpyAsync(e->emb, e->emb_h + ((size_t) c->ple[l] * XR + row0) * c->ple_dim, sizeof(float) * (size_t) T * c->ple_dim,
                    cudaMemcpyHostToDevice, e->s);
    mv(e, &L->key, e->emb, e->keys, T);
    kq4_norm1p(e->s, e->keys, HD, c->d, L->nk, c->eps, e->keys, T);
    mv(e, &L->val, e->emb, e->vals, T);
    kq4_norm1p(e->s, e->H, HD, c->d, L->nq, c->eps, e->qs, T);
    kq4_ple_gate(e->s, c, e->keys, e->qs, e->vals, e->gated, T);
    kq4_norm1p(e->s, e->gated, HD, c->d, L->nc, c->eps, e->keys, T);   // keys now holds the conv input
    kq4_ple_conv(e->s, c, L->pw, e->keys, e->gated, L->hist, e->H, T);
}

static int attention(Q4Eng* e, Layer* L, int T) {
    const Qwen4Cfg* c = &e->c;
    const int hd = c->head_dim, kvw = c->n_kv * hd, D = c->ix_dim, R = c->ix_ratio, iw = (c->ix_heads + 1) * D;
    mv(e, &L->q, e->x, e->qp, T);
    mv(e, &L->k, e->x, e->kk, T);
    mv(e, &L->v, e->x, e->vv, T);
    kq4_head_norm_rope(e->s, c, e->qp, c->n_head * 2 * hd, c->n_head, 2 * hd, hd, L->qn, 0, e->din, T);
    kq4_head_norm_rope(e->s, c, e->kk, kvw, c->n_kv, hd, hd, L->kn, 0, e->din, T);
    kq4_store(e->s, L->K, kvw, e->kk, kvw, 0, e->din, T);
    kq4_store(e->s, L->V, kvw, e->vv, kvw, 0, e->din, T);
    mv(e, &L->ix, e->x, e->ixo, T);
    mark(e, ST_PROJ);
    kq4_store(e->s, L->raw, D, e->ixo + (size_t) c->ix_heads * D, iw, 0, e->din, T);
    kq4_head_norm_rope(e->s, c, e->ixo, iw, c->ix_heads, D, D, L->iqn, 0, e->din, T);
    kq4_qsa_blocks(e->s, c, L->raw, L->ikn, 0, e->din, T, L->blk);
    kq4_qsa_rows(e->s, c, 0, e->din, e->done, e->nsel, T);
    // the score grid and stride: the blocks done by the last row, or in a graph (any position) every block slot
    kq4_qsa_select(e->s, c, e->ixo, iw, L->blk, e->done, e->cap ? e->nb : (e->pos + T) / R, e->sc, e->sel, T);
    if (kq4_attend(e->s, c, e->qp, L->K, L->V, 0, e->din, e->sel, e->nsel, e->o, T)) return fail(e, "head_dim %d not supported", hd);
    mark(e, ST_ATTN);
    mv(e, &L->o, e->o, e->br, T);
    mark(e, ST_PROJ);
    return 0;
}

static int gdn(Q4Eng* e, Layer* L, int T) {
    mv(e, &L->qkv, e->x, e->qkvb, T);
    mv(e, &L->z, e->x, e->zb, T);
    mv(e, &L->b, e->x, e->bb, T);
    mv(e, &L->a, e->x, e->ab, T);
    mark(e, ST_PROJ);
    if (kq4_gdn(e->s, &e->c, L->gw, e->qkvb, e->zb, e->ab, e->bb, L->conv, L->S, e->cv, e->y, T)) return fail(e, "Gated DeltaNet head widths not supported");
    mark(e, ST_GDN);
    mv(e, &L->out, e->y, e->br, T);
    mark(e, ST_PROJ);
    return 0;
}

static void moe(Q4Eng* e, int l, int T) {
    const Qwen4Cfg* c = &e->c;
    Layer* L = &e->L[l];
    const int k = c->top_k, P = T * k;
    mv(e, &L->gate, e->x, e->lg, T);
    kq4_route(e->s, c, e->lg, e->ind, e->wt, T);
    if (e->routes) cudaMemcpyAsync(e->droute + (size_t) l * P, e->ind, sizeof(int32_t) * (size_t) P, cudaMemcpyDeviceToDevice, e->s);
    mark(e, ST_EXPERT);   // the router's own matvec counts with the experts; the admit and copies below with the cache
    if (e->cache) {   // bring the routed experts missing from VRAM (the pool's slice tables then point at them)
        kc_cache_admit(e->s, &e->pool, l, e->ind, P, 0);
        if (!e->stream) kc_cache_copy(e->s, &e->pool, 64);
        else {   // the admit's (unit, slot) jobs back to the host, the units from the files, then to their slots
            cudaMemcpyAsync(e->hjobs, e->pool.njobs, 4, cudaMemcpyDeviceToHost, e->s);
            cudaMemcpyAsync(e->hjobs + 4, e->pool.jobs, 8 * (size_t) c->n_exp, cudaMemcpyDeviceToHost, e->s);
            cudaStreamSynchronize(e->s);   // also: the last layer's copies out of the staging buffer are done
            const uint64_t ub = e->pool.unit_bytes;
            for (int j = 0; j < e->hjobs[0]; ++j) {
                uint8_t* st = e->stage + (uint64_t) j * ub;
                unit_bytes(e, e->hjobs[4 + 2 * j], st);
                cudaMemcpyAsync(e->pool.vram + (uint64_t) e->hjobs[5 + 2 * j] * ub, st, ub, cudaMemcpyHostToDevice, e->s);
            }
        }
    }
    mark(e, ST_ROUTE);
    if (T > MV_MAXT) {   // prompt rows: pairs by expert, so each tile reads its expert's weights once (tensor cores)
        const int mt = P / MMT_BN + c->n_exp;
        kc_bucket(e->s, e->ind, P, c->n_exp, e->perm, e->tiles, e->ntiles);
        kc_mm_grouped_dev(e->s, L->eg.fmt, L->eg.tab, c->d, c->ff_exp, e->x, c->d, e->Gb, c->ff_exp, k, e->perm, e->tiles, e->ntiles, mt, G(e, &L->eg));
        kc_mm_grouped_dev(e->s, L->eu.fmt, L->eu.tab, c->d, c->ff_exp, e->x, c->d, e->Ub, c->ff_exp, k, e->perm, e->tiles, e->ntiles, mt, G(e, &L->eu));
        kc_swiglu(e->s, e->Gb, e->Ub, e->A, P * c->ff_exp);
        kc_mm_grouped_dev(e->s, L->ed.fmt, L->ed.tab, c->ff_exp, c->d, e->A, c->ff_exp, e->Dx, c->d, 1, e->perm, e->tiles, e->ntiles, mt, G(e, &L->ed));
        mv(e, &L->sh_g, e->x, e->Gb, T);
        mv(e, &L->sh_u, e->x, e->Ub, T);
        kc_swiglu(e->s, e->Gb, e->Ub, e->A, T * c->ff_shared);
    } else {
        kc_mv_gu(e->s, L->eg.fmt, L->eg.tab, L->eu.tab, c->d, c->ff_exp, e->x, c->d, e->A, c->ff_exp, e->ind, P, k, G(e, &L->eg));
        kc_mv_sel(e->s, L->ed.fmt, L->ed.tab, c->ff_exp, c->d, e->A, c->ff_exp, e->Dx, c->d, e->ind, P, 1, G(e, &L->ed));
        kc_mv_gu(e->s, L->sh_g.fmt, L->sh_g.tab, L->sh_u.tab, c->d, c->ff_shared, e->x, c->d, e->A, c->ff_shared, e->zero, T, 1, G(e, &L->sh_g));
    }
    mv(e, &L->sh_d, e->A, e->sh, T);
    mv(e, &L->sg, e->x, e->sgv, T);
    kq4_moe_combine(e->s, c->d, k, e->Dx, e->wt, e->sh, e->sgv, e->br, T);
}

// The device work of a chunk of T rows: its input slot to din, the embedding into H, a layer, the head (pinned logits)
static void rows_in(Q4Eng* e, int slot, int T) {
    cudaMemcpyAsync(e->din, e->hin + (size_t) slot * (MAXT + 1), sizeof(int32_t) * (size_t) (T + 1), cudaMemcpyHostToDevice, e->s);
}
static void embed(Q4Eng* e, int T) {
    mark(e, ST_EMBED);
    kc_embed(e->s, e->embed.fmt, e->embed.w, e->c.d, e->din + 1, e->x, T);
    kq4_tile(e->s, e->c.hc, e->c.d, e->x, e->H, T);
    mark(e, ST_EMBED);
}
static int layer(Q4Eng* e, int l, int T, int row0) {
    const Qwen4Cfg* c = &e->c;
    Layer* L = &e->L[l];
    if (c->ple[l] >= 0) ple(e, l, T, row0);
    mark(e, ST_PLE);
    hc_mix(e, L->hc_norm[0], &L->hc_down[0], &L->hc_up[0], &L->hc_inj[0], T);
    mark(e, ST_HC);
    if (c->attn[l] ? attention(e, L, T) : gdn(e, L, T)) return -1;
    mark(e, c->attn[l] ? ST_ATTN : ST_GDN);
    kq4_hc_add(e->s, c->hc, c->d, e->H, e->br, e->inj, T);
    hc_mix(e, L->hc_norm[1], &L->hc_down[1], &L->hc_up[1], &L->hc_inj[1], T);
    mark(e, ST_HC);
    moe(e, l, T);
    kq4_hc_add(e->s, c->hc, c->d, e->H, e->br, e->inj, T);
    mark(e, ST_EXPERT);
    return e->err[0] ? -1 : 0;
}
static int enqueue(Q4Eng* e, int T, int want) {   // the rows of slot 0 through every layer; want: with the head
    rows_in(e, 0, T);
    embed(e, T);
    for (int l = 0; l < e->c.n_layer; ++l) if (layer(e, l, T, 0)) return -1;
    if (want) {   // prompt rows without logits skip the mixer and the head
        hc_mix(e, e->mix_norm, &e->mix_down, &e->mix_up, NULL, T);
        mv(e, &e->head, e->x, e->logits, T);
        cudaMemcpyAsync(e->hlog, e->logits, sizeof(float) * (size_t) T * e->c.vocab, cudaMemcpyDeviceToHost, e->s);
        mark(e, ST_HEAD);
    }
    return e->err[0] ? -1 : 0;
}

// Prompt rows without logits (n <= XR), layer by layer over chunks of MAXT rows, so the cache brings each layer's
// experts once a pass, not once a chunk.  Each chunk sees the inputs of the row-major forward: the same logits.
static int forward_pass(Q4Eng* e, const int32_t* ids, int n) {
    const int pos0 = e->pos, nc = (n + MAXT - 1) / MAXT, HD = e->c.hc * e->c.d;
    for (int k = 0; k < nc; ++k) {
        int32_t* h = e->hin + (size_t) k * (MAXT + 1);
        h[0] = pos0 + k * MAXT;
        memcpy(h + 1, ids + (size_t) k * MAXT, sizeof(int32_t) * (size_t) (n - k * MAXT < MAXT ? n - k * MAXT : MAXT));
    }
    int rc = ple_rows(e, n);
    for (int l = -1; !rc && l < e->c.n_layer; ++l)   // l = -1: the embedding
        for (int k = 0; !rc && k < nc; ++k) {
            const int T = n - k * MAXT < MAXT ? n - k * MAXT : MAXT;
            e->H = e->Hall + (size_t) k * MAXT * HD;
            e->pos = pos0 + k * MAXT;
            rows_in(e, k, T);
            if (l < 0) embed(e, T);
            else rc = layer(e, l, T, k * MAXT);
        }
    e->H = e->Hall;
    e->pos = pos0;
    if (rc) return -1;
    if (cudaStreamSynchronize(e->s) != cudaSuccess || cudaGetLastError() != cudaSuccess)
        return fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    prof_collect(e);
    e->pos += n;
    return 0;
}

// Decode rows (up to MV_MAXT) run as a CUDA graph, captured once for each row count, with or without logits
static int forward_rows(Q4Eng* e, const int32_t* ids, int T, float* logits) {
    e->hin[0] = e->pos;
    memcpy(e->hin + 1, ids, sizeof(int32_t) * (size_t) T);
    if (ple_rows(e, T)) return -1;
    const int want = logits != NULL, graph = T <= MV_MAXT && !e->prof && !e->stream && !e->no_graph;
    cudaGraphExec_t* ge = &e->graph[want][T];
    if (graph && !*ge) {
        cudaGraph_t gr = NULL;
        if (cudaStreamBeginCapture(e->s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return fail(e, "CUDA graph capture failed");
        e->cap = 1;
        const int rc = enqueue(e, T, want);
        e->cap = 0;
        if (cudaStreamEndCapture(e->s, &gr) != cudaSuccess || rc) { if (gr) cudaGraphDestroy(gr); return fail(e, "CUDA graph capture failed"); }
        const cudaError_t ri = cudaGraphInstantiate(ge, gr, 0);
        cudaGraphDestroy(gr);
        if (ri != cudaSuccess) { *ge = NULL; return fail(e, "CUDA graph: %s", cudaGetErrorString(ri)); }
    }
    if (graph ? cudaGraphLaunch(*ge, e->s) != cudaSuccess : enqueue(e, T, want)) return fail(e, "CUDA: forward failed");
    if (cudaStreamSynchronize(e->s) != cudaSuccess || cudaGetLastError() != cudaSuccess)
        return fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    if (logits) memcpy(logits, e->hlog, sizeof(float) * (size_t) T * e->c.vocab);
    const size_t nr = (size_t) e->c.n_layer * T * e->c.top_k;
    if (e->routes && (cudaMemcpy(e->hroute, e->droute, sizeof(int32_t) * nr, cudaMemcpyDeviceToHost) != cudaSuccess ||
                      fwrite(&T, sizeof T, 1, e->routes) != 1 || fwrite(e->hroute, sizeof(int32_t), nr, e->routes) != nr))
        return fail(e, "Q4E_ROUTES: write failed");
    prof_collect(e);
    e->pos += T;
    return 0;
}

int q4e_forward(Q4Eng* e, const int32_t* ids, int n, float* logits) {
    e->err[0] = 0;
    if (e->pos + n > e->max_pos) return fail(e, "the context is full (%d positions)", e->max_pos);
    for (int i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= e->embed.rows) return fail(e, "token %d out of range", ids[i]);
    memcpy(e->ids + e->pos, ids, sizeof(int32_t) * (size_t) n);
    ple_run(e, n, 1);   // all PLE rows of the call at once: the OS reads them in parallel, not row by row
    if (!logits && n > MAXT && !e->routes) {   // a long prompt: layer-major passes (Q4E_ROUTES traces chunk by chunk)
        for (int i = 0; i < n; i += XR) if (forward_pass(e, ids + i, n - i < XR ? n - i : XR)) return -1;
        return 0;
    }
    const int step = logits ? Q4E_LOGIT_ROWS : MAXT;
    for (int i = 0; i < n; i += step) {
        const int T = n - i < step ? n - i : step;
        if (forward_rows(e, ids + i, T, logits ? logits + (size_t) i * e->c.vocab : NULL)) return -1;
    }
    return 0;
}
