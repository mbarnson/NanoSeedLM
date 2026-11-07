// engine/qwen4_cuda.c - the qwen4 CUDA engine (engine/qwen4_eng.h): the steps of nslm/qwen4_ref.h on the matrix
// kernels of engine/kernels_cuda.h and the qwen4 kernels of engine/kernels_qwen4.h.
#include "qwen4_eng.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "cpu_seed4.h"
#include "kernels_cuda.h"
#include "kernels_qwen4.h"
#include "lfsr.h"
#include "model_st.h"
#include "moe.h"
#include "qwen4_ops.h"
#include "rows.h"
#include "vram.h"

#define Q4E_NEV 32768   // Q4E_PROFILE events of one forward


typedef struct {   // a weight: its format and shape, slice 0, a device table of all its slices, NsTensor's group
    int fmt, slices, rows, cols;
    WSlice w;
    const WSlice* tab;
    int group;
} W;

typedef struct {
    W hc_di[2], hc_up[2];   // each mix: the down rows, then the hc inject rows (one GEMM of the same input); up
    float* hc_norm[2];
    W in, out;   // Gated DeltaNet: in = the rows of in_proj qkv, z, b and a (one GEMM of x)
    Q4GdnDev gw;
    float *conv, *S;
    W qkvi, o;   // QSA attention: qkvi = the rows of q, k, v and the indexer's qk (one GEMM of x)
    float *qn, *kn, *iqn, *ikn, *raw, *blk;
    uint16_t *K, *V;   // BF16 (plan K)
    W gate, sh_g, sh_u, sh_d, eg, eu, ed;   // MoE: gate = the router rows, then the shared expert gate row
    W key, val;                                 // PLE
    float *nk, *nq, *nc, *pw, *hist;
    float *kconv, *kS, *khist;   // q4e_forward_keep: the state after each row but the last (MTP)
} Layer;

// A share [i0, i1) of n items of fn on the pool; rc: fn's result (0, or the first item's error)
typedef struct { Q4Eng* e; int (*fn)(Q4Eng* e, const void* ctx, int i0, int i1); const void* ctx; int i0, i1, rc; } PoolJob;
static void* pool_worker(void* arg);
static size_t ct_floats(const Qwen4Cfg* c);   // plan D4 (below)
static void* ct_thread(void* arg);

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
    cudaStream_t s, s2;          // s2: decode's shared expert, beside the router and the expert copy
    cudaEvent_t ev_x, ev_sh;
    int max_pos, pos, host_experts, cache, stream;
    int mt;                        // rows of one prompt forward (a chunk): Q4EOpts.rows
    int xr;                        // rows of a layer-major pass: the context in chunks of mt, at most Q4E_PASS_ROWS
    CachePool pool;   // the expert cache (cache = 1)

    // Prompt prefetch (pf_ok): layer l + 1's experts admitted as layer l starts, copied on cs. pf: pool with its own
    // job list. It needs a pool of 3 layers, so the victims come from finished layers.
    CachePool pf;
    cudaStream_t cs;
    cudaEvent_t ev_admit, ev_ready[QWEN4_MAX_LAYERS];
    int32_t* all_ids;
    int pf_ok;
    uint64_t ulen[3][4];   // bytes of each expert stream in a unit
    uint8_t* stage;        // stream = 1: pinned staging of the units one admit brings
    int32_t* hjobs;        // stream = 1: the admit's jobs, read back (pinned)
    const NsTensor* et[QWEN4_MAX_LAYERS][3];   // the layers' gate, up and down experts
    int32_t* ids;   // the context (host), for the PLE n-grams
    float *Hall;   // the hidden streams of a pass, xr rows; H: the current chunk's
    float *x, *H, *normed, *lo, *up, *inj, *br, *qp, *o, *sc, *part, *emb, *emb_h, *keys, *qs, *vals, *gated, *inv;   // inv: RoPE's
    float *lg, *wt, *A, *As, *Dx, *sh, *logits, *y, *cv, *qkvb;   // As: the shared expert's activations (decode)
    int32_t *ind, *sel, *done, *nsel, *zero, *perm, *ntiles;
    // A chunk's input: its slot of hin (pinned) = {pos, ids[T]}, copied to din on the device first.  The kernels read the position
    // from din, so a CUDA graph of T decode rows serves every position.  emb_h (PLE rows) and hlog (logits) are pinned too.
    int32_t *hin, *din;
    float* hlog;
    int nb, nple, cap, no_graph;   // QSA block slots; PLE layers; capturing a graph; NSLM_NO_GRAPH
    int ple_threads;               // Q4E_PLE_THREADS: threads for the PLE rows, a pool of ple_threads - 1 workers and the caller
    pthread_t pth[16];             // the pool: a ple_run bumps pgen, workers w < pnw run pj[w], the caller pj[0], pndone counts
    pthread_mutex_t pmu;
    pthread_cond_t pgo, pdone;
    int pgen, pnw, pndone, pquit, nth;
    PoolJob pj[16];
    double host_ms[2];             // Q4E_PROFILE: host time of the PLE prefetch and the PLE rows
    double ct_ms;                  // Q4E_PROFILE: the D4 thread's work, its layers and pairs
    long ct_layers, ct_pairs;
    cudaGraphExec_t graph[4][MV_MAXT + 1];   // [with logits + 2 keep][rows]
    // The MTP head (Q4EOpts.mtp): it runs one row behind each forward. Hm: its streams; Hall[-1]: the last main row's
    // streams (its next input); lrow: the last row's logits in logits (-1: none); keep: a q4e_forward_keep of nkeep rows
    Layer M;
    W m_fe, m_fh, m_down, m_up;
    float *m_ne, *m_nh, *m_norm, *Hm, *ones, *mlog, *dprob, *hprob;   // dprob, hprob: the head's probability of each draft
    float *dtu, *htu, *dqp, *hqp;   // a sampled q4e_draft: {temp, u[k]}, and each draft's q (KC_DRAFT_Q ids and probabilities)
    int32_t *dqid, *hqid;
    int dqn;                        // KC_DRAFT_Q after a sampled q4e_draft, else 0
    int mtp, ndraft, lrow, keep, nkeep;   // ndraft: Q4EOpts.mtp, the most drafts (and state copies)
    int tail_skip, tail_renorm;           // Q4EOpts.tail_skip; Q4E_TAIL_RENORM=1 (an A/B): the kept weights scaled up
    int head_cached;                      // the MTP head's experts are the cache's sparse layer n_layer (layer 0's format)
    // Plan D4 (Q4EOpts.cpu_tail): the decode tail's missed experts on the CPU, from the pinned arena. Mapped host memory:
    // cj (the pick's list), cx (the rows' x), cy (each pair's output), hs (flag at 0, ack at 16); dseq counts hand-offs
    int cpu_tail;
    volatile int cfwd, cquit;   // a forward is in flight (the thread spins), stop
    int cstart;
    pthread_mutex_t cmu;
    pthread_cond_t ccv;         // the thread sleeps here between forwards
    int32_t *cj_h, *cj_d;
    float *cx_h, *cx_d, *cy_h, *cy_d, *cbuf;
    uint32_t *hs_h, *hs_d, *dseq;
    unsigned long long* dts;   // Q4E_PROFILE: the hand-off's GPU times (kq4_signal, kq4_wait)
    const uint8_t* arena_h;
    int32_t* ebias;   // [units][3]: each unit's exponent biases
    pthread_t cth;
    cudaGraphExec_t dgraph[2][Q4E_MAX_DRAFTS + 1];   // q4e_draft of k rows: [sampled][k]
    FILE* routes;     // Q4E_ROUTES=FILE: each forward's int32 T, then its routed experts [n_layer][T][top_k] (tools/qwen4_cache_sim.py)
    int32_t *droute, *hroute;
    float *Gb, *Ub;   // prompt rows: the gate and up outputs of the grouped GEMMs
    MmTile* tiles;
    int prof, nev;                     // Q4E_PROFILE=1: CUDA events between the stages of a forward
    cudaEvent_t ev[Q4E_NEV];   // a pass of 32,768 rows marks about 16,000 stages
    int evstage[Q4E_NEV];
    double stage_ms[16];
    void** dev;   // allocations, freed at close
    int ndev, cdev;
    void** host;
    int nhost, chost;
    char err[256];
};

// The stages of a forward, for Q4E_PROFILE. PROJ: projections; ROPE: head norms, RoPE, K / V stores; QSA: the indexer's
// block keys, scores and top-k; ATTN: the attention itself
enum { ST_EMBED, ST_PLE, ST_HC, ST_PROJ, ST_ROPE, ST_QSA, ST_ATTN, ST_GDN, ST_ROUTE, ST_EXPERT, ST_HEAD, ST_N };
static const char* const kStage[ST_N] = {"embed", "ple", "hyper-connections", "projections", "norm + rope + store",
                                         "qsa select", "attention", "gated deltanet", "router + cache", "experts", "head"};

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
        if (t->enc == NS_TCQ) tab[i].gs = t->group;   // the GEMM's L | K << 8 | T << 16
    }
    *w = (W) {t->enc, t->slices, t->rows, t->cols, tab[0], (const WSlice*) dmem(e, tab, sizeof(WSlice) * (size_t) t->slices), t->group};
    free(tab);
}
// A row of the merged input projections: attention's q (and gate), k, v and indexer q heads and key; DeltaNet's qkv, z, b, a
static int att_ld(const Qwen4Cfg* c) { return c->n_head * 2 * c->head_dim + 2 * c->n_kv * c->head_dim + (c->ix_heads + 1) * c->ix_dim; }
static int gdn_ld(const Qwen4Cfg* c) { return 2 * c->lin_hk * c->lin_dk + 2 * c->lin_hv * c->lin_dv + 2 * c->lin_hv; }
// Dense weights of the same input (BF16, Q8 or Q4, one encoding and width) as one in VRAM: their rows in order
static void load_cat(Q4Eng* e, W* w, const char* const* names, int n) {
    const NsTensor* t[4];
    int rows = 0;
    for (int i = 0; i < n; ++i) {
        if (!(t[i] = ns_find(&e->nm, names[i]))) { fail(e, "%s: missing", names[i]); return; }
        if (t[i]->enc != t[0]->enc || (t[0]->enc != NS_BF16 && t[0]->enc != NS_Q4 && t[0]->enc != NS_Q8) || t[i]->cols != t[0]->cols ||
            t[i]->group != t[0]->group || t[i]->slices != 1 || (t[0]->enc != NS_BF16 && t[0]->group != 32 && t[0]->group != 64)) {
            fail(e, "%s and %s: not one dense encoding and width", names[0], names[i]);
            return;
        }
        rows += t[i]->rows;
    }
    WSlice ws;
    memset(&ws, 0, sizeof ws);
    for (int s = 0; s < 4; ++s) {
        uint64_t len = 0;
        for (int i = 0; i < n; ++i) len += t[i]->s[s].len;
        uint8_t* h = len ? (uint8_t*) malloc(len) : NULL;
        if (!h) continue;
        for (int i = 0, at = 0; i < n; at += (int) t[i++]->s[s].len) memcpy(h + at, t[i]->s[s].p, t[i]->s[s].len);
        ws.p[s] = dmem(e, h, len);
        free(h);
    }
    if (t[0]->enc != NS_BF16) ws.gs = t[0]->group == 32 ? 5 : 6;
    *w = (W) {t[0]->enc, 1, rows, t[0]->cols, ws, (const WSlice*) dmem(e, &ws, sizeof ws), 0};
}
// A BF16 vector of n values (any shape) as f32 in VRAM
static float* load_v(Q4Eng* e, int64_t n, const char* fmt, ...) {
    char name[192];
    NAME(name, fmt);
    const NsTensor* t = ns_find(&e->nm, name);
    if (!t || t->enc != NS_BF16 || (int64_t) t->slices * t->rows * t->cols != n) { fail(e, "%s: missing, or not BF16 with %lld values", name, (long long) n); return NULL; }
    float* v = (float*) malloc(sizeof(float) * (size_t) n);
    for (int64_t i = 0; i < n; ++i) {   // memcpy: a file's tensor need not be 2-byte aligned
        uint16_t h;
        memcpy(&h, t->s[0].p + 2 * i, 2);
        const uint32_t u = (uint32_t) h << 16;
        memcpy(&v[i], &u, 4);
    }
    float* d = (float*) dmem(e, v, sizeof(float) * (size_t) n);
    free(v);
    return d;
}

static const uint32_t* G(const Q4Eng* e, const W* w) { return w->fmt == MF_SEED4P4 ? e->g32 : e->g24; }
// y[T][rows] = W x[T][cols]: the matvec for up to MV_MAXT rows, the tensor-core GEMM (BF16 inputs) for more
static void mv_on(Q4Eng* e, cudaStream_t s, const W* w, const float* x, int xs, float* y, int T) {   // x rows xs apart
    if (T <= MV_MAXT) kc_mv(s, w->fmt, w->w, w->cols, w->rows, x, xs, y, w->rows, T, 0, G(e, w));
    else kc_mm(s, w->fmt, w->w, w->cols, w->rows, x, xs, y, w->rows, T, 0, G(e, w));
}
static void mvx(Q4Eng* e, const W* w, const float* x, int xs, float* y, int T) { mv_on(e, e->s, w, x, xs, y, T); }
static void mv(Q4Eng* e, const W* w, const float* x, float* y, int T) { mvx(e, w, x, w->cols, y, T); }

// Layer L named lp (attention if attn): main layer l, or the MTP head's layer (l < 0: no PLE, its experts in VRAM)
static void load_layer(Q4Eng* e, Layer* L, const char* lp, int attn, int l) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d, cd = 2 * c->lin_hk * c->lin_dk + c->lin_hv * c->lin_dv;
    for (int b = 0; b < 2; ++b) {
        const char* hc = b ? "mlp_hyper_connection" : "attn_hyper_connection";
        L->hc_norm[b] = load_v(e, HD, "%s.%s.hc_norm.weight", lp, hc);
        char n[2][192];
        snprintf(n[0], sizeof n[0], "%s.%s.input_mix_weight_down.weight", lp, hc);
        snprintf(n[1], sizeof n[1], "%s.%s.block_inject_weight.weight", lp, hc);
        load_cat(e, &L->hc_di[b], (const char* const[]) {n[0], n[1]}, 2);
        load_w(e, &L->hc_up[b], 0, "%s.%s.input_mix_weight_up.weight", lp, hc);
    }
    if (attn) {
        const int nb = e->max_pos / c->ix_ratio + 1;
        const char* in[4] = {"q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"};
        char n[4][192];
        for (int i = 0; i < 4; ++i) snprintf(n[i], sizeof n[i], "%s.self_attn.%s.weight", lp, in[i]);
        load_cat(e, &L->qkvi, (const char* const[]) {n[0], n[1], n[2], n[3]}, 4);
        if (!e->err[0] && L->qkvi.rows != att_ld(c)) fail(e, "%s: attention projections of %d rows, not %d", lp, L->qkvi.rows, att_ld(c));
        load_w(e, &L->o, 0, "%s.self_attn.o_proj.weight", lp);
        L->qn = load_v(e, c->head_dim, "%s.self_attn.q_norm.weight", lp);
        L->kn = load_v(e, c->head_dim, "%s.self_attn.k_norm.weight", lp);
        L->iqn = load_v(e, c->ix_dim, "%s.self_attn.indexer.q_layernorm.weight", lp);
        L->ikn = load_v(e, c->ix_dim, "%s.self_attn.indexer.k_layernorm.weight", lp);
        L->K = (uint16_t*) dmem(e, NULL, sizeof(uint16_t) * (size_t) e->max_pos * c->n_kv * c->head_dim);
        L->V = (uint16_t*) dmem(e, NULL, sizeof(uint16_t) * (size_t) e->max_pos * c->n_kv * c->head_dim);
        L->raw = (float*) dmem(e, NULL, sizeof(float) * (size_t) e->max_pos * c->ix_dim);
        L->blk = (float*) dmem(e, NULL, sizeof(float) * (size_t) nb * c->ix_dim);
    } else {
        const char* in[4] = {"qkv", "z", "b", "a"};
        char n[4][192];
        for (int i = 0; i < 4; ++i) snprintf(n[i], sizeof n[i], "%s.linear_attn.in_proj_%s.weight", lp, in[i]);
        load_cat(e, &L->in, (const char* const[]) {n[0], n[1], n[2], n[3]}, 4);
        if (!e->err[0] && L->in.rows != gdn_ld(c)) fail(e, "%s: DeltaNet in_proj of %d rows, not %d", lp, L->in.rows, gdn_ld(c));
        load_w(e, &L->out, 0, "%s.linear_attn.out_proj.weight", lp);
        L->gw.conv_w = load_v(e, (int64_t) cd * c->lin_conv, "%s.linear_attn.conv1d.weight", lp);
        L->gw.A_log = load_v(e, c->lin_hv, "%s.linear_attn.A_log", lp);
        L->gw.dt_bias = load_v(e, c->lin_hv, "%s.linear_attn.dt_bias", lp);
        L->gw.norm_w = load_v(e, c->lin_dv, "%s.linear_attn.norm.weight", lp);
        L->conv = (float*) dmem(e, NULL, sizeof(float) * qwen4_gdn_conv_len(c));
        L->S = (float*) dmem(e, NULL, sizeof(float) * qwen4_gdn_state_len(c));
    }
    char gn[2][192];   // the router's rows, then the shared expert's gate row: one GEMM of x
    snprintf(gn[0], sizeof gn[0], "%s.mlp.gate.weight", lp);
    snprintf(gn[1], sizeof gn[1], "%s.mlp.shared_expert_gate.weight", lp);
    load_cat(e, &L->gate, (const char* const[]) {gn[0], gn[1]}, 2);
    if (!e->err[0] && L->gate.rows != c->n_exp + 1) fail(e, "%s: router of %d rows, not %d + 1", lp, L->gate.rows, c->n_exp);
    load_w(e, &L->sh_g, 0, "%s.mlp.shared_expert.gate_proj.weight", lp);
    load_w(e, &L->sh_u, 0, "%s.mlp.shared_expert.up_proj.weight", lp);
    load_w(e, &L->sh_d, 0, "%s.mlp.shared_expert.down_proj.weight", lp);
    W* ex[3] = {&L->eg, &L->eu, &L->ed};
    const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
    char name[3][160];
    const NsTensor* et[3];
    int incache = e->cache;   // the MTP head's experts join the cache (sparse layer n_layer) only in layer 0's format
    for (int i = 0; i < 3; ++i) {
        snprintf(name[i], sizeof name[i], "%s.mlp.switch_mlp.%s.weight", lp, proj[i]);
        et[i] = ns_find(&e->nm, name[i]);
        if (l < 0) incache = incache && et[i] && et[i]->enc == e->et[0][i]->enc && et[i]->s[0].len == e->et[0][i]->s[0].len;
    }
    if (l < 0) e->head_cached = incache;
    for (int i = 0; i < 3; ++i) {
        const NsTensor* t = et[i];
        if (incache) e->et[l >= 0 ? l : c->n_layer][i] = t;
        if (!incache) load_w(e, ex[i], e->host_experts && l >= 0, "%s", name[i]);
        else if (!t || t->enc == NS_SEED6P8 || t->slices != c->n_exp) fail(e, "%s: missing, P = 8, or not %d experts", name[i], c->n_exp);
        else {
            memset(ex[i], 0, sizeof *ex[i]);
            ex[i]->fmt = t->enc; ex[i]->slices = t->slices; ex[i]->rows = t->rows; ex[i]->cols = t->cols; ex[i]->group = t->group;
        }
    }
    if (l >= 0 && c->ple[l] >= 0) {
        load_w(e, &L->key, 0, "%s.ple.key_proj.weight", lp);
        load_w(e, &L->val, 0, "%s.ple.value_proj.weight", lp);
        L->nk = load_v(e, HD, "%s.ple.norm_key.weight", lp);
        L->nq = load_v(e, HD, "%s.ple.norm_query.weight", lp);
        L->nc = load_v(e, HD, "%s.ple.norm_conv.weight", lp);
        L->pw = load_v(e, (int64_t) HD * c->ple_conv, "%s.ple.conv1d.weight", lp);
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
    const int E = c->n_exp, NL = c->n_layer + (e->mtp && e->head_cached), units = NL * E;   // the head: sparse layer n_layer
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
    const int nstage = E < e->mt * c->top_k ? E : e->mt * c->top_k;   // the most units one admit brings
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
    if (e->cpu_tail) e->ebias = (int32_t*) calloc((size_t) units * 3, sizeof(int32_t));
    for (int l = 0; l < NL; ++l)
        for (int i = 0; i < 3; ++i) {
            const NsTensor* t = e->et[l][i];
            if (!t || t->enc != e->et[0][i]->enc || t->s[0].len != e->et[0][i]->s[0].len) { fail(e, "layer %d: experts differ from layer 0's", l); free(tab); return; }
            if ((t->enc == NS_Q4 || t->enc == NS_Q8) && t->group != 32 && t->group != 64) { fail(e, "%s: affine group %d not supported", t->name, t->group); free(tab); return; }
            for (int x = 0; x < E; ++x) {
                WSlice* w = &tab[((size_t) l * 3 + i) * E + x];
                for (int s = 0; s < 4 && host; ++s)
                    if (len[i][s]) memcpy(host + ((uint64_t) l * E + x) * ub + p->off[i][s], t->s[s].p + len[i][s] * (uint64_t) x, len[i][s]);
                if (t->enc == NS_SEED4 || t->enc == NS_SEED4P4) memcpy(&w->eb, t->s[2].p + 4 * (size_t) x, 4);
                if (e->ebias) e->ebias[((size_t) l * E + x) * 3 + i] = w->eb;
                if (t->enc == NS_Q4 || t->enc == NS_Q8) w->gs = t->group == 32 ? 5 : 6;
                if (t->enc == NS_TCQ) w->gs = t->group;
            }
            for (int s = 0; s < 4 && host; ++s) st_dontneed(t->s[s].p, t->s[s].len);   // the arena holds them: the file pages go
        }
    p->host = (const uint8_t*) dh;
    e->arena_h = host;
    p->vram = (uint8_t*) dmem(e, NULL, ub * (uint64_t) p->slots);
    int32_t* us = (int32_t*) malloc(4 * (size_t) units);
    int32_t* su = (int32_t*) malloc(4 * (size_t) p->slots);
    for (int u = 0; u < units; ++u) us[u] = -1;
    for (int s = 0, x = 0; x < E && s < p->slots; ++x)   // round-robin: every layer starts with the same share
        for (int l = 0; l < NL && s < p->slots; ++l, ++s) {
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
    for (int l = 0; l < NL && p->tab; ++l) {
        Layer* L = l < c->n_layer ? &e->L[l] : &e->M;
        L->eg.tab = p->tab + ((size_t) l * 3 + 0) * E;
        L->eu.tab = p->tab + ((size_t) l * 3 + 1) * E;
        L->ed.tab = p->tab + ((size_t) l * 3 + 2) * E;
    }
    free(us); free(su); free(tab);
}

void q4e_cache_stats(Q4Eng* e, uint64_t* hits, uint64_t* misses, int* slots) {   // the layers' admits and the prefetch's
    uint32_t st[2] = {0, 0}, sp[2] = {0, 0};
    if (e->cache) cudaMemcpy(st, e->pool.stats, 8, cudaMemcpyDeviceToHost);
    if (e->pf_ok) cudaMemcpy(sp, e->pf.stats, 8, cudaMemcpyDeviceToHost);
    *hits = (uint64_t) st[0] + sp[0];
    *misses = (uint64_t) st[1] + sp[1];
    *slots = e->cache ? e->pool.slots : 0;
}

typedef struct { float** p; size_t n; } Buf;   // a buffer of a stage's region: its pointer and floats
static size_t bufs_len(const Buf* b, int n) {   // floats, each buffer on a 256-byte boundary
    size_t len = 0;
    for (int i = 0; i < n; ++i) len += (b[i].n + 63) & ~(size_t) 63;
    return len;
}
static void bufs_set(const Buf* b, int n, float* w) {
    for (int i = 0; i < n; ++i) { *b[i].p = w; w += (b[i].n + 63) & ~(size_t) 63; }
}

Q4Eng* q4e_open(const char* model_dir, const Q4EOpts* o, char* err, int errlen) {
    int ndev = 0;   // else a machine without a GPU fails later as "out of VRAM"
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); return NULL; }
    Q4Eng* e = (Q4Eng*) calloc(1, sizeof *e);
    if (qwen4_cfg_load(&e->c, model_dir, err, errlen) || ns_open(&e->nm, model_dir, err, errlen)) { free(e); return NULL; }
    if (ns_add_banks(&e->nm, model_dir, err, errlen)) { ns_close(&e->nm); free(e); return NULL; }   // trellis experts (.blkt)
    const Qwen4Cfg* c = &e->c;
    if (c->n_exp > BUCKET_MAX_EXPERTS || c->d > Q4_ROW_MAX || c->rot_dim > 1024) {
        snprintf(err, (size_t) errlen, "%d experts (at most %d), hidden size %d (at most %d), RoPE width %d (at most 1024)", c->n_exp,
                 BUCKET_MAX_EXPERTS, c->d, Q4_ROW_MAX, c->rot_dim);
        ns_close(&e->nm); free(e); return NULL;
    }
    e->max_pos = o && o->max_pos > 0 ? o->max_pos : 4096;
    size_t vfree = 0, vtotal = 0;
    cudaMemGetInfo(&vfree, &vtotal);
    const int mt = o && o->rows > 0 ? o->rows : vtotal >= ((size_t) 64 << 30) ? Q4E_BIG_ROWS : Q4E_MAX_ROWS;
    const int ctx = (e->max_pos + Q4E_LOGIT_ROWS - 1) / Q4E_LOGIT_ROWS * Q4E_LOGIT_ROWS;
    e->mt = mt < Q4E_LOGIT_ROWS ? Q4E_LOGIT_ROWS : mt > Q4E_PASS_ROWS ? Q4E_PASS_ROWS : mt > ctx ? ctx : mt;
    const int pr = o && o->pass_rows > 0 ? o->pass_rows : vtotal < ((size_t) 32 << 30) ? Q4E_SMALL_PASS_ROWS : Q4E_PASS_ROWS;
    e->xr = (e->max_pos + e->mt - 1) / e->mt * e->mt;
    if (e->xr > pr) e->xr = pr < e->mt ? e->mt : pr / e->mt * e->mt;
    e->host_experts = !(o && o->vram_experts);
    e->tail_skip = o ? o->tail_skip : 0;
    e->cpu_tail = o ? o->cpu_tail : 0;
    e->tail_renorm = getenv("Q4E_TAIL_RENORM") && atoi(getenv("Q4E_TAIL_RENORM")) > 0;
    e->cache = e->host_experts && o && o->cache_gb != 0;
    e->stream = e->cache && o->stream_experts;
    if (e->cpu_tail && (!e->cache || e->stream || !cpu_seed4_on()))
        fail(e, "--cpu-tail needs --expert-cache-gb, pinned experts and the CPU seed kernel (a Clang build)");
    if (e->cpu_tail < 0 || e->cpu_tail > 16) fail(e, "--cpu-tail: 0 to 16 experts a layer");
    if (o && o->stream_experts && !e->cache) { snprintf(err, (size_t) errlen, "--stream-experts needs --expert-cache-gb"); ns_close(&e->nm); free(e); return NULL; }
    if (o && o->ple_gguf) {
        if (ple_open(&e->pt, o->ple_gguf, "per_layer_token_embd.weight", err, errlen)) { ns_close(&e->nm); free(e); return NULL; }
        e->table = &e->pt;
    }
    ns_rows_init(&e->rows);
    if (cudaStreamCreate(&e->s) || cudaStreamCreateWithFlags(&e->s2, cudaStreamNonBlocking) ||
        cudaEventCreateWithFlags(&e->ev_x, cudaEventDisableTiming) || cudaEventCreateWithFlags(&e->ev_sh, cudaEventDisableTiming))
        fail(e, "CUDA streams: %s", cudaGetErrorString(cudaGetLastError()));   // e.g. no CUDA-capable device
    e->g24 = (uint32_t*) dmem(e, e->rows.g24, sizeof e->rows.g24);
    e->g32 = (uint32_t*) dmem(e, e->rows.g32, sizeof e->rows.g32);
    // the embedding in mapped host memory: a token reads one row over PCIe, and its 0.4 to 0.7 GB go to the expert cache
    load_w(e, &e->embed, 1, "language_model.model.embed_tokens.weight");
    load_w(e, &e->head, 0, "language_model.lm_head.weight");
    load_w(e, &e->mix_down, 0, "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight");
    load_w(e, &e->mix_up, 0, "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight");
    e->mix_norm = load_v(e, (int64_t) c->hc * c->d, "language_model.model.hyper_connection_mixer.hc_norm.weight");
    float inv[512];
    kq4_rope_inv(c, inv);
    e->inv = (float*) dmem(e, inv, sizeof(float) * (size_t) (c->rot_dim / 2));
    for (int l = 0; l < c->n_layer && !e->err[0]; ++l) {
        if (c->ple[l] >= 0) qwen4_ngram_init(c, c->ple[l], &e->ng[c->ple[l]]);
        if (c->ple[l] >= 0 && e->table && (c->ple[l] > 0 || e->table->rows != e->ng[0].rows))
            fail(e, "the PLE GGUF has %lld rows, the model's table %lld (and one PLE layer)", (long long) e->table->rows, (long long) e->ng[0].rows);
        char lp[48];
        snprintf(lp, sizeof lp, "language_model.model.layers.%d", l);
        load_layer(e, &e->L[l], lp, c->attn[l], l);
    }
    if (o && o->mtp > 0 && !e->err[0] && !ns_find(&e->nm, "mtp.fc_embedding.weight")) fail(e, "no MTP head (mtp.*) in %s", model_dir);
    else if (o && o->mtp > 0 && !e->err[0]) {   // its experts in VRAM
        e->m_ne = load_v(e, c->d, "mtp.pre_fc_norm_embedding.weight");
        e->m_nh = load_v(e, (int64_t) c->hc * c->d, "mtp.pre_fc_norm_hidden.weight");
        load_w(e, &e->m_fe, 0, "mtp.fc_embedding.weight");
        load_w(e, &e->m_fh, 0, "mtp.fc_hidden.weight");
        load_w(e, &e->m_down, 0, "mtp.hyper_connection_mixer.input_mix_weight_down.weight");
        load_w(e, &e->m_up, 0, "mtp.hyper_connection_mixer.input_mix_weight_up.weight");
        e->m_norm = load_v(e, (int64_t) c->hc * c->d, "mtp.hyper_connection_mixer.hc_norm.weight");
        load_layer(e, &e->M, "mtp.layers.0", 1, -1);
        e->mtp = 1;
        e->ndraft = o->mtp < Q4E_MAX_DRAFTS ? o->mtp : Q4E_MAX_DRAFTS;
    }
    const size_t T = (size_t) e->mt, HD = (size_t) c->hc * c->d, cd = (size_t) 2 * c->lin_hk * c->lin_dk + (size_t) c->lin_hv * c->lin_dv;
    const size_t vd = (size_t) c->lin_hv * c->lin_dv, kf = (size_t) c->top_k * c->ff_exp;
    const size_t nb = (size_t) e->max_pos / c->ix_ratio + 1;
#define F(n) (float*) dmem(e, NULL, sizeof(float) * (n))
#define I(n) (int32_t*) dmem(e, NULL, sizeof(int32_t) * (n))
    e->x = F(T * c->d); e->Hall = e->H = F((size_t) (e->xr + 1) * HD) + HD; e->inj = F(T * c->hc); e->br = F(T * c->d);
    e->logits = F((size_t) Q4E_LOGIT_ROWS * c->vocab);
    // Each stage's buffers live only inside the stage (x, br and inj carry between stages), so the stages share one region
    const size_t ka = kf > (size_t) c->ff_shared ? kf : (size_t) c->ff_shared;
    const Buf mix[] = {{&e->normed, T * HD}, {&e->lo, T * (c->hc_rank + c->hc)}, {&e->up, T * HD}};
    const Buf pl[] = {{&e->emb, T * c->ple_dim}, {&e->keys, T * HD}, {&e->qs, T * HD}, {&e->vals, T * c->d}, {&e->gated, T * HD}};
    const Buf att[] = {{&e->qp, T * att_ld(c)}, {&e->o, T * c->n_head * c->head_dim}, {&e->sc, T * nb}, {&e->part, kq4_attend_part_len(c)}};
    const Buf lin[] = {{&e->y, T * vd}, {&e->cv, T * cd}, {&e->qkvb, T * gdn_ld(c)}};
    const Buf ffn[] = {{&e->lg, T * (c->n_exp + 1)}, {&e->wt, T * c->top_k}, {&e->A, T * ka}, {&e->Gb, T * ka}, {&e->Ub, T * ka},
                       {&e->Dx, T * c->top_k * c->d}, {&e->sh, T * c->d}, {&e->As, MV_MAXT * (size_t) c->ff_shared}};
    const Buf* grp[] = {mix, pl, att, lin, ffn};
    const int ng[] = {3, 5, 4, 3, 8};
    size_t most = 0;
    for (int i = 0; i < 5; ++i) if (bufs_len(grp[i], ng[i]) > most) most = bufs_len(grp[i], ng[i]);
    float* work = F(most);
    for (int i = 0; i < 5; ++i) bufs_set(grp[i], ng[i], work);
    e->din = I(T + 1); e->ind = I(T * c->top_k); e->sel = I(T * (c->ix_budget / c->ix_ratio)); e->done = I(T); e->nsel = I(T); e->zero = I(T);
    e->perm = I(T * c->top_k); e->ntiles = I(2 * (size_t) c->n_exp + 3);
    e->tiles = (MmTile*) dmem(e, NULL, sizeof(MmTile) * (T * c->top_k / MMT_BN + (size_t) c->n_exp));
    if (e->mtp) {
        float* one = (float*) malloc(sizeof(float) * T * c->hc);
        for (size_t i = 0; i < T * c->hc; ++i) one[i] = 1.0f;
        e->ones = (float*) dmem(e, one, sizeof(float) * T * c->hc);
        free(one);
        e->Hm = F(T * HD); e->mlog = F((size_t) c->vocab); e->dprob = F(Q4E_MAX_DRAFTS);
        e->hprob = (float*) pin(e, sizeof(float) * Q4E_MAX_DRAFTS);
        e->dtu = F(1 + Q4E_MAX_DRAFTS); e->dqp = F(Q4E_MAX_DRAFTS * KC_DRAFT_Q); e->dqid = I(Q4E_MAX_DRAFTS * KC_DRAFT_Q);
        e->htu = (float*) pin(e, sizeof(float) * (1 + Q4E_MAX_DRAFTS));
        e->hqp = (float*) pin(e, sizeof(float) * Q4E_MAX_DRAFTS * KC_DRAFT_Q);
        e->hqid = (int32_t*) pin(e, sizeof(int32_t) * Q4E_MAX_DRAFTS * KC_DRAFT_Q);
        for (int l = 0; l < c->n_layer; ++l) {
            Layer* L = &e->L[l];
            if (!c->attn[l]) { L->kconv = F(e->ndraft * qwen4_gdn_conv_len(c)); L->kS = F(e->ndraft * qwen4_gdn_state_len(c)); }
            if (c->ple[l] >= 0) L->khist = F(e->ndraft * qwen4_ple_hist_len(c));
        }
    }
#undef F
#undef I
    if (e->cache && !e->err[0]) {   // after the buffers: at most the free VRAM less the reserve (cache_gb < 0: all of it)
        size_t fr = 0, to = 0;
        const double res = nslm_vram_reserve_mb();
        cudaMemGetInfo(&fr, &to);
        const double most = ((double) fr - res * 1048576.0) / 1e9;
        if (o->cache_gb > most) fprintf(stderr, "qwen4: expert cache %.2f GB, not %.2f: %.0f MB of VRAM stay free (NSLM_VRAM_RESERVE_MB)\n", most, o->cache_gb, res);
        build_cache(e, o->cache_gb > 0 && o->cache_gb < most ? o->cache_gb : most);
    }
    if (e->cpu_tail && e->et[0][0] && e->et[0][0]->enc == NS_TCQ) fail(e, "--cpu-tail: the CPU computes seed experts only, not trellis ones");
    if (e->cpu_tail && !e->err[0]) {   // plan D4: the mapped list, rows and outputs, the hand-off words, the CPU thread
        const size_t P = (size_t) MV_MAXT * c->top_k;
        if (e->et[0][0]->enc != NS_SEED4) fail(e, "--cpu-tail: P = 3 seed experts only");
        void* m[4] = {NULL, NULL, NULL, NULL};
        const size_t len[4] = {4 * (2 + 4 * P), sizeof(float) * MV_MAXT * c->d, sizeof(float) * P * c->d, 4 * 64};
        for (int i = 0; i < 4 && !e->err[0]; ++i) {
            if (cudaHostAlloc(&m[i], len[i], cudaHostAllocMapped) != cudaSuccess) { fail(e, "--cpu-tail: cannot map host memory"); break; }
            memset(m[i], 0, len[i]);
            track(&e->host, &e->nhost, &e->chost, m[i]);
        }
        if (!e->err[0]) {
            e->cj_h = (int32_t*) m[0]; e->cx_h = (float*) m[1]; e->cy_h = (float*) m[2]; e->hs_h = (uint32_t*) m[3];
            cudaHostGetDevicePointer((void**) &e->cj_d, m[0], 0); cudaHostGetDevicePointer((void**) &e->cx_d, m[1], 0);
            cudaHostGetDevicePointer((void**) &e->cy_d, m[2], 0); cudaHostGetDevicePointer((void**) &e->hs_d, m[3], 0);
            e->dseq = (uint32_t*) dmem(e, NULL, 4);
            e->dts = (unsigned long long*) dmem(e, NULL, 32);
            e->cbuf = (float*) malloc(sizeof(float) * ct_floats(c));
            cpu_seed4_table();
            pthread_mutex_init(&e->cmu, NULL);
            pthread_cond_init(&e->ccv, NULL);
            if (!e->cbuf || pthread_create(&e->cth, NULL, ct_thread, e)) fail(e, "--cpu-tail: no CPU thread");
            else e->cstart = 1;
        }
    }
    for (int l = 0; l < c->n_layer; ++l) if (c->ple[l] + 1 > e->nple) e->nple = c->ple[l] + 1;
    e->nb = (int) nb;
    e->hin = (int32_t*) pin(e, sizeof(int32_t) * (e->xr / e->mt) * (T + 1));   // a slot of T + 1 for each chunk of a pass
    e->emb_h = (float*) pin(e, sizeof(float) * e->xr * (size_t) c->ple_dim * (size_t) (e->nple ? e->nple : 1));
    e->hlog = (float*) pin(e, sizeof(float) * (size_t) Q4E_LOGIT_ROWS * c->vocab);
    e->no_graph = getenv("NSLM_NO_GRAPH") != NULL;
    if (e->cache && !e->stream && !e->err[0] && e->pool.slots >= 3 * c->n_exp && c->n_exp <= CACHE_MAX_PER_LAYER && !getenv("Q4E_NO_PREFETCH")) {
        int32_t ids[CACHE_MAX_PER_LAYER];
        int lo = 0, hi = 0;   // the copy stream first: its blocks take SMs ahead of the prompt's kernels
        for (int i = 0; i < c->n_exp; ++i) ids[i] = i;
        e->pf = e->pool;
        e->pf.jobs = (int32_t*) dmem(e, NULL, 8 * (size_t) c->n_exp);
        e->pf.njobs = (int32_t*) dmem(e, NULL, 16);
        e->pf.stats = (uint32_t*) dmem(e, NULL, 16);
        e->all_ids = (int32_t*) dmem(e, ids, sizeof(int32_t) * (size_t) c->n_exp);
        e->pf_ok = !e->err[0] && cudaMemset(e->pf.stats, 0, 16) == cudaSuccess && cudaDeviceGetStreamPriorityRange(&lo, &hi) == cudaSuccess &&
                   cudaStreamCreateWithPriority(&e->cs, cudaStreamNonBlocking, hi) == cudaSuccess &&
                   cudaEventCreateWithFlags(&e->ev_admit, cudaEventDisableTiming) == cudaSuccess;
        for (int l = 0; e->pf_ok && l < c->n_layer; ++l) e->pf_ok = cudaEventCreateWithFlags(&e->ev_ready[l], cudaEventDisableTiming) == cudaSuccess;
        if (!e->pf_ok) fail(e, "prefetch: CUDA setup failed");
    }
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    e->ple_threads = getenv("Q4E_PLE_THREADS") ? atoi(getenv("Q4E_PLE_THREADS")) : ncpu > 16 ? 16 : ncpu > 1 ? (int) ncpu : 1;
    if (e->ple_threads < 1) e->ple_threads = 1;
    if (e->ple_threads > 16) e->ple_threads = 16;
    pthread_mutex_init(&e->pmu, NULL);
    pthread_cond_init(&e->pgo, NULL);
    pthread_cond_init(&e->pdone, NULL);
    for (e->pj[0].e = e; e->nth < e->ple_threads - 1; ++e->nth) {
        e->pj[e->nth + 1].e = e;
        if (pthread_create(&e->pth[e->nth], NULL, pool_worker, &e->pj[e->nth + 1])) break;
    }
    e->prof = getenv("Q4E_PROFILE") && atoi(getenv("Q4E_PROFILE")) > 0;
    if (getenv("Q4E_ROUTES") && !(e->routes = fopen(getenv("Q4E_ROUTES"), "wb"))) fail(e, "cannot write %s", getenv("Q4E_ROUTES"));
    if (e->routes) {
        e->droute = (int32_t*) dmem(e, NULL, sizeof(int32_t) * (size_t) c->n_layer * T * c->top_k);
        e->hroute = (int32_t*) pin(e, sizeof(int32_t) * (size_t) c->n_layer * T * c->top_k);
    }
    for (int i = 0; e->prof && i < Q4E_NEV; ++i) cudaEventCreate(&e->ev[i]);
    e->ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) e->max_pos);
    e->lrow = -1;
    if (!e->err[0] && cudaDeviceSynchronize() != cudaSuccess) fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    if (e->err[0]) { snprintf(err, (size_t) errlen, "%s", e->err); q4e_close(e); return NULL; }
    return e;
}

void q4e_close(Q4Eng* e) {
    if (!e) return;
    cudaDeviceSynchronize();
    if (e->cstart) {   // the D4 thread reads the mapped words freed below
        pthread_mutex_lock(&e->cmu);
        e->cquit = 1;
        pthread_cond_signal(&e->ccv);
        pthread_mutex_unlock(&e->cmu);
        pthread_join(e->cth, NULL);
    }
    free(e->cbuf); free(e->ebias);
    if (e->ple_threads) {   // the PLE pool (started once ple_threads is set)
        pthread_mutex_lock(&e->pmu);
        e->pquit = 1;
        pthread_cond_broadcast(&e->pgo);
        pthread_mutex_unlock(&e->pmu);
        for (int w = 0; w < e->nth; ++w) pthread_join(e->pth[w], NULL);
        pthread_mutex_destroy(&e->pmu);
        pthread_cond_destroy(&e->pgo);
        pthread_cond_destroy(&e->pdone);
    }
    if (e->prof) {
        double total = 0;
        for (int i = 0; i < ST_N; ++i) total += e->stage_ms[i];
        fprintf(stderr, "Q4E_PROFILE: %.0f ms on the GPU stream\n", total);
        for (int i = 0; i < ST_N; ++i)
            if (e->stage_ms[i] > 0) fprintf(stderr, "  %-18s %9.1f ms  %5.1f%%\n", kStage[i], e->stage_ms[i], 100 * e->stage_ms[i] / (total > 0 ? total : 1));
        fprintf(stderr, "Q4E_PROFILE: host PLE prefetch %.0f ms, PLE rows %.0f ms (%d threads)\n", e->host_ms[0], e->host_ms[1], e->ple_threads);
        if (e->ct_layers) {
            unsigned long long t[4] = {0};
            cudaMemcpy(t, e->dts, sizeof t, cudaMemcpyDeviceToHost);
            fprintf(stderr, "Q4E_PROFILE: CPU tail %.0f ms, %ld layers, %ld pairs (%.0f us a layer, %.0f us a pair)\n", e->ct_ms,
                    e->ct_layers, e->ct_pairs, 1e3 * e->ct_ms / e->ct_layers, e->ct_pairs ? 1e3 * e->ct_ms / e->ct_pairs : 0.0);
            if (t[3]) fprintf(stderr, "Q4E_PROFILE: GPU signal to ack %.0f us a layer, of which the GPU waited %.0f us (%llu layers)\n",
                              1e-3 * (double) t[1] / (double) t[3], 1e-3 * (double) t[2] / (double) t[3], t[3]);
        }
        int regs = 0, blocks = 0;
        kq4_attend_info(&e->c, &regs, &blocks);
        fprintf(stderr, "Q4E_PROFILE: prompt attention kernel %d registers a thread, %d blocks an SM\n", regs, blocks);
        for (int i = 0; i < Q4E_NEV; ++i) cudaEventDestroy(e->ev[i]);
    }
    for (int i = 0; i < e->ndev; ++i) cudaFree(e->dev[i]);
    for (int i = 0; i < e->nhost; ++i) cudaFreeHost(e->host[i]);
    for (int w = 0; w < 4; ++w) for (int t = 0; t <= MV_MAXT; ++t) if (e->graph[w][t]) cudaGraphExecDestroy(e->graph[w][t]);
    for (int s = 0; s < 2; ++s)
        for (int k = 0; k <= Q4E_MAX_DRAFTS; ++k) if (e->dgraph[s][k]) cudaGraphExecDestroy(e->dgraph[s][k]);
    free(e->dev); free(e->host); free(e->ids);
    if (e->routes) fclose(e->routes);
    if (e->s) cudaStreamDestroy(e->s);
    if (e->s2) cudaStreamDestroy(e->s2);
    if (e->ev_x) cudaEventDestroy(e->ev_x);
    if (e->ev_sh) cudaEventDestroy(e->ev_sh);
    if (e->cs) cudaStreamDestroy(e->cs);
    if (e->ev_admit) cudaEventDestroy(e->ev_admit);
    for (int l = 0; l < e->c.n_layer; ++l) if (e->ev_ready[l]) cudaEventDestroy(e->ev_ready[l]);
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
    e->lrow = -1;
    e->nkeep = 0;
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

// ---- the forward of up to mt rows ------------------------------------------------------------------------------------

// The streams of H mixed into e->x (nslm/qwen4_ops.h). down: hc_rank rows, then (inj) the hc rows of the weights that
// add the branch back, into e->inj. norm NULL: e->normed holds the normed streams already (kq4_hc_add_norm).
static void hc_mix(Q4Eng* e, const float* norm, const W* down, const W* upw, int inj, int T) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d, ld = down->rows;
    if (norm) kq4_norm1p(e->s, e->H, HD, c->d, norm, c->eps, e->normed, T);
    mv(e, down, e->normed, e->lo, T);
    kq4_hc_lo(e->s, c->hc, e->lo, T, c->hc_rank, ld, inj ? e->inj : NULL);
    mvx(e, upw, e->lo, ld, e->up, T);
    kq4_hc_mix(e->s, c->hc, c->d, e->normed, e->up, e->x, T);
}

static double host_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e3 * (double) ts.tv_sec + 1e-6 * (double) ts.tv_nsec;
}

// PLE items: rows (prefetch), else (row, head); 1 + the layer of a missing row
static int ple_items(Q4Eng* e, const void* ctx, int i0, int i1) {
    const Qwen4Cfg* c = &e->c;
    const int prefetch = *(const int*) ctx, H = prefetch ? 1 : e->ng[0].heads;
    for (int l = 0; l < c->n_layer; ++l)
        for (int i = i0; c->ple[l] >= 0 && i < i1; ++i) {
            const int t = i / H;
            if (prefetch) qwen4_ple_prefetch(&e->nm, e->table, c, &e->ng[c->ple[l]], l, e->ids, e->pos + t);
            else if (qwen4_ple_head(&e->nm, &e->rows, e->table, c, &e->ng[c->ple[l]], l, e->ids, e->pos + t, i % H,
                                    e->emb_h + ((size_t) c->ple[l] * e->xr + t) * c->ple_dim)) return l + 1;
        }
    return 0;
}

static void* pool_worker(void* arg) {   // worker w of the pool
    Q4Eng* e = ((PoolJob*) arg)->e;
    const int w = (int) ((PoolJob*) arg - e->pj);
    int gen = 0;
    pthread_mutex_lock(&e->pmu);
    for (;;) {
        while (e->pgen == gen && !e->pquit) pthread_cond_wait(&e->pgo, &e->pmu);
        if (e->pquit) break;
        gen = e->pgen;
        if (w >= e->pnw) continue;
        pthread_mutex_unlock(&e->pmu);
        PoolJob* j = &e->pj[w];
        j->rc = j->fn(e, j->ctx, j->i0, j->i1);
        pthread_mutex_lock(&e->pmu);
        if (++e->pndone == e->pnw - 1) pthread_cond_signal(&e->pdone);
    }
    pthread_mutex_unlock(&e->pmu);
    return NULL;
}

// fn over n items, split on the pool (the caller runs the first share); the first non-zero result
static int pool_run(Q4Eng* e, int n, int (*fn)(Q4Eng*, const void*, int, int), const void* ctx) {
    const int nw = n < e->nth + 1 ? n : e->nth + 1;
    for (int w = 0; w < nw; ++w) e->pj[w] = (PoolJob) {e, fn, ctx, (int) ((int64_t) n * w / nw), (int) ((int64_t) n * (w + 1) / nw), 0};
    if (nw > 1) {
        pthread_mutex_lock(&e->pmu);
        e->pnw = nw;
        e->pndone = 0;
        ++e->pgen;
        pthread_cond_broadcast(&e->pgo);
        pthread_mutex_unlock(&e->pmu);
    }
    if (nw > 0) e->pj[0].rc = fn(e, ctx, e->pj[0].i0, e->pj[0].i1);
    if (nw > 1) {
        pthread_mutex_lock(&e->pmu);
        while (e->pndone < nw - 1) pthread_cond_wait(&e->pdone, &e->pmu);
        pthread_mutex_unlock(&e->pmu);
    }
    for (int w = 0; w < nw; ++w) if (e->pj[w].rc) return e->pj[w].rc;
    return 0;
}

// ---- plan D4: the decode tail's missed experts on the CPU --------------------------------------------------------------
// A layer's list (kq4_cpu_pick) groups by expert. Pass 1: each group's gate and up rows; then SwiGLU on the host; pass 2:
// the down rows. Items are CT_ROWS-row chunks of one group's matrix, so the pool splits each pass.
#define CT_ROWS 16   // a P = 3 gate row of 2560 weights is about 2 us: items fine enough to balance 16 threads
typedef struct { int x, n, j[MV_MAXT]; } CtGroup;   // an expert and the list entries (pairs) that use it
// The scratch of G = MV_MAXT * top_k groups of MV_MAXT rows: x transposed (xt, xs), gate and up (gu), the SwiGLU
// transposed (act, axs), the down outputs (down), and tmp (a group's rows before a transpose)
typedef struct { const CtGroup* g; int pass, nitems; atomic_int next; int item[3 * MV_MAXT * 32 * 40]; float *xt, *xs, *gu, *act, *axs, *down, *tmp; } CtJob;
static size_t ct_floats(const Qwen4Cfg* c) {
    const size_t G = (size_t) MV_MAXT * c->top_k, M = MV_MAXT, D = c->d, F = c->ff_exp;
    return G * M * (D + D / 8 + 2 * F + F + F / 8 + D) + M * D;
}
static void ct_bufs(CtJob* j, float* b, const Qwen4Cfg* c) {
    const size_t G = (size_t) MV_MAXT * c->top_k, M = MV_MAXT, D = c->d, F = c->ff_exp;
    j->xt = b; j->xs = j->xt + G * M * D; j->gu = j->xs + G * M * (D / 8); j->act = j->gu + G * M * 2 * F;
    j->axs = j->act + G * M * F; j->down = j->axs + G * M * (F / 8); j->tmp = j->down + G * M * D;
}

static const uint8_t* ct_unit(Q4Eng* e, int l, int x, int i, int s) {
    return e->arena_h + ((uint64_t) l * e->c.n_exp + (uint64_t) x) * e->pool.unit_bytes + e->pool.off[i][s];
}
// One logical CPU stays free for the thread that waits on the GPU: with every CPU busy, the OS stops a worker for a time slice
static int ct_threads(const Q4Eng* e) { return e->nth > 0 ? e->nth : 1; }
// Each pool worker claims items until none are left: a worker the OS stops holds back one item, not a share
// BF16 at the GPU expert path's points (kernels_moe.cu mv_gu_body, store_y): the CPU tail differs only in sum order
static float ct_bf(float x) { return nslm_bf2f(nslm_f2bf(x)); }
static float ct_act(float g, float u) {   // bf16(silu(bf16(g)) * bf16(u)), silu on a BF16 tensor
    g = ct_bf(g);
    return ct_bf(ct_bf(g * ct_bf(1.0f / (1.0f + expf(-g)))) * ct_bf(u));
}
static int ct_items(Q4Eng* e, const void* ctx, int i0, int i1) {   // item: group << 16 | matrix << 8 | CT_ROWS-row chunk
    CtJob* j = (CtJob*) ctx;
    const Qwen4Cfg* c = &e->c;
    const size_t M = MV_MAXT, D = c->d, F = c->ff_exp;
    const int l = e->cj_h[1];
    (void) i0; (void) i1;
    for (int it; (it = atomic_fetch_add(&j->next, 1)) < j->nitems;) {
        const int g = j->item[it] >> 16, m = j->item[it] >> 8 & 255, r0 = (j->item[it] & 255) * CT_ROWS, i = j->pass ? 2 : m;
        const int rows = (int) (j->pass ? D : F), kb = (int) (j->pass ? F : D) / 8, r1 = r0 + CT_ROWS < rows ? r0 + CT_ROWS : rows;
        const CtGroup* G = &j->g[g];
        const float* xt = j->pass ? j->act + g * M * F : j->xt + g * M * D;
        const float* xs = j->pass ? j->axs + g * M * (F / 8) : j->xs + g * M * (D / 8);
        float* y = j->pass ? j->down + g * M * D : j->gu + ((size_t) g * 2 + m) * M * F;
        cpu_seed3_rows((const uint16_t*) ct_unit(e, l, G->x, i, 0), (const uint16_t*) ct_unit(e, l, G->x, i, 1),
                       e->ebias[((size_t) l * c->n_exp + G->x) * 3 + i], cpu_seed4_table(), xt, xs, G->n, kb, r0, r1, y, rows);
    }
    return 0;
}
static void ct_layer(Q4Eng* e) {
    const Qwen4Cfg* c = &e->c;
    const int n = e->cj_h[0];
    const size_t M = MV_MAXT, D = c->d, F = c->ff_exp;
    static CtGroup g[MV_MAXT * 32];
    static CtJob j;
    int ng = 0;
    if (!n) return;   // no pair: the pool sleeps
    for (int p = 0; p < n; ++p) {   // the groups, in list order
        const int x = e->cj_h[3 + 4 * p];
        int k = 0;
        while (k < ng && g[k].x != x) ++k;
        if (k == ng) g[ng++] = (CtGroup) {x, 0, {0}};
        g[k].j[g[k].n++] = p;
    }
    j.g = g; j.pass = 0; j.nitems = 0;
    ct_bufs(&j, e->cbuf, c);
    for (int k = 0; k < ng; ++k) {
        for (int r = 0; r < g[k].n; ++r) memcpy(j.tmp + r * D, e->cx_h + (size_t) e->cj_h[2 + 4 * g[k].j[r]] * D, sizeof(float) * D);
        cpu_seed_xt(j.tmp, g[k].n, (int) D / 8, j.xt + k * M * D, j.xs + k * M * (D / 8));
        for (int m = 0; m < 2; ++m) for (int ch = 0; ch * CT_ROWS < (int) F; ++ch) j.item[j.nitems++] = k << 16 | m << 8 | ch;
    }
    atomic_store(&j.next, 0);
    pool_run(e, ct_threads(e), ct_items, &j);
    j.pass = 1; j.nitems = 0;
    for (int k = 0; k < ng; ++k) {
        for (int r = 0; r < g[k].n; ++r)
            for (size_t f = 0; f < F; ++f) {
                const float gt = j.gu[((size_t) k * 2 * M + r) * F + f], up = j.gu[((size_t) (k * 2 + 1) * M + r) * F + f];
                j.tmp[r * F + f] = ct_act(gt, up);
            }
        cpu_seed_xt(j.tmp, g[k].n, (int) F / 8, j.act + k * M * F, j.axs + k * M * (F / 8));
        for (int ch = 0; ch * CT_ROWS < (int) D; ++ch) j.item[j.nitems++] = k << 16 | ch;
    }
    atomic_store(&j.next, 0);
    pool_run(e, ct_threads(e), ct_items, &j);
    for (int k = 0; k < ng; ++k)
        for (int r = 0; r < g[k].n; ++r)
            for (size_t i = 0; i < D; ++i) e->cy_h[(size_t) g[k].j[r] * D + i] = ct_bf(j.down[(k * M + r) * D + i]);
}
static void* ct_thread(void* arg) {   // answers each layer's signal (kq4_signal) after its CPU work; spins during a forward
    Q4Eng* e = (Q4Eng*) arg;
    volatile uint32_t* hs = e->hs_h;
    uint32_t done = 0;
    while (!e->cquit) {
        const uint32_t v = hs[0];
        if (v == done && !e->cfwd) {
            pthread_mutex_lock(&e->cmu);
            while (!e->cfwd && !e->cquit) pthread_cond_wait(&e->ccv, &e->cmu);
            pthread_mutex_unlock(&e->cmu);
        }
        if (v == done) continue;
        atomic_thread_fence(memory_order_acquire);   // the GPU's list and rows after its flag
        const double t0 = e->prof ? host_ms() : 0;
        ct_layer(e);
        if (e->prof) { e->ct_ms += host_ms() - t0; e->ct_layers++; e->ct_pairs += e->cj_h[0]; }
        done = v;
        atomic_thread_fence(memory_order_seq_cst);   // the outputs before the ack
        hs[16] = v;
    }
    return NULL;
}

// The PLE rows [pos, pos + T) of each PLE layer: their pages asked of the OS (prefetch), or into emb_h. Host work,
// before the rows' device work, on the pool; decode rows split by head, so their page reads overlap.
static int ple_run(Q4Eng* e, int T, int prefetch) {
    const double t0 = e->prof ? host_ms() : 0;
    const int bad = pool_run(e, prefetch ? T : T * e->ng[0].heads, ple_items, &prefetch);
    if (e->prof) e->host_ms[prefetch ? 0 : 1] += host_ms() - t0;
    return bad ? fail(e, "layer %d: a PLE table shard is missing or does not hold a row", bad - 1) : 0;
}
static int ple_rows(Q4Eng* e, int T) { return ple_run(e, T, 0); }

static void ple(Q4Eng* e, int l, int T, int row0) {   // row0: the chunk's first row in emb_h
    const Qwen4Cfg* c = &e->c;
    Layer* L = &e->L[l];
    const int HD = c->hc * c->d;
    cudaMemcpyAsync(e->emb, e->emb_h + ((size_t) c->ple[l] * e->xr + row0) * c->ple_dim, sizeof(float) * (size_t) T * c->ple_dim,
                    cudaMemcpyHostToDevice, e->s);
    mv(e, &L->key, e->emb, e->keys, T);
    kq4_norm1p(e->s, e->keys, HD, c->d, L->nk, c->eps, e->keys, T);
    mv(e, &L->val, e->emb, e->vals, T);
    kq4_norm1p(e->s, e->H, HD, c->d, L->nq, c->eps, e->qs, T);
    kq4_ple_gate(e->s, c, e->keys, e->qs, e->vals, e->gated, T);
    kq4_norm1p(e->s, e->gated, HD, c->d, L->nc, c->eps, e->keys, T);   // keys now holds the conv input
    kq4_ple_conv_keep(e->s, c, L->pw, e->keys, e->gated, L->hist, e->keep ? L->khist : NULL, e->H, T);
}

// Row t at *din + p0 + t. keys_only: the KV and QSA block-key stores, no output (the MTP head's forward rows)
static int attention(Q4Eng* e, Layer* L, int p0, int T, int keys_only) {
    const Qwen4Cfg* c = &e->c;
    const int hd = c->head_dim, kvw = c->n_kv * hd, D = c->ix_dim, R = c->ix_ratio;
    const int ld = att_ld(c);   // a row of qp: q, k, v, the indexer's q heads and key
    float* kk = e->qp + (size_t) c->n_head * 2 * hd, *vv = kk + kvw, *ixo = vv + kvw;
    mv(e, &L->qkvi, e->x, e->qp, T);
    mark(e, ST_PROJ);
    kq4_head_norm_rope_inv(e->s, c, e->qp, ld, c->n_head, 2 * hd, hd, L->qn, e->inv, p0, e->din, T);
    kq4_head_norm_rope_inv(e->s, c, kk, ld, c->n_kv, hd, hd, L->kn, e->inv, p0, e->din, T);
    kq4_store_bf16(e->s, L->K, kvw, kk, ld, p0, e->din, T);
    kq4_store_bf16(e->s, L->V, kvw, vv, ld, p0, e->din, T);
    kq4_store(e->s, L->raw, D, ixo + (size_t) c->ix_heads * D, ld, p0, e->din, T);
    kq4_head_norm_rope_inv(e->s, c, ixo, ld, c->ix_heads, D, D, L->iqn, e->inv, p0, e->din, T);
    mark(e, ST_ROPE);
    kq4_qsa_blocks_inv(e->s, c, L->raw, L->ikn, e->inv, p0, e->din, T, L->blk);
    if (keys_only) return 0;
    kq4_qsa_rows(e->s, c, p0, e->din, e->done, e->nsel, T);
    // the score grid and stride: the blocks done by the last row, or in a graph (any position) every block slot
    kq4_qsa_select(e->s, c, ixo, ld, L->blk, e->done, e->cap ? e->nb : (e->pos + p0 + T) / R, e->sc, e->sel, T);
    mark(e, ST_QSA);
    if (kq4_attend_part(e->s, c, e->qp, ld, L->K, L->V, p0, e->din, e->sel, e->nsel, e->o, e->part, T))
        return fail(e, "attention of head_dim %d, %d heads a KV head: not supported", hd, c->n_head / c->n_kv);
    mark(e, ST_ATTN);
    mv(e, &L->o, e->o, e->br, T);
    mark(e, ST_PROJ);
    return 0;
}

static int gdn(Q4Eng* e, Layer* L, int T) {
    const Qwen4Cfg* c = &e->c;
    const int cd = 2 * c->lin_hk * c->lin_dk + c->lin_hv * c->lin_dv;
    const float* z = e->qkvb + cd, *b = z + (size_t) c->lin_hv * c->lin_dv, *a = b + c->lin_hv;   // the columns of a row of qkvb
    mv(e, &L->in, e->x, e->qkvb, T);
    mark(e, ST_PROJ);
    if (kq4_gdn_ld(e->s, c, L->gw, e->qkvb, z, a, b, gdn_ld(c), L->conv, L->S, e->keep ? L->kconv : NULL, e->keep ? L->kS : NULL, e->cv, e->y, T))
        return fail(e, "Gated DeltaNet head widths not supported");
    mark(e, ST_GDN);
    mv(e, &L->out, e->y, e->br, T);
    mark(e, ST_PROJ);
    return 0;
}

// Trellis experts (NS_TCQ) in decode rows: the gathers take L, K, T (group L | K << 8 | T << 16) as arguments
static void tq_experts(Q4Eng* e, const Layer* L, int P, int k) {
    const Qwen4Cfg* c = &e->c;
    const int g = L->eg.group, d = L->ed.group;
    if (L->eu.group != g || kc_tq_mv_gu(e->s, g & 255, g >> 8 & 255, g >> 16, L->eg.tab, L->eu.tab, c->d, c->ff_exp, e->x, c->d, e->A, c->ff_exp, e->ind, P, k) ||
        kc_tq_mv_sel(e->s, d & 255, d >> 8 & 255, d >> 16, L->ed.tab, c->ff_exp, c->d, e->A, c->ff_exp, e->Dx, c->d, e->ind, P, 1))
        fail(e, "trellis experts: gate and up differ, or no kernel for L %d K %d T %d / L %d K %d T %d", g & 255, g >> 8 & 255, g >> 16,
             d & 255, d >> 8 & 255, d >> 16);
}

static void moe(Q4Eng* e, Layer* L, int l, int T) {   // l < 0: the MTP head's layer (the cache's sparse layer n_layer)
    const Qwen4Cfg* c = &e->c;
    const int k = c->top_k, P = T * k, side = T <= MV_MAXT;
    if (side) {   // decode: the shared expert on s2 while the router and the expert copy run on s; joined before the combine
        cudaEventRecord(e->ev_x, e->s);
        cudaStreamWaitEvent(e->s2, e->ev_x, 0);
        kc_mv_gu(e->s2, L->sh_g.fmt, L->sh_g.tab, L->sh_u.tab, c->d, c->ff_shared, e->x, c->d, e->As, c->ff_shared, e->zero, T, 1, G(e, &L->sh_g));
        mv_on(e, e->s2, &L->sh_d, e->As, c->ff_shared, e->sh, T);
        cudaEventRecord(e->ev_sh, e->s2);
    }
    mv(e, &L->gate, e->x, e->lg, T);
    kq4_route_ld(e->s, c, e->lg, c->n_exp + 1, e->ind, e->wt, T);
    if (e->routes && l >= 0) cudaMemcpyAsync(e->droute + (size_t) l * P, e->ind, sizeof(int32_t) * (size_t) P, cudaMemcpyDeviceToDevice, e->s);
    const int ct = e->cpu_tail && side && l >= 0;   // D4: the tail's misses to the CPU, beside the GPU's work
    if (ct) {
        kq4_cpu_pick(e->s, e->ind, e->wt, e->pool.unit_slot + (size_t) l * c->n_exp, T, k, e->cpu_tail, e->cj_d, l);
        kq4_signal(e->s, e->dseq, e->hs_d, e->x, e->cx_d, T * c->d, e->prof ? e->dts : NULL);
    } else if (e->tail_skip && e->cache && side && l >= 0)
        kq4_tail_skip(e->s, e->ind, e->wt, e->pool.unit_slot + (size_t) l * c->n_exp, T, k, e->tail_skip, e->tail_renorm);
    mark(e, ST_EXPERT);   // the router's own matvec counts with the experts; the admit and copies below with the cache
    if (e->cache && (l >= 0 || e->head_cached)) {   // bring the routed experts missing from VRAM (the pool's tables then point at them)
        kc_cache_admit(e->s, &e->pool, l >= 0 ? l : c->n_layer, e->ind, P, 0);
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
        mv(e, &L->sh_d, e->A, e->sh, T);
    } else {
        if (L->eg.fmt == NS_TCQ) tq_experts(e, L, P, k);
        else {
            kc_mv_gu(e->s, L->eg.fmt, L->eg.tab, L->eu.tab, c->d, c->ff_exp, e->x, c->d, e->A, c->ff_exp, e->ind, P, k, G(e, &L->eg));
            kc_mv_sel(e->s, L->ed.fmt, L->ed.tab, c->ff_exp, c->d, e->A, c->ff_exp, e->Dx, c->d, e->ind, P, 1, G(e, &L->ed));
        }
        cudaStreamWaitEvent(e->s, e->ev_sh, 0);
    }
    kq4_moe_combine(e->s, c->d, k, e->Dx, e->wt, e->sh, e->lg + c->n_exp, c->n_exp + 1, e->br, T);
    if (ct) {
        kq4_wait(e->s, e->dseq, e->hs_d + 16, e->prof ? e->dts : NULL);
        kq4_cpu_add(e->s, e->br, e->cy_d, e->cj_d, c->d, T);
    }
}

// The device work of a chunk of T rows: its input slot to din, the embedding into H, a layer, the head (pinned logits)
static void rows_in(Q4Eng* e, int slot, int T) {
    cudaMemcpyAsync(e->din, e->hin + (size_t) slot * (e->mt + 1), sizeof(int32_t) * (size_t) (T + 1), cudaMemcpyHostToDevice, e->s);
}
static void embed(Q4Eng* e, int T) {
    mark(e, ST_EMBED);
    kc_embed(e->s, e->embed.fmt, e->embed.w, e->c.d, e->din + 1, e->x, T);
    kq4_tile(e->s, e->c.hc, e->c.d, e->x, e->H, T);
    mark(e, ST_EMBED);
}
// Layer L on the streams H: main layer l (row0: its first row in the pass), or the MTP head's (l < 0, rows at p0 = -1)
static int run_layer(Q4Eng* e, Layer* L, int l, int attn, int T, int row0, int p0) {
    const Qwen4Cfg* c = &e->c;
    if (l >= 0 && c->ple[l] >= 0) ple(e, l, T, row0);
    mark(e, ST_PLE);
    hc_mix(e, L->hc_norm[0], &L->hc_di[0], &L->hc_up[0], 1, T);
    mark(e, ST_HC);
    if (attn ? attention(e, L, p0, T, 0) : gdn(e, L, T)) return -1;
    mark(e, attn ? ST_ATTN : ST_GDN);
    kq4_hc_add_norm(e->s, c->hc, c->d, e->H, e->br, e->inj, L->hc_norm[1], c->eps, e->normed, T);
    hc_mix(e, NULL, &L->hc_di[1], &L->hc_up[1], 1, T);
    mark(e, ST_HC);
    moe(e, L, l, T);
    kq4_hc_add(e->s, c->hc, c->d, e->H, e->br, e->inj, T);
    mark(e, ST_EXPERT);
    return e->err[0] ? -1 : 0;
}
static int layer(Q4Eng* e, int l, int T, int row0) { return run_layer(e, &e->L[l], l, e->c.attn[l], T, row0, 0); }

// The MTP head's input streams Hm[n]: fc_embedding(norm(embed(tok))) added to each stream of fc_hidden(norm(Hin))
static void head_in(Q4Eng* e, const int32_t* tok, const float* Hin, int n) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d;
    kc_embed(e->s, e->embed.fmt, e->embed.w, c->d, tok, e->x, n);
    kq4_norm1p(e->s, e->x, c->d, c->d, e->m_ne, c->eps, e->x, n);
    mv(e, &e->m_fe, e->x, e->br, n);
    kq4_norm1p(e->s, Hin, HD, HD, e->m_nh, c->eps, e->normed, n);
    mv(e, &e->m_fh, e->normed, e->Hm, n * c->hc);   // a stream a row
    kq4_hc_add(e->s, c->hc, c->d, e->Hm, e->br, e->ones, n);
}
// The head's keys for a chunk's rows: row t on token din[1 + s + t] and the main streams of row t - 1 + s (Hall[-1]
// before the first chunk), at *din - 1 + s + t. s = 1 at position 0, where no row comes before.
static int head_rows(Q4Eng* e, int T, int s) {
    float* H = e->H;
    if (!e->mtp || T - s <= 0) return 0;
    head_in(e, e->din + 1 + s, H + (ptrdiff_t) (s - 1) * e->c.hc * e->c.d, T - s);
    e->H = e->Hm;
    hc_mix(e, e->M.hc_norm[0], &e->M.hc_di[0], &e->M.hc_up[0], 1, T - s);
    const int rc = attention(e, &e->M, s - 1, T - s, 1);
    e->H = H;
    return rc;
}
static void keep_last(Q4Eng* e, int row) {   // Hall[-1] = Hall[row]: the head's next input
    const size_t HD = (size_t) e->c.hc * e->c.d;
    if (e->mtp) cudaMemcpyAsync(e->Hall - HD, e->Hall + (size_t) row * HD, sizeof(float) * HD, cudaMemcpyDeviceToDevice, e->s);
}
// Admit every expert of layer l on the main stream (after the last reader of the slots it may replace), copy the
// misses on the copy stream; ev_ready[l] when they are in VRAM
static void prefetch(Q4Eng* e, int l) {
    kc_cache_admit(e->s, &e->pf, l, e->all_ids, e->c.n_exp, 0);
    cudaEventRecord(e->ev_admit, e->s);
    cudaStreamWaitEvent(e->cs, e->ev_admit, 0);
    kc_cache_copy(e->cs, &e->pf, 8);
    cudaEventRecord(e->ev_ready[l], e->cs);
}
// At layer l's start (pre: a forward of T rows with Q4E_PREFETCH_PICKS router picks an expert, so nearly every expert
// in use): layer l's experts in VRAM, layer l + 1's on their way. The layers' own admits then find their experts resident.
static int prefetch_on(const Q4Eng* e, int T) { return e->pf_ok && (int64_t) T * e->c.top_k >= (int64_t) Q4E_PREFETCH_PICKS * e->c.n_exp; }
static void layer_start(Q4Eng* e, int l, int pre) {
    if (!pre) return;
    if (l == 0) prefetch(e, 0);
    cudaStreamWaitEvent(e->s, e->ev_ready[l], 0);
    mark(e, ST_ROUTE);
    if (l + 1 < e->c.n_layer) prefetch(e, l + 1);
}
static int enqueue(Q4Eng* e, int T, int want) {   // the rows of slot 0 through every layer; want: with the head
    rows_in(e, 0, T);
    embed(e, T);
    for (int l = 0; l < e->c.n_layer; ++l) {
        layer_start(e, l, prefetch_on(e, T));
        if (layer(e, l, T, 0)) return -1;
    }
    if (want) {   // prompt rows without logits skip the mixer and the head
        hc_mix(e, e->mix_norm, &e->mix_down, &e->mix_up, 0, T);
        mv(e, &e->head, e->x, e->logits, T);
        cudaMemcpyAsync(e->hlog, e->logits, sizeof(float) * (size_t) T * e->c.vocab, cudaMemcpyDeviceToHost, e->s);
        mark(e, ST_HEAD);
    }
    if (head_rows(e, T, e->pos == 0)) return -1;
    keep_last(e, T - 1);
    return e->err[0] ? -1 : 0;
}

// Prompt rows without logits (n <= xr), layer by layer over chunks of mt rows, so the cache brings each layer's
// experts once a pass, not once a chunk.  Each chunk sees the inputs of the row-major forward: the same logits.
static int forward_pass(Q4Eng* e, const int32_t* ids, int n) {
    const int pos0 = e->pos, nc = (n + e->mt - 1) / e->mt, HD = e->c.hc * e->c.d;
    for (int k = 0; k < nc; ++k) {
        int32_t* h = e->hin + (size_t) k * (e->mt + 1);
        h[0] = pos0 + k * e->mt;
        memcpy(h + 1, ids + (size_t) k * e->mt, sizeof(int32_t) * (size_t) (n - k * e->mt < e->mt ? n - k * e->mt : e->mt));
    }
    int rc = ple_rows(e, n);
    for (int l = -1; !rc && l < e->c.n_layer + e->mtp; ++l) {   // l = -1: the embedding; l = n_layer: the MTP head
        if (l >= 0 && l < e->c.n_layer) layer_start(e, l, prefetch_on(e, n));
        for (int k = 0; !rc && k < nc; ++k) {
            const int T = n - k * e->mt < e->mt ? n - k * e->mt : e->mt;
            e->H = e->Hall + (size_t) k * e->mt * HD;
            e->pos = pos0 + k * e->mt;
            rows_in(e, k, T);
            if (l < 0) embed(e, T);
            else if (l < e->c.n_layer) rc = layer(e, l, T, k * e->mt);
            else rc = head_rows(e, T, e->pos == 0);
        }
    }
    e->H = e->Hall;
    e->pos = pos0;
    if (rc) return -1;
    keep_last(e, n - 1);
    if (cudaStreamSynchronize(e->s) != cudaSuccess || cudaGetLastError() != cudaSuccess)
        return fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    prof_collect(e);
    e->pos += n;
    e->lrow = -1;
    return 0;
}

// Decode rows (up to MV_MAXT) run as a CUDA graph, captured once for each row count, with or without logits
static int forward_rows(Q4Eng* e, const int32_t* ids, int T, float* logits) {
    e->hin[0] = e->pos;
    memcpy(e->hin + 1, ids, sizeof(int32_t) * (size_t) T);
    if (ple_rows(e, T)) return -1;
    const int want = logits != NULL, graph = T <= MV_MAXT && !e->prof && !e->stream && !e->no_graph && !(e->mtp && !e->pos);
    cudaGraphExec_t* ge = &e->graph[want | e->keep << 1][T];
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
    if (e->cstart) { pthread_mutex_lock(&e->cmu); e->cfwd = 1; pthread_cond_signal(&e->ccv); pthread_mutex_unlock(&e->cmu); }
    if (graph ? cudaGraphLaunch(*ge, e->s) != cudaSuccess : enqueue(e, T, want)) { e->cfwd = 0; return fail(e, "CUDA: forward failed"); }
    const cudaError_t se = cudaStreamSynchronize(e->s);
    e->cfwd = 0;
    if (se != cudaSuccess || cudaGetLastError() != cudaSuccess) return fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    if (logits) memcpy(logits, e->hlog, sizeof(float) * (size_t) T * e->c.vocab);
    const size_t nr = (size_t) e->c.n_layer * T * e->c.top_k;
    if (e->routes && (cudaMemcpy(e->hroute, e->droute, sizeof(int32_t) * nr, cudaMemcpyDeviceToHost) != cudaSuccess ||
                      fwrite(&T, sizeof T, 1, e->routes) != 1 || fwrite(e->hroute, sizeof(int32_t), nr, e->routes) != nr))
        return fail(e, "Q4E_ROUTES: write failed");
    prof_collect(e);
    e->pos += T;
    e->lrow = want ? T - 1 : -1;
    return 0;
}

int q4e_forward(Q4Eng* e, const int32_t* ids, int n, float* logits) {
    e->err[0] = 0;
    if (e->pos + n > e->max_pos) return fail(e, "the context is full (%d positions)", e->max_pos);
    for (int i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= e->embed.rows) return fail(e, "token %d out of range", ids[i]);
    memcpy(e->ids + e->pos, ids, sizeof(int32_t) * (size_t) n);
    if (n > MV_MAXT) ple_run(e, n, 1);   // a prompt's PLE rows at once: the OS reads them in parallel (decode: by head)
    if (!logits && n > e->mt && !e->routes) {   // a long prompt: layer-major passes (Q4E_ROUTES traces chunk by chunk)
        for (int i = 0; i < n; i += e->xr) if (forward_pass(e, ids + i, n - i < e->xr ? n - i : e->xr)) return -1;
        return 0;
    }
    const int step = logits ? Q4E_LOGIT_ROWS : e->mt;
    for (int i = 0; i < n; i += step) {
        const int T = n - i < step ? n - i : step;
        if (forward_rows(e, ids + i, T, logits ? logits + (size_t) i * e->c.vocab : NULL)) return -1;
    }
    return 0;
}

// ---- MTP: drafts, and verify forwards that keep each row's state -----------------------------------------------------

// k head rows on din = {pos, bonus, ...}: row j at pos - 1 + j on token din[1 + j] and Hall[-1], then its own streams;
// draft j to din[2 + j]: the arg max, or sampled (a sample of q at dtu)
static int draft_rows(Q4Eng* e, int k, int sampled) {
    const Qwen4Cfg* c = &e->c;
    float* H = e->H;
    for (int j = 0; j < k && !e->err[0]; ++j) {
        head_in(e, e->din + 1 + j, j ? e->Hm : e->Hall - (size_t) c->hc * c->d, 1);
        e->H = e->Hm;
        if (run_layer(e, &e->M, -1, 1, 1, 0, j - 1)) break;
        hc_mix(e, e->m_norm, &e->m_down, &e->m_up, 0, 1);
        mv(e, &e->head, e->x, e->mlog, 1);
        if (sampled) kc_draft_q(e->s, e->mlog, c->vocab, e->dtu, j, e->din + 2 + j, e->dqid + j * KC_DRAFT_Q, e->dqp + j * KC_DRAFT_Q, e->dprob + j);
        else kc_argmax_p(e->s, e->mlog, e->din + 2 + j, e->dprob + j, c->vocab, 1);
        mark(e, ST_HEAD);
    }
    e->H = H;
    return e->err[0] ? -1 : 0;
}

typedef char q4e_draft_q_size[KC_DRAFT_Q == Q4E_DRAFT_Q ? 1 : -1];
int q4e_draft(Q4Eng* e, int k, int32_t* drafts) { return q4e_draft_from(e, -1, k, 0, NULL, drafts); }
int q4e_draft_from(Q4Eng* e, int32_t bonus, int k, float temp, const float* u, int32_t* drafts) {
    const Qwen4Cfg* c = &e->c;
    const int sampled = temp > 0 && u;
    e->err[0] = 0;
    if (!e->mtp || e->lrow < 0 || k < 1 || k > e->ndraft || e->pos + k > e->max_pos || bonus >= c->vocab)
        return fail(e, "q4e_draft: %s", !e->mtp ? "no MTP head" : e->lrow < 0 ? "no logits of the last row" : bonus >= c->vocab ? "bonus out of range" : "k out of range");
    e->hin[0] = e->pos;
    e->hin[1] = bonus;
    cudaMemcpyAsync(e->din, e->hin, sizeof(int32_t) * (bonus < 0 ? 1 : 2), cudaMemcpyHostToDevice, e->s);
    if (bonus < 0) kc_argmax(e->s, e->logits + (size_t) e->lrow * c->vocab, e->din + 1, c->vocab, 1);   // the bonus
    if (sampled) {
        e->htu[0] = temp;
        memcpy(e->htu + 1, u, sizeof(float) * (size_t) k);
        cudaMemcpyAsync(e->dtu, e->htu, sizeof(float) * (size_t) (k + 1), cudaMemcpyHostToDevice, e->s);
    }
    cudaGraphExec_t* ge = &e->dgraph[sampled][k];
    if (!e->prof && !e->no_graph && !*ge) {   // a graph for each k: the positions come from din
        cudaGraph_t gr = NULL;
        if (cudaStreamBeginCapture(e->s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return fail(e, "CUDA graph capture failed");
        e->cap = 1;
        const int rc = draft_rows(e, k, sampled);
        e->cap = 0;
        if (cudaStreamEndCapture(e->s, &gr) != cudaSuccess || rc) { if (gr) cudaGraphDestroy(gr); return fail(e, "CUDA graph capture failed"); }
        const cudaError_t ri = cudaGraphInstantiate(ge, gr, 0);
        cudaGraphDestroy(gr);
        if (ri != cudaSuccess) { *ge = NULL; return fail(e, "CUDA graph: %s", cudaGetErrorString(ri)); }
    }
    if (*ge ? cudaGraphLaunch(*ge, e->s) != cudaSuccess : draft_rows(e, k, sampled)) return fail(e, "CUDA: draft failed");
    cudaMemcpyAsync(e->hin + 1, e->din + 2, sizeof(int32_t) * (size_t) k, cudaMemcpyDeviceToHost, e->s);
    cudaMemcpyAsync(e->hprob, e->dprob, sizeof(float) * (size_t) k, cudaMemcpyDeviceToHost, e->s);
    if (sampled) {
        cudaMemcpyAsync(e->hqid, e->dqid, sizeof(int32_t) * (size_t) k * KC_DRAFT_Q, cudaMemcpyDeviceToHost, e->s);
        cudaMemcpyAsync(e->hqp, e->dqp, sizeof(float) * (size_t) k * KC_DRAFT_Q, cudaMemcpyDeviceToHost, e->s);
    }
    e->dqn = sampled ? KC_DRAFT_Q : 0;
    if (cudaStreamSynchronize(e->s) != cudaSuccess || cudaGetLastError() != cudaSuccess)
        return fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    memcpy(drafts, e->hin + 1, sizeof(int32_t) * (size_t) k);
    prof_collect(e);
    return 0;
}

int q4e_draft_q(const Q4Eng* e, int j, int32_t* id, float* p) {
    if (!e->mtp || j < 0 || j >= e->ndraft) return -1;
    memcpy(id, e->hqid + (size_t) j * KC_DRAFT_Q, sizeof(int32_t) * (size_t) e->dqn);
    memcpy(p, e->hqp + (size_t) j * KC_DRAFT_Q, sizeof(float) * (size_t) e->dqn);
    return e->dqn;
}
int q4e_draft_probs(const Q4Eng* e, int k, float* p) {
    if (!e->mtp || k < 1 || k > e->ndraft) return -1;
    memcpy(p, e->hprob, sizeof(float) * (size_t) k);
    return 0;
}

int q4e_forward_keep(Q4Eng* e, const int32_t* ids, int n, float* logits) {
    e->err[0] = 0;
    e->nkeep = 0;
    if (!e->mtp || n < 1 || n > e->ndraft + 1) return fail(e, "q4e_forward_keep: %s", e->mtp ? "more rows than Q4EOpts.mtp + 1" : "no MTP head");
    e->keep = 1;
    const int rc = q4e_forward(e, ids, n, logits);
    e->keep = 0;
    if (!rc) e->nkeep = n;
    return rc;
}

int q4e_accept(Q4Eng* e, int m) {
    const Qwen4Cfg* c = &e->c;
    const int n = e->nkeep;
    e->err[0] = 0;
    e->nkeep = 0;
    if (m < 1 || m > n) return fail(e, "q4e_accept: %d of %d kept rows", m, n);
    if (m == n) return 0;
    const size_t nc = qwen4_gdn_conv_len(c), ns = qwen4_gdn_state_len(c), nh = qwen4_ple_hist_len(c);
    for (int l = 0; l < c->n_layer; ++l) {   // row m - 1's state: its snapshot
        Layer* L = &e->L[l];
        if (L->kS) {
            cudaMemcpyAsync(L->conv, L->kconv + (size_t) (m - 1) * nc, sizeof(float) * nc, cudaMemcpyDeviceToDevice, e->s);
            cudaMemcpyAsync(L->S, L->kS + (size_t) (m - 1) * ns, sizeof(float) * ns, cudaMemcpyDeviceToDevice, e->s);
        }
        if (L->khist) cudaMemcpyAsync(L->hist, L->khist + (size_t) (m - 1) * nh, sizeof(float) * nh, cudaMemcpyDeviceToDevice, e->s);
    }
    keep_last(e, m - 1);
    if (e->lrow >= 0) e->lrow = m - 1;
    e->pos -= n - m;
    if (cudaStreamSynchronize(e->s) != cudaSuccess || cudaGetLastError() != cudaSuccess)
        return fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
