// engine/mova_cuda.c - CUDA host code of the GPU engine for K2-Horizon MoVA (implements engine_api.h, mova_ext.h); the
// counterpart of engine/mova_gpu.m (Metal), with the same forward, the same sequence logic and the same kernels'
// arithmetic (engine/kernels_moe.cu).  C11 on the CUDA runtime API.
//
// Memory: a discrete GPU may hold less than the model.  Every tensor except the routed MLP experts and the MoVA value
// experts goes to VRAM.  Experts are tiered per expert (an MLP expert's gate, up and down slices move together): as
// many as fit in VRAM (NSLM_VRAM_RESERVE_MB left free) live there, the others in pinned, mapped host memory that the
// kernels read over PCIe.  Kernels address experts through per-slice tables (WSlice), so placement is invisible to
// them.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>

#include <cuda_runtime_api.h>

#include "engine_api.h"
#include "kernels_cuda.h"
#include "lfsr.h"
#include "model_st.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"
#include "mova_ext.h"

#define MAX_ROWS 512             // rows per forward (prompt chunks); decode and T <= 8 use the matvecs
#define MAXP (MAX_ROWS * 8)      // (row, expert) pairs per forward
#define MAX_GRAPHS 64            // instantiated decode graphs kept (decode at one length range needs a handful)
#ifndef MAX_SPLITS
#define MAX_SPLITS 32            // decode attention: at most this many key splits
#endif
#define MAX_LOGIT_ROWS 64        // LM head rows per pass
#define MAX_TILES (MAXP / MMT_BN + 128)

static int cuda_ok(cudaError_t r, const char* what, const char* file, int line) {
    if (r == cudaSuccess) return 1;
    fprintf(stderr, "mova engine: %s failed at %s:%d: %s\n", what, file, line, cudaGetErrorString(r));
    return 0;
}
#define CK(x) cuda_ok((x), #x, __FILE__, __LINE__)

typedef struct {
    int fmt, slices, rows, cols;
    WSlice w0;      // slice 0: dense tensors
    WSlice* d;      // device table [slices] (stacked experts: inside the expert cache's table)
    WSlice* h;      // host copy of the slices' exponent biases (stacked experts)
    const NsTensor* nt;   // stacked experts: the folder's tensor, or NULL (BF16 from the original checkpoint, mt)
    MovaTensor mt;
} MW;

typedef struct {
    MW ln1, ln2, q, k, o, g, v;          // v: dense layers
    MW vr, vb, vx;                        // value router, bias, experts (sparse layers)
    MW mg, mu, md;                        // dense MLP
    MW r, rb, eg, eu, ed, sg, su, sd;     // MoE router, bias, experts, shared expert
    int sparse;
} Layer;

typedef struct {
    int32_t* hist;
    int len, cap;
} Seq;


struct Eng {
    MovaCfg c;
    cudaStream_t st;
    NsModel nm;
    char model_dir[1024];
    MovaCkpt* ck;
    Layer* L;
    MW embed, norm, head;
    uint32_t* stab;     // per-seed 24-bit stream table (SEED4)
    uint32_t* stab32;   // 32-bit stream table (SEED4P4)
    // KV cache (kernels_cuda.h KvView): positions [0, kv_nv) of every layer in VRAM, the rest in mapped host memory; a
    // prompt stages the host rows of the layer it computes in kv_stage
    int kv_fmt;
    int64_t kv_cap, kv_nv;
    uint64_t kv_row, kv_srow;   // bytes per position and layer: values (K or V), scales (K or V; Q8)
    uint8_t *kv_vk, *kv_vv, *kv_hk, *kv_hv, *kv_sk, *kv_sv;   // VRAM, host (device addresses), staging: values
    float *kv_vks, *kv_vvs, *kv_hks, *kv_hvs, *kv_sks, *kv_svs;   // ... scales (Q8)
    int kv_staged;   // the layer whose host rows kv_stage holds for the running prompt pass, or -1
    // scratch (MAX_ROWS rows), device
    float *x, *xn, *q, *k, *v, *gq, *ao, *ga, *ua, *aa, *G, *U, *A, *D, *V, *sh, *logits, *inv, *part, *wts, *vwts, *rsel, *vsel, *rscore;
    int32_t *ids, *inds, *vinds, *am, *perm, *vperm, *ntiles, *vntiles;
    RowInfo* ri;
    MmTile *tiles, *vtiles;
    // pinned host staging
    int32_t *h_ids, *h_inds, *h_perm, *h_am;
    RowInfo* h_ri;
    MmTile* h_tiles;
    float* h_logits;
    // device allocations (freed at close)
    void** allocs;
    int nalloc, capalloc;
    void** hallocs;   // pinned host allocations
    int nhalloc, caphalloc;
    CachePool pm, pv;   // expert caches: MLP experts, value experts
    // prompt prefetch: the next layer's experts come over PCIe on a second stream while a layer computes
    // Views of pm / pv (the same slots, tables and LRU ticks) with their own job lists, one pair per producer: a view's
    // copy reads its job list after the admit that wrote it, so two producers must never share one.  pm_pre / pv_pre:
    // decode's next-layer prediction; pm_pf / pv_pf: a prompt's whole-layer prefetch.
    CachePool pm_pre, pv_pre, pm_pf, pv_pf;
    cudaStream_t cst;
    cudaEvent_t ev_admit, *ev_ready;   // admitted (main stream), a layer's experts in VRAM (copy stream)
    int32_t* all_ids;                  // 0 .. 127
    // decode prefetch: the next sparse layer's routers on this layer's post-attention state predict its experts,
    // which come over PCIe on the copy stream while this layer finishes; ev_pred[l]: layer l's predicted experts in
    // VRAM
    int predict;
    int32_t *pred_inds, *pred_vinds;
    float *pred_w, *pred_sel;
    cudaEvent_t *ev_pred, *ev_predv;   // a layer's predicted MLP / value experts in VRAM
    int pred_pending;   // the layer whose prediction the main stream must join first, or -1
    int seed_f32;       // NSLM_SEED_GEMM_F32: the prefill GEMM's seed weights exact (kc_seed_gemm_f32)
    int in_lm;          // forward_lm's layer-major pass: no prediction (it could replace a slot a pending prefetch copy fills)
    // layer-major prefill: the hidden states, tokens and rows of up to xmax prompt rows
    float* x_all;
    int32_t *ids_all, *h_ids_all;
    RowInfo *ri_all, *h_ri_all;
    int xmax;
    cudaGraphExec_t graphs[(MV_MAXT + 1) * (MAX_SPLITS + 1) * 2];   // small forwards, by (rows, splits, head)
    uint64_t graph_use[(MV_MAXT + 1) * (MAX_SPLITS + 1) * 2], graph_tick;   // last launch of each (LRU)
    int ngraphs;        // instantiated (at most MAX_GRAPHS: each is a whole forward's thousands of nodes)
    int no_graph;
    EngMem mem;
    Seq seq;
    char desc[640];   // eng_describe (the device name alone can be 255 bytes)
    // route capture
    int route_on, route_max;
    int32_t *route_mlp, *route_val;
    float *route_mlp_sel, *route_val_sel;
    int route_rows;
    // timing
    int timing_on;
    double tg_sec[MOVA_TG_N];
    cudaEvent_t ev[16384];
    short ev_group[16384];
    int nev, cur_group;
};

// ---- allocation -----------------------------------------------------------------------------------------------------

static void* dalloc(Eng* e, uint64_t bytes, int64_t* kind) {
    bytes = (bytes + 255) & ~255ull;
    void* p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    if (e->nalloc == e->capalloc) {
        e->capalloc = e->capalloc ? 2 * e->capalloc : 256;
        e->allocs = (void**) realloc(e->allocs, sizeof(void*) * (size_t) e->capalloc);
    }
    e->allocs[e->nalloc++] = p;
    if (kind) *kind += (int64_t) bytes;
    return p;
}
static void* scratch(Eng* e, uint64_t bytes) { return dalloc(e, bytes, &e->mem.scratch); }
static void* halloc(Eng* e, uint64_t bytes) {
    void* p = NULL;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return NULL;
    if (e->nhalloc == e->caphalloc) {
        e->caphalloc = e->caphalloc ? 2 * e->caphalloc : 64;
        e->hallocs = (void**) realloc(e->hallocs, sizeof(void*) * (size_t) e->caphalloc);
    }
    e->hallocs[e->nhalloc++] = p;
    return p;
}
// ---- weights --------------------------------------------------------------------------------------------------------

static uint64_t stream_slice_len(const MW* w, int s) {   // bytes of one slice of stream s (0: unused)
    if ((w->fmt == MF_SEED4 || w->fmt == MF_SEED4P4) && s == 2) return 0;   // exponent biases: in the WSlice
    return ns_stream_len(w->fmt, w->slices, w->rows, w->cols, s) / (uint64_t) w->slices;
}

// A dense tensor (or a router / norm / bias): every stream copied into VRAM.
static int upload_dense(Eng* e, const NsTensor* t, MW* w, char* err, int errlen) {
    memset(w, 0, sizeof *w);
    w->fmt = t->enc; w->slices = t->slices; w->rows = t->rows; w->cols = t->cols;
    for (int s = 0; s < 4; ++s) {
        const NsStream* st = &t->s[s];
        if (!st->len) continue;
        if ((t->enc == MF_SEED4 || t->enc == MF_SEED4P4) && s == 2) { memcpy(&w->w0.eb, st->p, 4); continue; }
        void* d = dalloc(e, st->len, &e->mem.weights);
        if (!d || !CK(cudaMemcpy(d, st->p, st->len, cudaMemcpyHostToDevice))) {
            snprintf(err, (size_t) errlen, "%s: out of GPU memory", t->name);
            return -1;
        }
        w->w0.p[s] = d;
    }
    return 0;
}

// BF16 from the original checkpoint (slices > 1: stacked experts), as dense VRAM tensors.
static int upload_ckpt(Eng* e, const MovaTensor* t, MW* w, char* err, int errlen) {
    if (!e->ck && !(e->ck = mova_ckpt_open(e->model_dir, err, errlen))) return -1;
    memset(w, 0, sizeof *w);
    w->fmt = MF_BF16; w->slices = t->slices; w->rows = t->rows; w->cols = t->cols;
    if (t->kind == MOVA_K_EXPERTS || t->kind == MOVA_K_VEXPERTS) {   // to the expert cache
        w->mt = *t;
        w->h = (WSlice*) calloc((size_t) t->slices, sizeof(WSlice));
        return 0;
    }
    const uint64_t one = (uint64_t) t->rows * t->cols * 2;
    uint8_t* d = (uint8_t*) dalloc(e, one * (uint64_t) t->slices, &e->mem.weights);
    if (!d) { snprintf(err, (size_t) errlen, "%s: out of GPU memory", t->name); return -1; }
    char nm[128];
    for (int s = 0; s < t->slices; ++s) {
        const uint16_t* src = mova_ckpt_bf16(e->ck, mova_slice_name(t, s, nm, sizeof nm), t->rows, t->cols, err, errlen);
        if (!src || !CK(cudaMemcpy(d + one * (uint64_t) s, src, one, cudaMemcpyHostToDevice))) return -1;
    }
    w->w0.p[0] = d;
    return 0;
}

// Stacked experts: shape and source only; their bytes go to the expert cache (build_pools).
static int describe_stacked(const NsTensor* t, MW* w) {
    memset(w, 0, sizeof *w);
    w->fmt = t->enc; w->slices = t->slices; w->rows = t->rows; w->cols = t->cols;
    w->nt = t;
    w->h = (WSlice*) calloc((size_t) t->slices, sizeof(WSlice));
    if (t->enc == MF_SEED4 || t->enc == MF_SEED4P4)
        for (int s = 0; s < t->slices; ++s) memcpy(&w->h[s].eb, t->s[2].p + 4 * (size_t) s, 4);
    return 0;
}

// ---- the expert cache (kernels_cuda.h CachePool) --------------------------------------------------------------------

// The bytes of slice x of stream s of a stacked tensor: from the model folder, or BF16 from the original checkpoint.
static const uint8_t* slice_src(Eng* e, const MW* w, int s, int x, char* err, int errlen) {
    const uint64_t len = stream_slice_len(w, s);
    if (w->nt) return w->nt->s[s].p + len * (uint64_t) x;
    char nm[160];
    return (const uint8_t*) mova_ckpt_bf16(e->ck, mova_slice_name(&w->mt, x, nm, sizeof nm), w->rows, w->cols, err, errlen);
}

typedef struct {
    Eng* e;
    CachePool* p;
    MW* const* ws;     // [sparse layer][ntens]
    uint8_t* host;
    int units, next, fail;
    pthread_mutex_t mu;
} Fill;
static void* fill_worker(void* arg) {   // host arena: every unit, in unit order
    Fill* f = (Fill*) arg;
    char err[256];
    for (;;) {
        pthread_mutex_lock(&f->mu);
        const int u = f->next++;
        pthread_mutex_unlock(&f->mu);
        if (u >= f->units) return NULL;
        const int sl = u / f->p->per_layer, x = u % f->p->per_layer;
        for (int i = 0; i < f->p->ntens; ++i) {
            const MW* w = f->ws[sl * f->p->ntens + i];
            for (int s = 0; s < 4; ++s) {
                if (f->p->off[i][s] == 0xFFFFFFFFu) continue;
                const uint8_t* src = slice_src(f->e, w, s, x, err, sizeof err);
                if (!src) { f->fail = 1; return NULL; }
                memcpy(f->host + (uint64_t) u * f->p->unit_bytes + f->p->off[i][s], src, stream_slice_len(w, s));
            }
        }
    }
}

// One pool over the stacked tensors ws[sparse layer][ntens] (MLP experts: gate, up, down; value experts: one) with
// `slots` VRAM slots.  The first slots are filled round-robin over the layers (expert-major), so every layer starts
// with the same share; the cache adapts from there.
static int build_pool(Eng* e, CachePool* p, MW* const* ws, int ntens, int per_layer, uint64_t vram_bytes, char* err, int errlen) {
    const MovaCfg* c = &e->c;
    const int nsl = c->n_layer - c->first_sparse, units = nsl * per_layer;
    memset(p, 0, sizeof *p);
    p->per_layer = per_layer;
    p->ntens = ntens;
    uint64_t cur = 0;
    for (int i = 0; i < 3; ++i) for (int s = 0; s < 4; ++s) p->off[i][s] = 0xFFFFFFFFu;
    for (int i = 0; i < ntens; ++i)
        for (int s = 0; s < 4; ++s) {
            const uint64_t len = stream_slice_len(ws[i], s);
            if (!len) continue;
            p->off[i][s] = (uint32_t) cur;
            cur += (len + 255) & ~255ull;
        }
    p->unit_bytes = cur;
    p->slots = (int) (vram_bytes / cur);
    if (p->slots > units) p->slots = units;
    if (p->slots < units && p->slots < 3 * per_layer) {   // a prompt keeps two layers' streamed experts while it brings a third
        snprintf(err, (size_t) errlen, "expert cache: room for %d experts, a layer needs %d (free GPU memory or a smaller --ctx)", p->slots, per_layer);
        return -1;
    }
    // the host arena, filled by threads (the folder's pages come from disk or the page cache)
    uint8_t* host = (uint8_t*) halloc(e, cur * (uint64_t) units);
    if (!host) { snprintf(err, (size_t) errlen, "cannot pin %.2f GB of host memory for the experts", cur * (double) units / 1e9); return -1; }
    e->mem.staging += (int64_t) (cur * (uint64_t) units);
    {
        Fill f;
        memset(&f, 0, sizeof f);
        f.e = e; f.p = p; f.ws = ws; f.host = host; f.units = units;
        pthread_mutex_init(&f.mu, NULL);
        pthread_t th[8];
        int nth = 0;
        for (int i = 0; i < 8; ++i) nth += pthread_create(&th[nth], NULL, fill_worker, &f) == 0;
        if (!nth) fill_worker(&f);   // no thread could start: fill on this one
        for (int i = 0; i < nth; ++i) pthread_join(th[i], NULL);
        pthread_mutex_destroy(&f.mu);
        if (f.fail) { snprintf(err, (size_t) errlen, "expert cache: cannot read the experts"); return -1; }
    }
    void* dh = NULL;
    if (!CK(cudaHostGetDevicePointer(&dh, host, 0))) return -1;
    p->host = (const uint8_t*) dh;
    p->vram = (uint8_t*) dalloc(e, cur * (uint64_t) p->slots, &e->mem.weights);
    p->unit_slot = (int32_t*) dalloc(e, 4 * (uint64_t) units, &e->mem.scratch);
    p->slot_unit = (int32_t*) dalloc(e, 4 * (uint64_t) p->slots, &e->mem.scratch);
    p->slot_last = (uint32_t*) dalloc(e, 4 * (uint64_t) p->slots, &e->mem.scratch);
    p->tick = (uint32_t*) dalloc(e, 16, &e->mem.scratch);
    p->stats = (uint32_t*) dalloc(e, 16, &e->mem.scratch);
    p->jobs = (int32_t*) dalloc(e, 8 * (uint64_t) per_layer, &e->mem.scratch);
    p->njobs = (int32_t*) dalloc(e, 16, &e->mem.scratch);
    p->tab = (WSlice*) dalloc(e, sizeof(WSlice) * (uint64_t) units * (uint64_t) ntens, &e->mem.scratch);
    if (!p->vram || !p->unit_slot || !p->slot_unit || !p->slot_last || !p->tick || !p->stats || !p->jobs || !p->njobs || !p->tab) {
        snprintf(err, (size_t) errlen, "expert cache: out of GPU memory");
        return -1;
    }
    // initial residency and the slice tables
    const uint32_t tick0[4] = {CACHE_TICK_BASE, 0, 0, 0};
    int32_t* us = (int32_t*) malloc(4 * (size_t) units);
    int32_t* su = (int32_t*) malloc(4 * (size_t) p->slots);
    WSlice* tab = (WSlice*) calloc((size_t) units * (size_t) ntens, sizeof(WSlice));
    for (int u = 0; u < units; ++u) us[u] = -1;
    int s = 0;
    for (int x = 0; x < per_layer && s < p->slots; ++x)
        for (int sl = 0; sl < nsl && s < p->slots; ++sl, ++s) {
            const int u = sl * per_layer + x;
            us[u] = s;
            su[s] = u;
            if (!CK(cudaMemcpy(p->vram + (uint64_t) s * cur, host + (uint64_t) u * cur, cur, cudaMemcpyHostToDevice))) {
                free(us); free(su); free(tab);
                return -1;
            }
        }
    for (int sl = 0; sl < nsl; ++sl)
        for (int i = 0; i < ntens; ++i)
            for (int x = 0; x < per_layer; ++x) {
                WSlice* t = &tab[((size_t) sl * ntens + i) * per_layer + x];
                const int u = sl * per_layer + x;
                t->eb = ws[sl * ntens + i]->h[x].eb;
                if (us[u] < 0) continue;
                for (int st = 0; st < 4; ++st)
                    if (p->off[i][st] != 0xFFFFFFFFu) t->p[st] = p->vram + (uint64_t) us[u] * cur + p->off[i][st];
            }
    int ok = CK(cudaMemcpy(p->unit_slot, us, 4 * (size_t) units, cudaMemcpyHostToDevice)) &&
             CK(cudaMemcpy(p->slot_unit, su, 4 * (size_t) p->slots, cudaMemcpyHostToDevice)) &&
             CK(cudaMemset(p->slot_last, 0, 4 * (size_t) p->slots)) && CK(cudaMemcpy(p->tick, tick0, 16, cudaMemcpyHostToDevice)) &&
             CK(cudaMemset(p->njobs, 0, 16)) && CK(cudaMemset(p->stats, 0, 16)) &&
             CK(cudaMemcpy(p->tab, tab, sizeof(WSlice) * (size_t) units * ntens, cudaMemcpyHostToDevice));
    free(us); free(su); free(tab);
    if (!ok) return -1;
    for (int sl = 0; sl < nsl; ++sl)
        for (int i = 0; i < ntens; ++i) ws[sl * ntens + i]->d = p->tab + ((size_t) sl * ntens + i) * per_layer;
    return 0;
}

// Whether place_kv will keep part of the KV cache in host memory (its budget, before the prefill buffers exist).
static int kv_spills(Eng* e) {
    const uint64_t all = (e->kv_row + e->kv_srow) * 2 * (uint64_t) e->c.n_layer;
    if (getenv("NSLM_KV_VRAM_MB")) return (int64_t) ((uint64_t) (atof(getenv("NSLM_KV_VRAM_MB")) * 1048576.0) / all) < e->kv_cap;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    const char* rs = getenv("NSLM_VRAM_RESERVE_MB");
    const char* em = getenv("NSLM_EXPERT_MIN_MB");
    const uint64_t reserve = (uint64_t) (rs ? atoll(rs) : 512) << 20, emin = (uint64_t) (em ? atoll(em) : 6144) << 20;
    return all * (uint64_t) e->kv_cap > (fr > reserve + emin ? fr - reserve - emin : 0);
}

// The KV cache: as many positions of every layer in VRAM as the budget allows (every token reads all of its context,
// so a GB of KV in VRAM saves far more than a GB of expert cache beyond its minimum, NSLM_EXPERT_MIN_MB), the rest in
// pinned host memory, plus one layer's staging for prompts.  NSLM_KV_VRAM_MB sets the VRAM share explicitly.
static int place_kv(Eng* e, char* err, int errlen) {
    const MovaCfg* c = &e->c;
    const uint64_t per = (e->kv_row + e->kv_srow) * 2;   // K and V, one layer, one position
    const uint64_t all = per * (uint64_t) c->n_layer;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    const char* rs = getenv("NSLM_VRAM_RESERVE_MB");
    const char* em = getenv("NSLM_EXPERT_MIN_MB");
    const uint64_t reserve = (uint64_t) (rs ? atoll(rs) : 512) << 20, emin = (uint64_t) (em ? atoll(em) : 6144) << 20;
    const uint64_t avail = fr > reserve + emin ? fr - reserve - emin : 0;
    int64_t nv = e->kv_cap;
    const char* kvm = getenv("NSLM_KV_VRAM_MB");
    if (kvm) nv = (int64_t) (((uint64_t) (atof(kvm) * 1048576.0)) / all);
    else if (all * (uint64_t) e->kv_cap > avail) {
        // VRAM rows + one layer's staging of the rest must fit: nv * all + (cap - nv) * per <= avail
        const uint64_t st = per * (uint64_t) e->kv_cap;
        nv = avail > st ? (int64_t) ((avail - st) / (all - per)) : 0;
    }
    if (nv > e->kv_cap) nv = e->kv_cap;
    if (nv < 0) nv = 0;
    e->kv_nv = nv;
    e->kv_staged = -1;
    const uint64_t nh = (uint64_t) (e->kv_cap - nv), L = (uint64_t) c->n_layer;
    if (nv) {
        e->kv_vk = (uint8_t*) dalloc(e, (uint64_t) nv * e->kv_row * L, &e->mem.kv);
        e->kv_vv = (uint8_t*) dalloc(e, (uint64_t) nv * e->kv_row * L, &e->mem.kv);
        if (e->kv_srow) {
            e->kv_vks = (float*) dalloc(e, (uint64_t) nv * e->kv_srow * L, &e->mem.kv);
            e->kv_vvs = (float*) dalloc(e, (uint64_t) nv * e->kv_srow * L, &e->mem.kv);
        }
        if (!e->kv_vk || !e->kv_vv || (e->kv_srow && (!e->kv_vks || !e->kv_vvs))) {
            snprintf(err, (size_t) errlen, "KV cache (%lld positions in VRAM): out of GPU memory", (long long) nv);
            return -1;
        }
    }
    if (nh) {
        void* dp = NULL;
        uint8_t* h[4] = {NULL, NULL, NULL, NULL};
        const uint64_t sz[4] = {nh * e->kv_row * L, nh * e->kv_row * L, nh * e->kv_srow * L, nh * e->kv_srow * L};
        for (int i = 0; i < 4; ++i) {
            if (!sz[i]) continue;
            h[i] = (uint8_t*) halloc(e, sz[i]);
            if (!h[i] || !CK(cudaHostGetDevicePointer(&dp, h[i], 0))) {
                snprintf(err, (size_t) errlen, "KV cache: cannot pin %.2f GB of host memory", sz[i] / 1e9);
                return -1;
            }
            e->mem.staging += (int64_t) sz[i];
            h[i] = (uint8_t*) dp;
        }
        e->kv_hk = h[0]; e->kv_hv = h[1]; e->kv_hks = (float*) h[2]; e->kv_hvs = (float*) h[3];
        e->kv_sk = (uint8_t*) dalloc(e, nh * e->kv_row, &e->mem.kv);
        e->kv_sv = (uint8_t*) dalloc(e, nh * e->kv_row, &e->mem.kv);
        if (e->kv_srow) {
            e->kv_sks = (float*) dalloc(e, nh * e->kv_srow, &e->mem.kv);
            e->kv_svs = (float*) dalloc(e, nh * e->kv_srow, &e->mem.kv);
        }
        if (!e->kv_sk || !e->kv_sv || (e->kv_srow && (!e->kv_sks || !e->kv_svs))) {
            snprintf(err, (size_t) errlen, "KV staging: out of GPU memory");
            return -1;
        }
    }
    return 0;
}

// The two pools share the VRAM left after everything else (NSLM_VRAM_RESERVE_MB stays free; NSLM_EXPERT_VRAM_MB caps
// it), split by the pools' byte totals.
static int build_pools(Eng* e, char* err, int errlen) {
    const MovaCfg* c = &e->c;
    const int nsl = c->n_layer - c->first_sparse;
    MW** wm = (MW**) malloc(sizeof(MW*) * (size_t) nsl * 3);
    MW** wv = (MW**) malloc(sizeof(MW*) * (size_t) nsl);
    for (int sl = 0; sl < nsl; ++sl) {
        Layer* L = &e->L[c->first_sparse + sl];
        wm[sl * 3] = &L->eg; wm[sl * 3 + 1] = &L->eu; wm[sl * 3 + 2] = &L->ed;
        wv[sl] = &L->vx;
    }
    uint64_t bm = 0, bv = 0;
    for (int s = 0; s < 4; ++s) {
        for (int i = 0; i < 3; ++i) bm += (stream_slice_len(wm[i], s) + 255) & ~255ull;
        bv += (stream_slice_len(wv[0], s) + 255) & ~255ull;
    }
    bm *= (uint64_t) nsl * (uint64_t) c->n_exp;
    bv *= (uint64_t) nsl * (uint64_t) c->n_vexp;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    const char* rs = getenv("NSLM_VRAM_RESERVE_MB");
    const uint64_t reserve = (uint64_t) (rs ? atoll(rs) : 512) << 20;
    uint64_t budget = fr > reserve ? fr - reserve : 0;
    const char* cap = getenv("NSLM_EXPERT_VRAM_MB");
    if (cap && (uint64_t) (atof(cap) * 1048576.0) < budget) budget = (uint64_t) (atof(cap) * 1048576.0);   // MB, fractions allowed
    if (budget > bm + bv) budget = bm + bv;
    const uint64_t vm = (uint64_t) ((double) budget * (double) bm / (double) (bm + bv));
    const int rc = build_pool(e, &e->pm, wm, 3, c->n_exp, vm, err, errlen) || build_pool(e, &e->pv, wv, 1, c->n_vexp, budget - vm, err, errlen);
    free(wm);
    free(wv);
    return rc ? -1 : 0;
}

static int load_tensor(Eng* e, const MovaTensor* all, int n, const char* name, MW* w, char* err, int errlen) {
    const MovaTensor* t = mova_find(all, n, name);
    if (!t) { snprintf(err, (size_t) errlen, "no logical tensor %s", name); return -1; }
    const NsTensor* nt = ns_find(&e->nm, name);
    if (!nt) return upload_ckpt(e, t, w, err, errlen);
    if (nt->slices != t->slices || nt->rows != t->rows || nt->cols != t->cols) {
        snprintf(err, (size_t) errlen, "%s: shape %d x %d x %d, expected %d x %d x %d", name, nt->slices, nt->rows, nt->cols,
                 t->slices, t->rows, t->cols);
        return -1;
    }
    const int f = nt->enc;
    const int ok = t->kind == MOVA_K_EXPERTS ? 1
                   : (t->kind == MOVA_K_ROUTER || t->kind == MOVA_K_NORM || t->kind == MOVA_K_ROUTER_BIAS) ? f == MF_BF16
                   : t->kind == MOVA_K_EMBED ? (f == MF_BF16 || f == MF_Q8)
                   : f != MF_SEED4 || t->kind == MOVA_K_VEXPERTS || t->kind == MOVA_K_LINEAR || t->kind == MOVA_K_HEAD;
    if (!ok) { snprintf(err, (size_t) errlen, "%s: encoding %d not supported for this tensor", name, f); return -1; }
    if (t->kind == MOVA_K_EXPERTS || t->kind == MOVA_K_VEXPERTS) return describe_stacked(nt, w);
    return upload_dense(e, nt, w, err, errlen);
}

// ---- open / close ---------------------------------------------------------------------------------------------------

Eng* eng_open(const EngOpts* o, char* err, int errlen) {
    Eng* e = (Eng*) calloc(1, sizeof *e);
    if (mova_cfg_load(&e->c, o->model_dir, err, errlen)) { free(e); return NULL; }
    const MovaCfg* c = &e->c;
    if (c->n_head != ATTF_G * c->n_kv) {
        snprintf(err, (size_t) errlen, "attention: %d query heads per KV head, the prefill kernel is built for %d", c->n_head / c->n_kv, ATTF_G);
        free(e);
        return NULL;
    }
    // the kernels' and buffers' limits: experts per layer (shared-memory tables of 128), selections (MAXP, h_inds: 8)
    if (c->n_exp > 128 || c->n_vexp > 128 || c->top_k < 1 || c->top_k > 8 || c->top_kv < 1 || c->top_kv > 8) {
        snprintf(err, (size_t) errlen, "experts: %d / %d per layer, top %d / %d; the CUDA engine supports up to 128 and 8",
                 c->n_exp, c->n_vexp, c->top_k, c->top_kv);
        free(e);
        return NULL;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); free(e); return NULL; }
    if (!CK(cudaSetDevice(0)) || !CK(cudaStreamCreateWithFlags(&e->st, cudaStreamNonBlocking))) { snprintf(err, (size_t) errlen, "CUDA init failed"); free(e); return NULL; }
    if (ns_open(&e->nm, o->model_dir, err, errlen)) { cudaStreamDestroy(e->st); free(e); return NULL; }
    snprintf(e->model_dir, sizeof e->model_dir, "%s", o->model_dir);
    MovaTensor* all = NULL;
    const int na = mova_tensors(c, &all);
    e->L = (Layer*) calloc((size_t) c->n_layer, sizeof(Layer));
    char nm[160];
    int rc = 0;
#define LOAD(dst, ...) do { snprintf(nm, sizeof nm, __VA_ARGS__); if (!rc && load_tensor(e, all, na, nm, (dst), err, errlen)) rc = -1; } while (0)
    LOAD(&e->embed, "model.embed_tokens.weight");
    LOAD(&e->norm, "model.norm.weight");
    LOAD(&e->head, "lm_head.weight");
    int seeds = 0, seeds4 = 0;
    for (int l = 0; l < c->n_layer && !rc; ++l) {
        Layer* L = &e->L[l];
        L->sparse = l >= c->first_sparse;
        LOAD(&L->ln1, "model.layers.%d.input_layernorm.weight", l);
        LOAD(&L->ln2, "model.layers.%d.post_attention_layernorm.weight", l);
        LOAD(&L->q, "model.layers.%d.self_attn.q_proj.weight", l);
        LOAD(&L->k, "model.layers.%d.self_attn.k_proj.weight", l);
        LOAD(&L->o, "model.layers.%d.self_attn.o_proj.weight", l);
        LOAD(&L->g, "model.layers.%d.self_attn.gate_proj.weight", l);
        if (!L->sparse) {
            LOAD(&L->v, "model.layers.%d.self_attn.v_proj.weight", l);
            LOAD(&L->mg, "model.layers.%d.mlp.gate_proj.weight", l);
            LOAD(&L->mu, "model.layers.%d.mlp.up_proj.weight", l);
            LOAD(&L->md, "model.layers.%d.mlp.down_proj.weight", l);
            continue;
        }
        LOAD(&L->vr, "model.layers.%d.self_attn.v_router.weight", l);
        LOAD(&L->vb, "model.layers.%d.self_attn.v_router.bias", l);
        LOAD(&L->vx, "model.layers.%d.self_attn.v_experts.weight", l);
        LOAD(&L->r, "model.layers.%d.mlp.gate.weight", l);
        LOAD(&L->rb, "model.layers.%d.mlp.gate.bias", l);
        LOAD(&L->eg, "model.layers.%d.mlp.experts.gate_proj.weight", l);
        LOAD(&L->eu, "model.layers.%d.mlp.experts.up_proj.weight", l);
        LOAD(&L->ed, "model.layers.%d.mlp.experts.down_proj.weight", l);
        LOAD(&L->sg, "model.layers.%d.mlp.shared_experts.gate_proj.weight", l);
        LOAD(&L->su, "model.layers.%d.mlp.shared_experts.up_proj.weight", l);
        LOAD(&L->sd, "model.layers.%d.mlp.shared_experts.down_proj.weight", l);
        seeds += L->eg.fmt == MF_SEED4 || L->ed.fmt == MF_SEED4;
        seeds4 += L->eg.fmt == MF_SEED4P4 || L->eu.fmt == MF_SEED4P4 || L->ed.fmt == MF_SEED4P4;
    }
#undef LOAD
    free(all);
    if (rc) { eng_close(e); return NULL; }
    // stream tables
    {
        uint32_t* g = (uint32_t*) malloc(65536 * 4);
        e->stab = (uint32_t*) dalloc(e, 65536 * 4, &e->mem.lut);
        for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream24((uint16_t) s);
        int ok = e->stab && CK(cudaMemcpy(e->stab, g, 65536 * 4, cudaMemcpyHostToDevice));
        if (ok && seeds4) {
            e->stab32 = (uint32_t*) dalloc(e, 65536 * 4, &e->mem.lut);
            for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream32((uint16_t) s);
            ok = e->stab32 && CK(cudaMemcpy(e->stab32, g, 65536 * 4, cudaMemcpyHostToDevice));
        } else e->stab32 = e->stab;
        free(g);
        (void) seeds;
        if (!ok) { snprintf(err, (size_t) errlen, "seed tables: out of GPU memory"); eng_close(e); return NULL; }
    }
    // KV caches
    const int kvd = c->n_kv * c->head_dim;
    e->kv_cap = o->kv_tokens > 0 ? o->kv_tokens : 32768;
    e->kv_fmt = o->kv_format == ENG_KV_Q8 ? KV_Q8 : KV_BF16;
    e->kv_row = (uint64_t) kvd * (e->kv_fmt == KV_Q8 ? 1 : 2);
    e->kv_srow = e->kv_fmt == KV_Q8 ? 4 * (uint64_t) c->n_kv : 0;
    // scratch
    const int T = MAX_ROWS, d = c->d, qd = c->n_head * c->head_dim, ffmax = c->ff_dense > c->ff_exp ? c->ff_dense : c->ff_exp;
    e->x = (float*) scratch(e, (uint64_t) T * d * 4);
    e->xn = (float*) scratch(e, (uint64_t) T * d * 4);
    e->q = (float*) scratch(e, (uint64_t) T * qd * 4);
    e->gq = (float*) scratch(e, (uint64_t) T * qd * 4);
    e->ao = (float*) scratch(e, (uint64_t) T * qd * 4);
    e->k = (float*) scratch(e, (uint64_t) T * kvd * 4);
    e->v = (float*) scratch(e, (uint64_t) T * kvd * 4);
    e->ga = (float*) scratch(e, (uint64_t) T * ffmax * 4);
    e->ua = (float*) scratch(e, (uint64_t) T * ffmax * 4);
    e->aa = (float*) scratch(e, (uint64_t) T * ffmax * 4);
    e->G = (float*) scratch(e, (uint64_t) MAXP * c->ff_exp * 4);
    e->U = (float*) scratch(e, (uint64_t) MAXP * c->ff_exp * 4);
    e->A = (float*) scratch(e, (uint64_t) MAXP * c->ff_exp * 4);
    e->D = (float*) scratch(e, (uint64_t) MAXP * d * 4);
    e->V = (float*) scratch(e, (uint64_t) MAXP * kvd * 4);
    e->sh = (float*) scratch(e, (uint64_t) T * d * 4);
    e->logits = (float*) scratch(e, (uint64_t) MAX_LOGIT_ROWS * c->vocab * 4);
    e->perm = (int32_t*) scratch(e, (uint64_t) MAXP * 4);
    e->tiles = (MmTile*) scratch(e, (uint64_t) MAX_TILES * sizeof(MmTile));
    e->vperm = (int32_t*) scratch(e, (uint64_t) MAXP * 4);
    e->vtiles = (MmTile*) scratch(e, (uint64_t) MAX_TILES * sizeof(MmTile));
    e->ntiles = (int32_t*) scratch(e, 260 * 4);
    e->vntiles = (int32_t*) scratch(e, 260 * 4);
    e->ids = (int32_t*) scratch(e, (uint64_t) T * 4);
    e->ri = (RowInfo*) scratch(e, (uint64_t) T * sizeof(RowInfo));
    e->inv = (float*) scratch(e, 64 * 4);
    e->part = (float*) scratch(e, (uint64_t) 1024 * c->n_head * (ATT_HD + 2) * 4);
    e->inds = (int32_t*) scratch(e, (uint64_t) c->n_layer * T * c->top_k * 4);
    e->wts = (float*) scratch(e, (uint64_t) c->n_layer * T * c->top_k * 4);
    e->vinds = (int32_t*) scratch(e, (uint64_t) c->n_layer * T * c->top_kv * 4);
    e->vwts = (float*) scratch(e, (uint64_t) c->n_layer * T * c->top_kv * 4);
    e->rsel = (float*) scratch(e, (uint64_t) c->n_layer * T * c->n_exp * 4);
    e->vsel = (float*) scratch(e, (uint64_t) c->n_layer * T * c->n_vexp * 4);
    e->rscore = (float*) scratch(e, (uint64_t) T * 128 * 4);
    e->am = (int32_t*) scratch(e, (uint64_t) T * 4);
    if (!e->x || !e->xn || !e->q || !e->gq || !e->ao || !e->k || !e->v || !e->ga || !e->ua || !e->aa || !e->G || !e->U || !e->A ||
        !e->D || !e->V || !e->sh || !e->logits || !e->perm || !e->tiles || !e->vperm || !e->vtiles || !e->ids || !e->ri ||
        !e->inv || !e->part || !e->inds || !e->wts || !e->vinds || !e->vwts || !e->rsel || !e->vsel || !e->rscore || !e->am) {
        snprintf(err, (size_t) errlen, "scratch: out of GPU memory");
        eng_close(e);
        return NULL;
    }
    float inv[64];
    for (int p = 0; p < 64; ++p) inv[p] = (float) pow((double) c->rope_theta, -2.0 * p / (double) c->head_dim);
    cudaMemcpy(e->inv, inv, sizeof inv, cudaMemcpyHostToDevice);
    e->h_ids = (int32_t*) halloc(e, (uint64_t) T * 4);
    e->h_ri = (RowInfo*) halloc(e, (uint64_t) T * sizeof(RowInfo));
    e->h_inds = (int32_t*) halloc(e, (uint64_t) T * 8 * 4);
    e->h_perm = (int32_t*) halloc(e, (uint64_t) MAXP * 4);
    e->h_tiles = (MmTile*) halloc(e, (uint64_t) MAX_TILES * sizeof(MmTile));
    e->h_am = (int32_t*) halloc(e, (uint64_t) T * 4);
    e->h_logits = (float*) halloc(e, (uint64_t) MAX_LOGIT_ROWS * c->vocab * 4);
    // Rows per layer-major range.  When KV spills to host memory, every range stages each layer's earlier host rows
    // into VRAM (PCIe traffic ~ prompt^2 / range): a larger range when the KV will not fit (NSLM_PREFILL_RANGE sets it).
    {
        const char* pr = getenv("NSLM_PREFILL_RANGE");
        int64_t xm = pr ? atoll(pr) : kv_spills(e) ? 32768 : 8192;
        if (xm < MAX_ROWS) xm = MAX_ROWS;
        e->xmax = (int) (e->kv_cap < xm ? e->kv_cap : xm);
    }
    e->x_all = (float*) scratch(e, (uint64_t) e->xmax * d * 4);
    e->ids_all = (int32_t*) scratch(e, (uint64_t) e->xmax * 4);
    e->ri_all = (RowInfo*) scratch(e, (uint64_t) e->xmax * sizeof(RowInfo));
    e->h_ids_all = (int32_t*) halloc(e, (uint64_t) e->xmax * 4);
    e->h_ri_all = (RowInfo*) halloc(e, (uint64_t) e->xmax * sizeof(RowInfo));
    if (!e->x_all || !e->ids_all || !e->ri_all || !e->h_ids_all || !e->h_ri_all) {
        snprintf(err, (size_t) errlen, "prefill buffers: out of memory");
        eng_close(e);
        return NULL;
    }
    if (!e->h_ids || !e->h_ri || !e->h_inds || !e->h_perm || !e->h_tiles || !e->h_am || !e->h_logits) {
        snprintf(err, (size_t) errlen, "pinned host staging: out of memory");
        eng_close(e);
        return NULL;
    }
    if (place_kv(e, err, errlen)) { eng_close(e); return NULL; }
    // experts last: they take the VRAM that is left
    if (build_pools(e, err, errlen)) { eng_close(e); return NULL; }
    {
        int32_t ids128[128];
        int lo_prio = 0, hi_prio = 0;   // the copy stream first: its blocks take SMs ahead of the prompt's GEMMs
        for (int i = 0; i < 128; ++i) ids128[i] = i;
        e->all_ids = (int32_t*) scratch(e, sizeof ids128);
        e->pm_pre = e->pm;
        e->pv_pre = e->pv;
        e->pm_pre.jobs = (int32_t*) scratch(e, 8 * (uint64_t) c->n_exp);
        e->pv_pre.jobs = (int32_t*) scratch(e, 8 * (uint64_t) c->n_vexp);
        e->pm_pre.njobs = (int32_t*) scratch(e, 16);
        e->pv_pre.njobs = (int32_t*) scratch(e, 16);
        e->pm_pre.stats = (uint32_t*) scratch(e, 16);
        e->pv_pre.stats = (uint32_t*) scratch(e, 16);
        if (e->pm_pre.stats) cudaMemset(e->pm_pre.stats, 0, 16);
        if (e->pv_pre.stats) cudaMemset(e->pv_pre.stats, 0, 16);
        e->pm_pf = e->pm_pre;   // the prefetch's own job lists; statistics shared with the prediction's
        e->pv_pf = e->pv_pre;
        e->pm_pf.jobs = (int32_t*) scratch(e, 8 * (uint64_t) c->n_exp);
        e->pv_pf.jobs = (int32_t*) scratch(e, 8 * (uint64_t) c->n_vexp);
        e->pm_pf.njobs = (int32_t*) scratch(e, 16);
        e->pv_pf.njobs = (int32_t*) scratch(e, 16);
        e->ev_ready = (cudaEvent_t*) calloc((size_t) c->n_layer, sizeof(cudaEvent_t));
        e->ev_pred = (cudaEvent_t*) calloc((size_t) c->n_layer, sizeof(cudaEvent_t));
        e->ev_predv = (cudaEvent_t*) calloc((size_t) c->n_layer, sizeof(cudaEvent_t));
        e->predict = getenv("NSLM_NO_PREDICT") == NULL;
        e->pred_pending = -1;
        e->pred_inds = (int32_t*) scratch(e, (uint64_t) MV_MAXT * 16 * 4);
        e->pred_vinds = (int32_t*) scratch(e, (uint64_t) MV_MAXT * 16 * 4);
        e->pred_w = (float*) scratch(e, (uint64_t) MV_MAXT * 16 * 4);
        e->pred_sel = (float*) scratch(e, (uint64_t) MV_MAXT * 128 * 4);
        int ok = e->all_ids && e->pm_pre.jobs && e->pv_pre.jobs && e->pm_pre.njobs && e->pv_pre.njobs && e->pm_pf.jobs &&
                 e->pv_pf.jobs && e->pm_pf.njobs && e->pv_pf.njobs &&
                 CK(cudaMemcpy(e->all_ids, ids128, sizeof ids128, cudaMemcpyHostToDevice)) &&
                 CK(cudaDeviceGetStreamPriorityRange(&lo_prio, &hi_prio)) && CK(cudaStreamCreateWithPriority(&e->cst, cudaStreamNonBlocking, hi_prio)) &&
                 CK(cudaEventCreateWithFlags(&e->ev_admit, cudaEventDisableTiming));
        for (int l = 0; ok && l < c->n_layer; ++l)
            ok = CK(cudaEventCreateWithFlags(&e->ev_ready[l], cudaEventDisableTiming)) && CK(cudaEventCreateWithFlags(&e->ev_pred[l], cudaEventDisableTiming)) &&
                 CK(cudaEventCreateWithFlags(&e->ev_predv[l], cudaEventDisableTiming));
        ok = ok && e->pred_inds && e->pred_vinds && e->pred_w && e->pred_sel;
        if (!ok) { snprintf(err, (size_t) errlen, "prefetch: CUDA setup failed"); eng_close(e); return NULL; }
    }
    e->no_graph = getenv("NSLM_NO_GRAPH") != NULL || getenv("MOVA_DUMP") != NULL;
    e->seed_f32 = getenv("NSLM_SEED_GEMM_F32") != NULL;   // a rounding-point experiment; BF16 seed weights by default
    e->seq.cap = 1024;
    e->seq.hist = (int32_t*) malloc(sizeof(int32_t) * (size_t) e->seq.cap);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const char* fm[5] = {"bf16", "seed4", "q8", "q4", "seed4p4"};
    snprintf(e->desc, sizeof e->desc,
             "mova engine (CUDA, %s): experts %s/%s/%s, attention %s, value experts %s, embed %s, head %s%s; "
             "expert cache %d + %d of %d + %d experts in VRAM",
             prop.name, fm[e->L[c->first_sparse].eg.fmt], fm[e->L[c->first_sparse].eu.fmt], fm[e->L[c->first_sparse].ed.fmt],
             fm[e->L[0].q.fmt], fm[e->L[c->first_sparse].vx.fmt], fm[e->embed.fmt], fm[e->head.fmt],
             e->kv_fmt == KV_Q8 ? ", KV q8" : "", e->pm.slots, e->pv.slots, (c->n_layer - c->first_sparse) * c->n_exp,
             (c->n_layer - c->first_sparse) * c->n_vexp);
    return e;
}

void eng_close(Eng* e) {
    if (!e) return;
    if (e->st) cudaStreamSynchronize(e->st);
    if (getenv("NSLM_CACHE_STATS") && e->pm.stats && e->pm_pre.stats) {
        uint32_t a[2], b[2], pa[2], pb[2];
        cudaMemcpy(a, e->pm.stats, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(b, e->pv.stats, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(pa, e->pm_pre.stats, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(pb, e->pv_pre.stats, 8, cudaMemcpyDeviceToHost);
        fprintf(stderr, "expert cache: MLP %u hits %u misses (%.1f%%), values %u hits %u misses (%.1f%%); "
                        "prefetches loaded MLP %u, values %u (already resident: %u, %u)\n",
                a[0], a[1], 100.0 * a[0] / (a[0] + a[1] + 1e-9), b[0], b[1], 100.0 * b[0] / (b[0] + b[1] + 1e-9), pa[1], pb[1], pa[0], pb[0]);
    }
    for (int i = 0; i < e->nalloc; ++i) cudaFree(e->allocs[i]);
    for (int i = 0; i < e->nhalloc; ++i) cudaFreeHost(e->hallocs[i]);
    free(e->allocs);
    free(e->hallocs);
    if (e->L) {
        for (int l = 0; l < e->c.n_layer; ++l) { free(e->L[l].vx.h); free(e->L[l].eg.h); free(e->L[l].eu.h); free(e->L[l].ed.h); }
        free(e->L);
    }
    for (int i = 0; i < 16384; ++i) if (e->ev[i]) cudaEventDestroy(e->ev[i]);
    for (size_t i = 0; i < sizeof e->graphs / sizeof e->graphs[0]; ++i) if (e->graphs[i]) cudaGraphExecDestroy(e->graphs[i]);
    if (e->ev_ready) { for (int l = 0; l < e->c.n_layer; ++l) if (e->ev_ready[l]) cudaEventDestroy(e->ev_ready[l]); free(e->ev_ready); }
    if (e->ev_pred) { for (int l = 0; l < e->c.n_layer; ++l) if (e->ev_pred[l]) cudaEventDestroy(e->ev_pred[l]); free(e->ev_pred); }
    if (e->ev_predv) { for (int l = 0; l < e->c.n_layer; ++l) if (e->ev_predv[l]) cudaEventDestroy(e->ev_predv[l]); free(e->ev_predv); }
    if (e->ev_admit) cudaEventDestroy(e->ev_admit);
    if (e->cst) { cudaStreamSynchronize(e->cst); cudaStreamDestroy(e->cst); }
    if (e->st) cudaStreamDestroy(e->st);
    ns_close(&e->nm);
    if (e->ck) mova_ckpt_close(e->ck);
    free(e->seq.hist);
    free(e->route_mlp); free(e->route_val); free(e->route_mlp_sel); free(e->route_val_sel);
    free(e);
}

const char* eng_describe(Eng* e) { return e->desc; }
int eng_vocab(Eng* e) { return e->c.vocab; }
void eng_mem(Eng* e, EngMem* m) {
    *m = e->mem;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    m->gpu_allocated = (int64_t) (tot - fr);
}

// ---- timing: an event at every kernel-group change; a group is charged the time to the next event ------------------

static void tgroup(Eng* e, int group) {
    if (!e->timing_on || group == e->cur_group || e->nev >= 16384) return;
    if (!e->ev[e->nev]) cudaEventCreate(&e->ev[e->nev]);
    cudaEventRecord(e->ev[e->nev], e->st);
    e->ev_group[e->nev++] = (short) group;
    e->cur_group = group;
}
static int sync_stream(Eng* e) {
    if (!CK(cudaStreamSynchronize(e->st))) return -1;
    if (e->timing_on && e->nev > 0) {
        tgroup(e, -2);   // closing event
        cudaEventSynchronize(e->ev[e->nev - 1]);
        for (int i = 0; i + 1 < e->nev; ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, e->ev[i], e->ev[i + 1]);
            const int g = e->ev_group[i] < 0 ? MOVA_TG_EMBED_NORM : e->ev_group[i];
            e->tg_sec[g] += ms / 1e3;
        }
        // keep the events for reuse; start over
        e->nev = 0;
        e->cur_group = -1;
    }
    return 0;
}

// ---- forward --------------------------------------------------------------------------------------------------------

static const uint32_t* stab_for(Eng* e, const MW* w) { return w->fmt == MF_SEED4P4 ? e->stab32 : e->stab; }

// Dense projection: matvec for T <= MV_MAXT rows, GEMM above.
static void enc_dense(Eng* e, const MW* W, const float* X, int xs, float* Y, int ys, int T, int add) {
    if (T <= MV_MAXT) kc_mv(e->st, W->fmt, W->w0, W->cols, W->rows, X, xs, Y, ys, T, add, stab_for(e, W));
    else kc_mm(e->st, W->fmt, W->w0, W->cols, W->rows, X, xs, Y, ys, T, add, stab_for(e, W));
}
static void enc_router(Eng* e, const MW* W, const MW* bias, const float* X, int n, int k, int32_t* inds, float* wts, float* sel, int T) {
    RouterArgs a = {e->c.d, n, k, e->c.d / e->c.router_parts, e->c.route_scale};
    kc_router(e->st, a, (const uint16_t*) W->w0.p[0], (const uint16_t*) bias->w0.p[0], X, e->rscore, sel, inds, wts, T);
}
// Decode attention: each row splits its keys in up to MAX_SPLITS parts of >= 128 (rows x splits <= 1024).
static int attn_splits(int T, int max_ctx) {
    int ns = (max_ctx + 127) / 128;
    if (ns > MAX_SPLITS) ns = MAX_SPLITS;
    while (ns > 1 && T * ns > 1024) --ns;
    return ns < 1 ? 1 : ns;
}

// Layer l's KV cache: VRAM rows, then the host rows, or their staged copy while a prompt pass computes the layer.
static KvView kv_view(Eng* e, int l) {
    KvView v;
    memset(&v, 0, sizeof v);
    v.fmt = e->kv_fmt;
    v.nv = (int32_t) e->kv_nv;
    const uint64_t nv = (uint64_t) e->kv_nv, nh = (uint64_t) (e->kv_cap - e->kv_nv);
    v.a.k = e->kv_vk + nv * e->kv_row * (uint64_t) l;
    v.a.v = e->kv_vv + nv * e->kv_row * (uint64_t) l;
    if (e->kv_fmt == KV_Q8) {
        v.a.ks = (float*) ((uint8_t*) e->kv_vks + nv * e->kv_srow * (uint64_t) l);
        v.a.vs = (float*) ((uint8_t*) e->kv_vvs + nv * e->kv_srow * (uint64_t) l);
    }
    if (!nh) return v;
    if (e->kv_staged == l) {
        v.b.k = e->kv_sk; v.b.v = e->kv_sv; v.b.ks = e->kv_sks; v.b.vs = e->kv_svs;
        return v;
    }
    v.b.k = e->kv_hk + nh * e->kv_row * (uint64_t) l;
    v.b.v = e->kv_hv + nh * e->kv_row * (uint64_t) l;
    if (e->kv_fmt == KV_Q8) {
        v.b.ks = (float*) ((uint8_t*) e->kv_hks + nh * e->kv_srow * (uint64_t) l);
        v.b.vs = (float*) ((uint8_t*) e->kv_hvs + nh * e->kv_srow * (uint64_t) l);
    }
    return v;
}
// Copy host rows [r0, r1) (row 0 = position kv_nv) of layer l into the staging buffer (in) or back (out).
static int kv_stage_copy(Eng* e, int l, int64_t r0, int64_t r1, int in) {
    if (r1 <= r0) return 0;
    const uint64_t nh = (uint64_t) (e->kv_cap - e->kv_nv);
    const enum cudaMemcpyKind kd = cudaMemcpyDeviceToDevice;   // the host rows are mapped: device addresses
    uint8_t* hs[4] = {e->kv_hk + nh * e->kv_row * (uint64_t) l, e->kv_hv + nh * e->kv_row * (uint64_t) l,
                      e->kv_srow ? (uint8_t*) e->kv_hks + nh * e->kv_srow * (uint64_t) l : NULL,
                      e->kv_srow ? (uint8_t*) e->kv_hvs + nh * e->kv_srow * (uint64_t) l : NULL};
    uint8_t* ss[4] = {e->kv_sk, e->kv_sv, (uint8_t*) e->kv_sks, (uint8_t*) e->kv_svs};
    for (int i = 0; i < 4; ++i) {
        if (!hs[i]) continue;
        const uint64_t rb = i < 2 ? e->kv_row : e->kv_srow, off = (uint64_t) r0 * rb, len = (uint64_t) (r1 - r0) * rb;
        if (!CK(cudaMemcpyAsync(in ? ss[i] + off : hs[i] + off, in ? hs[i] + off : ss[i] + off, len, kd, e->st))) return -1;
    }
    return 0;
}

// Layer l for T rows: hidden states X [T][d] (updated in place), rows RI (positions), keys up to max_ctx.
static int encode_layer(Eng* e, int l, int T, float* X, const RowInfo* RI, int max_ctx) {
    const MovaCfg* g = &e->c;
    const int d = g->d, qd = g->n_head * g->head_dim, kvd = g->n_kv * g->head_dim;
    const int big = T > MV_MAXT;
    AttnArgs aa = {g->n_head, g->n_kv, 128, attn_splits(T, max_ctx), 1.0f / sqrtf((float) g->head_dim)};
    Layer* L = &e->L[l];
    tgroup(e, MOVA_TG_EMBED_NORM);
    kc_gnorm(e->st, d, g->eps, X, (const uint16_t*) L->ln1.w0.p[0], e->xn, T);
    tgroup(e, MOVA_TG_ATTN_PROJ);
    enc_dense(e, &L->q, e->xn, d, e->q, qd, T, 0);
    enc_dense(e, &L->k, e->xn, d, e->k, kvd, T, 0);
    enc_dense(e, &L->g, e->xn, d, e->gq, qd, T, 0);
    tgroup(e, MOVA_TG_VALUES);
    if (!L->sparse) enc_dense(e, &L->v, e->xn, d, e->v, kvd, T, 0);
    else {
        int32_t* vi = e->vinds + (size_t) l * MAX_ROWS * g->top_kv;
        float* vw = e->vwts + (size_t) l * MAX_ROWS * g->top_kv;
        enc_router(e, &L->vr, &L->vb, e->xn, g->n_vexp, g->top_kv, vi, vw, e->vsel + (size_t) l * MAX_ROWS * g->n_vexp, T);
        if (e->pred_pending == l) CK(cudaStreamWaitEvent(e->st, e->ev_predv[l], 0));   // the predicted value experts
        kc_cache_admit(e->st, &e->pv, l - g->first_sparse, vi, T * g->top_kv, 0);
        kc_cache_copy(e->st, &e->pv, 256);
        if (!big) kc_mv_sel(e->st, L->vx.fmt, L->vx.d, L->vx.cols, L->vx.rows, e->xn, d, e->V, kvd, vi, T * g->top_kv, g->top_kv, stab_for(e, &L->vx));
        else {
            kc_bucket(e->st, vi, T * g->top_kv, g->n_vexp, e->vperm, e->vtiles, e->vntiles);
            kc_mm_grouped_dev(e->st, L->vx.fmt, L->vx.d, L->vx.cols, L->vx.rows, e->xn, d, e->V, kvd, g->top_kv, e->vperm, e->vtiles,
                              e->vntiles, T * g->top_kv / MMT_BN + g->n_vexp, stab_for(e, &L->vx));
        }
        kc_vcombine(e->st, e->V, vw, e->v, kvd, g->top_kv, T);
    }
    tgroup(e, MOVA_TG_ATTN);
    const KvView kv = kv_view(e, l);
    kc_rope_kv(e->st, e->q, e->k, e->v, kv, RI, e->inv, g->n_head, g->n_kv, T);
    if (big) kc_attn_prefill(e->st, aa, e->q, kv, RI, e->gq, e->ao, T);
    else kc_attn(e->st, aa, e->q, kv, RI, e->part, e->gq, e->ao, T);
    tgroup(e, MOVA_TG_ATTN_PROJ);
    enc_dense(e, &L->o, e->ao, qd, X, d, T, 1);
    tgroup(e, MOVA_TG_EMBED_NORM);
    kc_gnorm(e->st, d, g->eps, X, (const uint16_t*) L->ln2.w0.p[0], e->xn, T);
    if (!L->sparse) {
        tgroup(e, MOVA_TG_SHARED);
        enc_dense(e, &L->mg, e->xn, d, e->ga, g->ff_dense, T, 0);
        enc_dense(e, &L->mu, e->xn, d, e->ua, g->ff_dense, T, 0);
        kc_swiglu(e->st, e->ga, e->ua, e->aa, T * g->ff_dense);
        enc_dense(e, &L->md, e->aa, g->ff_dense, X, d, T, 1);
        return 0;
    }
    tgroup(e, MOVA_TG_ROUTER);
    int32_t* ri = e->inds + (size_t) l * MAX_ROWS * g->top_k;
    float* rw = e->wts + (size_t) l * MAX_ROWS * g->top_k;
    enc_router(e, &L->r, &L->rb, e->xn, g->n_exp, g->top_k, ri, rw, e->rsel + (size_t) l * MAX_ROWS * g->n_exp, T);
    if (e->pred_pending == l) { CK(cudaStreamWaitEvent(e->st, e->ev_pred[l], 0)); e->pred_pending = -1; }
    kc_cache_admit(e->st, &e->pm, l - g->first_sparse, ri, T * g->top_k, 0);
    kc_cache_copy(e->st, &e->pm, 256);
    if (!big && e->predict && !e->in_lm && l + 1 < g->n_layer) {   // decode: prefetch the next layer's likely experts
        const Layer* N = &e->L[l + 1];
        enc_router(e, &N->vr, &N->vb, e->xn, g->n_vexp, g->top_kv, e->pred_vinds, e->pred_w, e->pred_sel, T);
        kc_cache_admit(e->st, &e->pv_pre, l + 1 - g->first_sparse, e->pred_vinds, T * g->top_kv, CACHE_PROTECT_PREV);
        enc_router(e, &N->r, &N->rb, e->xn, g->n_exp, g->top_k, e->pred_inds, e->pred_w, e->pred_sel, T);
        kc_cache_admit(e->st, &e->pm_pre, l + 1 - g->first_sparse, e->pred_inds, T * g->top_k, CACHE_PROTECT_PREV);
        CK(cudaEventRecord(e->ev_admit, e->st));
        CK(cudaStreamWaitEvent(e->cst, e->ev_admit, 0));
        kc_cache_copy(e->cst, &e->pv_pre, 8);   // the value experts first: they are needed first
        CK(cudaEventRecord(e->ev_predv[l + 1], e->cst));
        kc_cache_copy(e->cst, &e->pm_pre, 8);
        CK(cudaEventRecord(e->ev_pred[l + 1], e->cst));
        e->pred_pending = l + 1;
    }
    tgroup(e, MOVA_TG_EXPERTS);
    const int P = T * g->top_k;
    if (!big) {
        if (L->eg.fmt == L->eu.fmt)
            kc_mv_gu(e->st, L->eg.fmt, L->eg.d, L->eu.d, L->eg.cols, L->eg.rows, e->xn, d, e->A, g->ff_exp, ri, P, g->top_k, stab_for(e, &L->eg));
        else {
            kc_mv_sel(e->st, L->eg.fmt, L->eg.d, L->eg.cols, L->eg.rows, e->xn, d, e->G, g->ff_exp, ri, P, g->top_k, stab_for(e, &L->eg));
            kc_mv_sel(e->st, L->eu.fmt, L->eu.d, L->eu.cols, L->eu.rows, e->xn, d, e->U, g->ff_exp, ri, P, g->top_k, stab_for(e, &L->eu));
            kc_swiglu(e->st, e->G, e->U, e->A, P * g->ff_exp);
        }
        kc_mv_sel(e->st, L->ed.fmt, L->ed.d, L->ed.cols, L->ed.rows, e->A, g->ff_exp, e->D, d, ri, P, 1, stab_for(e, &L->ed));
    } else {
        const int mt = P / MMT_BN + g->n_exp;
        kc_bucket(e->st, ri, P, g->n_exp, e->perm, e->tiles, e->ntiles);
        kc_mm_grouped_dev(e->st, L->eg.fmt, L->eg.d, L->eg.cols, L->eg.rows, e->xn, d, e->G, g->ff_exp, g->top_k, e->perm, e->tiles, e->ntiles, mt, stab_for(e, &L->eg));
        kc_mm_grouped_dev(e->st, L->eu.fmt, L->eu.d, L->eu.cols, L->eu.rows, e->xn, d, e->U, g->ff_exp, g->top_k, e->perm, e->tiles, e->ntiles, mt, stab_for(e, &L->eu));
        kc_swiglu(e->st, e->G, e->U, e->A, P * g->ff_exp);
        kc_mm_grouped_dev(e->st, L->ed.fmt, L->ed.d, L->ed.cols, L->ed.rows, e->A, g->ff_exp, e->D, d, 1, e->perm, e->tiles, e->ntiles, mt, stab_for(e, &L->ed));
    }
    tgroup(e, MOVA_TG_SHARED);
    enc_dense(e, &L->sg, e->xn, d, e->ga, g->ff_exp, T, 0);
    enc_dense(e, &L->su, e->xn, d, e->ua, g->ff_exp, T, 0);
    kc_swiglu(e->st, e->ga, e->ua, e->aa, T * g->ff_exp);
    enc_dense(e, &L->sd, e->aa, g->ff_exp, e->sh, d, T, 0);
    tgroup(e, MOVA_TG_EXPERTS);
    kc_moe_combine(e->st, e->D, rw, e->sh, X, d, g->top_k, T);
    return 0;
}

// Embedding, every layer and the final norm for T rows at e->ids / e->ri (one chunk; decode and verification).
static void dump_x(Eng* e, const float* X, int T) {   // debugging (MOVA_DUMP): append X (T x d f32)
    const char* dump = getenv("MOVA_DUMP");
    if (!dump) return;
    float* hx = (float*) malloc(sizeof(float) * (size_t) T * e->c.d);
    cudaMemcpyAsync(hx, X, sizeof(float) * (size_t) T * e->c.d, cudaMemcpyDeviceToHost, e->st);
    cudaStreamSynchronize(e->st);
    FILE* f = fopen(dump, "ab");
    if (f) { fwrite(hx, 4, (size_t) T * e->c.d, f); fclose(f); }
    free(hx);
}
static int encode_forward(Eng* e, int T, int max_ctx, int h0) {
    const MovaCfg* g = &e->c;
    tgroup(e, MOVA_TG_EMBED_NORM);
    kc_embed(e->st, e->embed.fmt, e->embed.w0, g->d, e->ids, e->x, T);
    for (int l = 0; l < g->n_layer; ++l) {
        dump_x(e, e->x, T);
        if (encode_layer(e, l, T, e->x, e->ri, max_ctx)) return -1;
    }
    dump_x(e, e->x, T);
    if (h0 < T) {
        tgroup(e, MOVA_TG_HEAD);
        kc_gnorm(e->st, g->d, g->eps, e->x, (const uint16_t*) e->norm.w0.p[0], e->xn, T);
    }
    return 0;
}

// The LM head for rows [h0, h0 + n) (n <= MAX_LOGIT_ROWS) of the normed activations: logits + argmax.
static void encode_head(Eng* e, int h0, int n) {
    const MovaCfg* g = &e->c;
    tgroup(e, MOVA_TG_HEAD);
    enc_dense(e, &e->head, e->xn + (size_t) h0 * g->d, g->d, e->logits, g->vocab, n, 0);
    kc_argmax(e->st, e->logits, e->am, g->vocab, n);
}

static void route_collect(Eng* e, int T);

// Forward of T rows (tokens tok[0..T-1] at positions pos0..pos0+T-1).  Rows [h0, T) get logits (into logits_out,
// (T - h0) x vocab, may be NULL) and arg max (into am, may be NULL); h0 = T: no head.
static int forward(Eng* e, const int32_t* tok, int T, int pos0, int h0, float* logits_out, int32_t* am) {
    kc_seed_gemm_f32(e->seed_f32);   // a launch setting of the kernels' host side (engines in one process may differ)
    if (T < 1 || T > MAX_ROWS) return -1;
    if (pos0 + T > e->kv_cap) { fprintf(stderr, "mova engine: KV capacity %lld exceeded\n", (long long) e->kv_cap); return -1; }
    memcpy(e->h_ids, tok, (size_t) T * 4);
    for (int t = 0; t < T; ++t) { memset(&e->h_ri[t], 0, sizeof e->h_ri[t]); e->h_ri[t].pos = pos0 + t; }
    if (!CK(cudaMemcpyAsync(e->ids, e->h_ids, (size_t) T * 4, cudaMemcpyHostToDevice, e->st)) ||
        !CK(cudaMemcpyAsync(e->ri, e->h_ri, (size_t) T * sizeof(RowInfo), cudaMemcpyHostToDevice, e->st)))
        return -1;
    // Forwards of up to MV_MAXT rows (decode, prompt-lookup verification) have no host round trip: they run as a CUDA
    // graph, captured once per (rows, attention splits, head) and launched as one unit.  The head is all rows or none
    // (a head from row 0 < h0 < T would be baked into the graph).
    const int graph = T <= MV_MAXT && !e->timing_on && !e->no_graph && (h0 == 0 || h0 >= T);
    if (graph) {
        const int key = ((T * (MAX_SPLITS + 1) + attn_splits(T, pos0 + T)) * 2 + (h0 < T));
        if (!e->graphs[key]) {
            if (e->ngraphs >= MAX_GRAPHS) {   // drop the least recently launched
                int lru = -1;
                for (int i = 0; i < (int) (sizeof e->graphs / sizeof e->graphs[0]); ++i)
                    if (e->graphs[i] && (lru < 0 || e->graph_use[i] < e->graph_use[lru])) lru = i;
                if (!CK(cudaStreamSynchronize(e->st))) return -1;   // it may still be running
                cudaGraphExecDestroy(e->graphs[lru]);
                e->graphs[lru] = NULL;
                --e->ngraphs;
            }
            cudaGraph_t gr;
            if (!CK(cudaStreamBeginCapture(e->st, cudaStreamCaptureModeThreadLocal))) return -1;
            const int rc = encode_forward(e, T, pos0 + T, h0);
            if (h0 < T) encode_head(e, h0, T - h0);
            if (!CK(cudaStreamEndCapture(e->st, &gr)) || rc) return -1;
            if (!CK(cudaGraphInstantiate(&e->graphs[key], gr, 0))) { e->graphs[key] = NULL; cudaGraphDestroy(gr); return -1; }
            cudaGraphDestroy(gr);
            ++e->ngraphs;
        }
        e->graph_use[key] = ++e->graph_tick;
        if (!CK(cudaGraphLaunch(e->graphs[key], e->st))) return -1;
    } else if (encode_forward(e, T, pos0 + T, h0)) return -1;
    for (int r = h0; r < T; r += MAX_LOGIT_ROWS) {
        const int n = T - r < MAX_LOGIT_ROWS ? T - r : MAX_LOGIT_ROWS;
        if (!graph) encode_head(e, r, n);
        if (logits_out && !CK(cudaMemcpyAsync(e->h_logits, e->logits, (size_t) n * e->c.vocab * 4, cudaMemcpyDeviceToHost, e->st))) return -1;
        if (am && !CK(cudaMemcpyAsync(e->h_am, e->am, (size_t) n * 4, cudaMemcpyDeviceToHost, e->st))) return -1;
        if (sync_stream(e)) return -1;
        if (logits_out) memcpy(logits_out + (size_t) (r - h0) * e->c.vocab, e->h_logits, (size_t) n * e->c.vocab * 4);
        if (am) memcpy(am + (r - h0), e->h_am, (size_t) n * 4);
    }
    if (sync_stream(e)) return -1;
    if (!CK(cudaGetLastError())) return -1;
    if (e->route_on) route_collect(e, T);
    return 0;
}

#define PREFETCH_ROWS 64
// Admit every expert of sparse layer l (main stream) and copy the misses on the copy stream; ev_ready[l] when done.
static void prefetch_layer(Eng* e, int l) {
    const int sl = l - e->c.first_sparse;
    kc_cache_admit(e->st, &e->pv_pf, sl, e->all_ids, e->c.n_vexp, 0);
    kc_cache_admit(e->st, &e->pm_pf, sl, e->all_ids, e->c.n_exp, 0);
    cudaEventRecord(e->ev_admit, e->st);
    cudaStreamWaitEvent(e->cst, e->ev_admit, 0);
    kc_cache_copy(e->cst, &e->pv_pf, 8);
    kc_cache_copy(e->cst, &e->pm_pf, 8);
    cudaEventRecord(e->ev_ready[l], e->cst);
}

// Rows ids[0..n) at positions pos0.., layer by layer over up to xmax rows at a time (sub-chunks of MAX_ROWS): every
// layer's experts come to VRAM once per xmax rows, not once per chunk.  Rows [h0, n) get logits (into logits_out,
// (n - h0) x vocab, may be NULL).  Route capture takes the chunked forward (its records are per forward).
static int forward_lm_body(Eng* e, const int32_t* ids, int n, int pos0, int h0, float* logits_out);
static int forward_lm(Eng* e, const int32_t* ids, int n, int pos0, int h0, float* logits_out) {
    if (e->route_on || n <= MV_MAXT) return forward_lm_body(e, ids, n, pos0, h0, logits_out);
    // a sub-chunk of 1 .. MV_MAXT rows takes the decode path, whose next-layer prediction could replace a slot the
    // pending prefetch copy is filling
    e->in_lm = 1;
    const int rc = forward_lm_body(e, ids, n, pos0, h0, logits_out);
    e->in_lm = 0;
    return rc;
}
static int forward_lm_body(Eng* e, const int32_t* ids, int n, int pos0, int h0, float* logits_out) {
    kc_seed_gemm_f32(e->seed_f32);
    if (e->route_on || n <= MV_MAXT) {
        for (int p = 0; p < n; p += MAX_ROWS) {
            const int T = n - p < MAX_ROWS ? n - p : MAX_ROWS;
            int hh = h0 - p;
            if (hh < 0) hh = 0;
            if (hh > T) hh = T;
            if (forward(e, ids + p, T, pos0 + p, hh, hh < T && logits_out ? logits_out + (size_t) (p + hh - h0) * e->c.vocab : NULL, NULL)) return -1;
        }
        return 0;
    }
    const MovaCfg* g = &e->c;
    const int d = g->d;
    if (pos0 + n > e->kv_cap) { fprintf(stderr, "mova engine: KV capacity %lld exceeded\n", (long long) e->kv_cap); return -1; }
    for (int s0 = 0; s0 < n; s0 += e->xmax) {
        const int N = n - s0 < e->xmax ? n - s0 : e->xmax;
        memcpy(e->h_ids_all, ids + s0, sizeof(int32_t) * (size_t) N);
        for (int i = 0; i < N; ++i) { memset(&e->h_ri_all[i], 0, sizeof(RowInfo)); e->h_ri_all[i].pos = pos0 + s0 + i; }
        if (!CK(cudaMemcpyAsync(e->ids_all, e->h_ids_all, sizeof(int32_t) * (size_t) N, cudaMemcpyHostToDevice, e->st)) ||
            !CK(cudaMemcpyAsync(e->ri_all, e->h_ri_all, sizeof(RowInfo) * (size_t) N, cudaMemcpyHostToDevice, e->st)))
            return -1;
        tgroup(e, MOVA_TG_EMBED_NORM);
        kc_embed(e->st, e->embed.fmt, e->embed.w0, d, e->ids_all, e->x_all, N);
        // Prefetch (prompts of more than PREFETCH_ROWS rows use nearly every expert of every layer): every expert of
        // sparse layer l + 1 is admitted in stream order on the main stream when layer l starts, and copied on the
        // copy stream, which first waits for that point (layer l - 1, the last reader of the slots it may replace, is
        // done); layer l + 1 waits for the copy.  The layer's own admits then find their experts resident.
        const int pre = N > PREFETCH_ROWS;
        for (int l = 0; l < g->n_layer; ++l) {
            if (pre) {
                if (l == 0) prefetch_layer(e, g->first_sparse);   // once: a second admit would race its copy's job list
                if (l >= g->first_sparse) CK(cudaStreamWaitEvent(e->st, e->ev_ready[l], 0));
                if (l + 1 < g->n_layer && l + 1 > g->first_sparse) prefetch_layer(e, l + 1);
            }
            // host KV rows: this pass reads rows [0, p0 - nv) and writes [p0 - nv, p1 - nv); staged while the layer runs
            const int64_t p0 = pos0 + s0, p1 = p0 + N, hr0 = p0 > e->kv_nv ? p0 - e->kv_nv : 0, hr1 = p1 - e->kv_nv;
            const int stage = hr1 > 0;
            if (stage) {
                if (kv_stage_copy(e, l, 0, hr0, 1)) return -1;
                e->kv_staged = l;
            }
            for (int c0 = 0; c0 < N; c0 += MAX_ROWS) {
                const int T = N - c0 < MAX_ROWS ? N - c0 : MAX_ROWS;
                if (encode_layer(e, l, T, e->x_all + (size_t) c0 * d, e->ri_all + c0, pos0 + s0 + c0 + T)) { e->kv_staged = -1; return -1; }
            }
            if (stage) {
                e->kv_staged = -1;
                if (kv_stage_copy(e, l, hr0, hr1, 0)) return -1;
            }
        }
        // the head for this range's rows at or after h0
        for (int r = h0 > s0 ? h0 - s0 : 0; r < N; r += MAX_LOGIT_ROWS) {
            const int m = N - r < MAX_LOGIT_ROWS ? N - r : MAX_LOGIT_ROWS;
            tgroup(e, MOVA_TG_HEAD);
            kc_gnorm(e->st, d, g->eps, e->x_all + (size_t) r * d, (const uint16_t*) e->norm.w0.p[0], e->xn, m);
            enc_dense(e, &e->head, e->xn, d, e->logits, g->vocab, m, 0);
            if (logits_out && !CK(cudaMemcpyAsync(e->h_logits, e->logits, (size_t) m * g->vocab * 4, cudaMemcpyDeviceToHost, e->st))) return -1;
            if (sync_stream(e)) return -1;
            if (logits_out) memcpy(logits_out + (size_t) (s0 + r - h0) * g->vocab, e->h_logits, (size_t) m * g->vocab * 4);
        }
    }
    if (sync_stream(e) || !CK(cudaGetLastError())) return -1;
    return 0;
}

// ---- sequences (one slot) -------------------------------------------------------------------------------------------

static void hist_push(Seq* s, int32_t t) {
    if (s->len == s->cap) { s->cap *= 2; s->hist = (int32_t*) realloc(s->hist, sizeof(int32_t) * (size_t) s->cap); }
    s->hist[s->len++] = t;
}

int eng_prefill(Eng* e, int seq, const int32_t* ids, int n) {
    if (seq != 0 || n < 1 || n > e->kv_cap) return -1;
    e->seq.len = 0;
    for (int i = 0; i < n; ++i) hist_push(&e->seq, ids[i]);
    if (n > 1 && forward_lm(e, ids, n - 1, 0, n - 1, NULL)) { e->seq.len = 0; return -1; }   // no half-written KV
    return 0;
}
int eng_prefill_cached(Eng* e, int seq, const int32_t* ids, int n, int* reused) {
    if (seq != 0 || n < 1 || n > e->kv_cap) return -1;
    int c = 0;
    while (c < e->seq.len - 1 && c < n - 1 && e->seq.hist[c] == ids[c]) ++c;   // KV is valid for hist[0 .. len-2]
    e->seq.len = c;
    for (int i = c; i < n; ++i) hist_push(&e->seq, ids[i]);
    if (n - 1 > c && forward_lm(e, ids + c, n - 1 - c, c, n - 1 - c, NULL)) { e->seq.len = 0; return -1; }
    if (reused) *reused = c;
    return 0;
}
int eng_rewind(Eng* e, int seq, int n) {
    if (seq != 0 || n < 1 || n > e->seq.len) return -1;
    e->seq.len = n;
    return 0;
}
int eng_fork(Eng* e, int dst, int src) { (void) e; return dst == src ? 0 : -1; }
void eng_free(Eng* e, int seq) { if (seq == 0) e->seq.len = 0; }
int eng_len(Eng* e, int seq) { return seq == 0 ? e->seq.len : 0; }

int eng_step(Eng* e, int seq, float* logits) {
    if (seq != 0 || e->seq.len < 1) return -1;
    const int32_t t = e->seq.hist[e->seq.len - 1];
    return forward(e, &t, 1, e->seq.len - 1, 0, logits, NULL);
}
int eng_push(Eng* e, int seq, int32_t tok) {
    if (seq != 0) return -1;
    hist_push(&e->seq, tok);
    return 0;
}

// Prompt-lookup speculative decoding (ENG_MODE_PL), as engine/mova_gpu.m: the draft continues the most recent earlier
// occurrence of the history's last PL_NMAX..PL_NMIN tokens; one forward verifies [pending, d_1 .. d_k] with the decode
// kernels, and the matching prefix plus the correction / bonus token is committed.
#ifndef PL_K
#define PL_K 4
#endif
#define PL_NMAX 4
#define PL_NMIN 3
static int pl_draft(const Seq* s, int32_t* d) {
    const int L = s->len;
    for (int n = PL_NMAX; n >= PL_NMIN; --n) {
        if (L < n + 1) continue;
        const int32_t* suf = s->hist + L - n;
        for (int i = L - n - 1; i >= 0; --i) {
            if (memcmp(s->hist + i, suf, (size_t) n * 4)) continue;
            int k = 0;
            while (k < PL_K && i + n + k < L) { d[k] = s->hist[i + n + k]; ++k; }
            if (k) return k;
        }
    }
    return 0;
}
static int gen_pl(Eng* e, int n_new, int32_t* out, EngStats* st) {
    int32_t tok[PL_K + 1], am[PL_K + 1], d[PL_K];
    for (int done = 0; done < n_new;) {
        const int nd = pl_draft(&e->seq, d), p = e->seq.len - 1;
        tok[0] = e->seq.hist[p];
        for (int j = 0; j < nd; ++j) tok[j + 1] = d[j];
        if (forward(e, tok, nd + 1, p, 0, NULL, am)) return -1;
        int a = 0;
        while (a < nd && d[a] == am[a]) ++a;
        for (int j = 0; j <= a && done < n_new; ++j) {
            hist_push(&e->seq, am[j]);
            out[done++] = am[j];
            if (st) st->tokens++;
        }
        if (st) { st->forwards++; st->rows += nd + 1; st->cycles++; st->proposals += nd; st->accepted += a; }
    }
    return 0;
}

int eng_generate(Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st) {
    if (nseq != 1 || seqs[0] != 0) return -1;
    if (mode == ENG_MODE_PL) return gen_pl(e, n_new, out, st);
    for (int j = 0; j < n_new; ++j) {
        int32_t am;
        const int32_t t = e->seq.hist[e->seq.len - 1];
        if (forward(e, &t, 1, e->seq.len - 1, 0, NULL, &am)) return -1;
        hist_push(&e->seq, am);
        out[j] = am;
        if (st) { st->tokens++; st->forwards++; st->rows++; }
    }
    return 0;
}

int eng_score(Eng* e, int seq, const int32_t* ids, int from, int count, float* logits) {
    if (seq != 0 || from < 1 || count < 1) return -1;
    const int last = from + count - 2;
    if (forward_lm(e, ids, last + 1, 0, from - 1, logits)) return -1;
    e->seq.len = 0;
    return 0;
}

// ---- route capture and timing ---------------------------------------------------------------------------------------

int eng_mova_routes(Eng* e, int on, int max_rows) {
    free(e->route_mlp); free(e->route_val); free(e->route_mlp_sel); free(e->route_val_sel);
    e->route_mlp = NULL; e->route_val = NULL; e->route_mlp_sel = NULL; e->route_val_sel = NULL;
    e->route_on = on;
    e->route_rows = 0;
    e->route_max = max_rows;
    if (!on) return 0;
    const MovaCfg* c = &e->c;
    const size_t ns = (size_t) (c->n_layer - c->first_sparse);
    e->route_mlp = (int32_t*) malloc((size_t) max_rows * ns * (size_t) c->top_k * 4);
    e->route_val = (int32_t*) malloc((size_t) max_rows * ns * (size_t) c->top_kv * 4);
    e->route_mlp_sel = (float*) malloc((size_t) max_rows * ns * (size_t) c->n_exp * 4);
    e->route_val_sel = (float*) malloc((size_t) max_rows * ns * (size_t) c->n_vexp * 4);
    return 0;
}
static void route_collect(Eng* e, int T) {
    const MovaCfg* c = &e->c;
    const int ns = c->n_layer - c->first_sparse;
    for (int t = 0; t < T && e->route_rows < e->route_max; ++t, ++e->route_rows) {
        const size_t r = (size_t) e->route_rows;
        for (int s = 0; s < ns; ++s) {
            const int l = c->first_sparse + s;
            cudaMemcpy(e->route_mlp + (r * ns + s) * c->top_k, e->inds + ((size_t) l * MAX_ROWS + t) * c->top_k, (size_t) c->top_k * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(e->route_val + (r * ns + s) * c->top_kv, e->vinds + ((size_t) l * MAX_ROWS + t) * c->top_kv, (size_t) c->top_kv * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(e->route_mlp_sel + (r * ns + s) * c->n_exp, e->rsel + ((size_t) l * MAX_ROWS + t) * c->n_exp, (size_t) c->n_exp * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(e->route_val_sel + (r * ns + s) * c->n_vexp, e->vsel + ((size_t) l * MAX_ROWS + t) * c->n_vexp, (size_t) c->n_vexp * 4, cudaMemcpyDeviceToHost);
        }
    }
}
int eng_mova_routes_read(Eng* e, int rows, int32_t* mlp, int32_t* val, float* mlp_sel, float* val_sel) {
    if (!e->route_on || rows > e->route_rows) return -1;
    const MovaCfg* c = &e->c;
    const size_t ns = (size_t) (c->n_layer - c->first_sparse), r0 = (size_t) (e->route_rows - rows);
    if (mlp) memcpy(mlp, e->route_mlp + r0 * ns * c->top_k, (size_t) rows * ns * c->top_k * 4);
    if (val) memcpy(val, e->route_val + r0 * ns * c->top_kv, (size_t) rows * ns * c->top_kv * 4);
    if (mlp_sel) memcpy(mlp_sel, e->route_mlp_sel + r0 * ns * c->n_exp, (size_t) rows * ns * c->n_exp * 4);
    if (val_sel) memcpy(val_sel, e->route_val_sel + r0 * ns * c->n_vexp, (size_t) rows * ns * c->n_vexp * 4);
    return 0;
}
int eng_mova_timing(Eng* e, int on) {
    e->timing_on = on;
    e->nev = 0;
    e->cur_group = -1;
    if (on) memset(e->tg_sec, 0, sizeof e->tg_sec);
    return 0;
}
int eng_mova_timing_read(Eng* e, double* seconds) {
    memcpy(seconds, e->tg_sec, sizeof e->tg_sec);
    return 0;
}
