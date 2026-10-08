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
#ifndef MAX_SPLITS
#define MAX_SPLITS 32            // decode attention: at most this many key splits
#endif
#define MAX_LOGIT_ROWS 64        // LM head rows per pass
#define MAX_TILES (MAXP / MM_BN + 128)

static int cuda_ok(cudaError_t r, const char* what, const char* file, int line) {
    if (r == cudaSuccess) return 1;
    fprintf(stderr, "mova engine: %s failed at %s:%d: %s\n", what, file, line, cudaGetErrorString(r));
    return 0;
}
#define CK(x) cuda_ok((x), #x, __FILE__, __LINE__)

typedef struct {
    int fmt, slices, rows, cols;
    WSlice w0;      // slice 0: dense tensors
    WSlice* d;      // device table [slices] (stacked experts)
    WSlice* h;      // its host copy
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

typedef struct {   // a sub-allocated arena (VRAM or mapped host memory)
    uint8_t* base;   // device address
    uint8_t* host;   // host address (host arenas)
    uint64_t size, used;
} Arena;

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
    uint16_t* Kc;       // [n_layer][kv_cap][n_kv * 128]
    uint16_t* Vc;
    int64_t kv_cap;
    // scratch (MAX_ROWS rows), device
    float *x, *xn, *q, *k, *v, *gq, *ao, *ga, *ua, *aa, *G, *U, *A, *D, *V, *sh, *logits, *inv, *part, *wts, *vwts, *rsel, *vsel, *rscore;
    int32_t *ids, *inds, *vinds, *am, *perm, *vperm;
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
    Arena vram, host;
    int64_t experts_vram, experts_host;   // expert units by placement
    EngMem mem;
    Seq seq;
    char desc[320];
    // route capture
    int route_on, route_max;
    int32_t *route_mlp, *route_val;
    float *route_mlp_sel, *route_val_sel;
    int route_rows;
    // timing
    int timing_on;
    double tg_sec[MOVA_TG_N];
    cudaEvent_t ev[2048];
    short ev_group[2048];
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
static void* arena_take(Arena* a, uint64_t bytes, uint8_t** host) {
    const uint64_t at = (a->used + 255) & ~255ull;
    if (at + bytes > a->size) return NULL;
    a->used = at + bytes;
    if (host) *host = a->host ? a->host + at : NULL;
    return a->base + at;
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
    const uint64_t one = (uint64_t) t->rows * t->cols * 2;
    uint8_t* d = (uint8_t*) dalloc(e, one * (uint64_t) t->slices, &e->mem.weights);
    if (!d) { snprintf(err, (size_t) errlen, "%s: out of GPU memory", t->name); return -1; }
    char nm[128];
    for (int s = 0; s < t->slices; ++s) {
        const uint16_t* src = mova_ckpt_bf16(e->ck, mova_slice_name(t, s, nm, sizeof nm), t->rows, t->cols, err, errlen);
        if (!src || !CK(cudaMemcpy(d + one * (uint64_t) s, src, one, cudaMemcpyHostToDevice))) return -1;
    }
    w->w0.p[0] = d;
    if (t->slices > 1) {
        w->h = (WSlice*) calloc((size_t) t->slices, sizeof(WSlice));
        for (int s = 0; s < t->slices; ++s) w->h[s].p[0] = d + one * (uint64_t) s;
    }
    return 0;
}

// Stacked experts: shape only (placement and upload come later, in place_experts).
static int describe_stacked(const NsTensor* t, MW* w) {
    memset(w, 0, sizeof *w);
    w->fmt = t->enc; w->slices = t->slices; w->rows = t->rows; w->cols = t->cols;
    w->h = (WSlice*) calloc((size_t) t->slices, sizeof(WSlice));
    if (t->enc == MF_SEED4 || t->enc == MF_SEED4P4)
        for (int s = 0; s < t->slices; ++s) memcpy(&w->h[s].eb, t->s[2].p + 4 * (size_t) s, 4);
    return 0;
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

// One expert unit: an MLP expert (gate, up, down slice) or a value expert (one slice of vx).
typedef struct {
    MW* w[3];
    const NsTensor* t[3];
    int nw, slice;
    uint64_t bytes;
} Unit;

static uint64_t unit_bytes(const Unit* u) {
    uint64_t b = 0;
    for (int i = 0; i < u->nw; ++i)
        for (int s = 0; s < 4; ++s) b += (stream_slice_len(u->w[i], s) + 255) & ~255ull;
    return b;
}
// Copy a unit's slices into an arena and point the tables at them.
static int put_unit(Eng* e, Unit* u, Arena* a) {
    for (int i = 0; i < u->nw; ++i) {
        MW* w = u->w[i];
        for (int s = 0; s < 4; ++s) {
            const uint64_t len = stream_slice_len(w, s);
            if (!len) continue;
            uint8_t* host = NULL;
            uint8_t* d = (uint8_t*) arena_take(a, len, &host);
            if (!d) return -1;
            const uint8_t* src = u->t[i]->s[s].p + len * (uint64_t) u->slice;
            if (host) memcpy(host, src, len);
            else if (!CK(cudaMemcpy(d, src, len, cudaMemcpyHostToDevice))) return -1;
            w->h[u->slice].p[s] = d;
        }
    }
    return 0;
}

// Placement: units round-robin over layers so every layer gets the same share of VRAM, value experts and MLP experts
// interleaved by their byte share; the rest go to host memory.
static int place_experts(Eng* e, char* err, int errlen) {
    const MovaCfg* c = &e->c;
    const int nl = c->n_layer - c->first_sparse;
    Unit* units = (Unit*) calloc((size_t) nl * (size_t) (c->n_exp + c->n_vexp), sizeof(Unit));
    int nu = 0;
    uint64_t total = 0;
    // order: expert index major, layer minor, so a prefix of the list is an even share of every layer
    const int emax = c->n_exp > c->n_vexp ? c->n_exp : c->n_vexp;
    for (int x = 0; x < emax; ++x)
        for (int l = c->first_sparse; l < c->n_layer; ++l) {
            Layer* L = &e->L[l];
            char nm[160];
            if (x < c->n_vexp && L->vx.h && !L->vx.w0.p[0]) {
                Unit* u = &units[nu++];
                u->nw = 1; u->slice = x; u->w[0] = &L->vx;
                snprintf(nm, sizeof nm, "model.layers.%d.self_attn.v_experts.weight", l);
                u->t[0] = ns_find(&e->nm, nm);
                u->bytes = unit_bytes(u);
                total += u->bytes;
            }
            if (x < c->n_exp && L->eg.h && !L->eg.w0.p[0]) {
                Unit* u = &units[nu++];
                u->nw = 3; u->slice = x; u->w[0] = &L->eg; u->w[1] = &L->eu; u->w[2] = &L->ed;
                const char* pr[3] = {"gate_proj", "up_proj", "down_proj"};
                for (int i = 0; i < 3; ++i) {
                    snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%s.weight", l, pr[i]);
                    u->t[i] = ns_find(&e->nm, nm);
                }
                u->bytes = unit_bytes(u);
                total += u->bytes;
            }
        }
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    const char* rs = getenv("NSLM_VRAM_RESERVE_MB");
    const uint64_t reserve = (uint64_t) (rs ? atoll(rs) : 768) << 20;
    uint64_t budget = fr > reserve ? fr - reserve : 0;
    const char* cap = getenv("NSLM_EXPERT_VRAM_MB");   // testing: cap the experts' VRAM
    if (cap && ((uint64_t) atoll(cap) << 20) < budget) budget = (uint64_t) atoll(cap) << 20;
    if (budget > total) budget = total;
    // VRAM arena
    while (budget > 0 && !(e->vram.base = (uint8_t*) dalloc(e, budget, NULL))) budget -= 256ull << 20;
    e->vram.size = e->vram.base ? budget : 0;
    e->mem.weights += (int64_t) e->vram.size;
    int i = 0;
    for (; i < nu; ++i) {
        const uint64_t at = e->vram.used;
        if (put_unit(e, &units[i], &e->vram)) { e->vram.used = at; break; }
        e->experts_vram++;
    }
    // host arena for the rest
    uint64_t rest = 0;
    for (int j = i; j < nu; ++j) rest += units[j].bytes + 4 * 256;
    if (rest) {
        uint8_t* h = (uint8_t*) halloc(e, rest);
        if (!h) { snprintf(err, (size_t) errlen, "cannot pin %.2f GB of host memory for the experts", rest / 1e9); free(units); return -1; }
        void* dp = NULL;
        if (!CK(cudaHostGetDevicePointer(&dp, h, 0))) { free(units); return -1; }
        e->host.base = (uint8_t*) dp;
        e->host.host = h;
        e->host.size = rest;
        e->mem.staging += (int64_t) rest;
        for (; i < nu; ++i) {
            if (put_unit(e, &units[i], &e->host)) { snprintf(err, (size_t) errlen, "host expert arena overflow"); free(units); return -1; }
            e->experts_host++;
        }
    }
    free(units);
    return 0;
}

// Device copies of the slice tables of every stacked tensor.
static int upload_tables(Eng* e, MW* w) {
    if (!w->h) return 0;
    w->d = (WSlice*) dalloc(e, sizeof(WSlice) * (size_t) w->slices, &e->mem.scratch);
    return w->d && CK(cudaMemcpy(w->d, w->h, sizeof(WSlice) * (size_t) w->slices, cudaMemcpyHostToDevice)) ? 0 : -1;
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
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); free(e); return NULL; }
    if (!CK(cudaSetDevice(0)) || !CK(cudaStreamCreateWithFlags(&e->st, cudaStreamNonBlocking))) { snprintf(err, (size_t) errlen, "CUDA init failed"); free(e); return NULL; }
    if (ns_open(&e->nm, o->model_dir, err, errlen)) { free(e); return NULL; }
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
        cudaMemcpy(e->stab, g, 65536 * 4, cudaMemcpyHostToDevice);
        if (seeds4) {
            e->stab32 = (uint32_t*) dalloc(e, 65536 * 4, &e->mem.lut);
            for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream32((uint16_t) s);
            cudaMemcpy(e->stab32, g, 65536 * 4, cudaMemcpyHostToDevice);
        } else e->stab32 = e->stab;
        free(g);
        (void) seeds;
    }
    // KV caches
    const int kvd = c->n_kv * c->head_dim;
    e->kv_cap = o->kv_tokens > 0 ? o->kv_tokens : 32768;
    const uint64_t kvl = (uint64_t) e->kv_cap * kvd * 2;
    e->Kc = (uint16_t*) dalloc(e, kvl * (uint64_t) c->n_layer, &e->mem.kv);
    e->Vc = (uint16_t*) dalloc(e, kvl * (uint64_t) c->n_layer, &e->mem.kv);
    if (!e->Kc || !e->Vc) { snprintf(err, (size_t) errlen, "KV cache (%lld tokens): out of GPU memory", (long long) e->kv_cap); eng_close(e); return NULL; }
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
    if (!e->h_ids || !e->h_ri || !e->h_inds || !e->h_perm || !e->h_tiles || !e->h_am || !e->h_logits) {
        snprintf(err, (size_t) errlen, "pinned host staging: out of memory");
        eng_close(e);
        return NULL;
    }
    // experts last: they take the VRAM that is left
    if (place_experts(e, err, errlen)) { eng_close(e); return NULL; }
    for (int l = 0; l < c->n_layer; ++l) {
        Layer* L = &e->L[l];
        if (upload_tables(e, &L->vx) || upload_tables(e, &L->eg) || upload_tables(e, &L->eu) || upload_tables(e, &L->ed)) {
            snprintf(err, (size_t) errlen, "slice tables: out of GPU memory");
            eng_close(e);
            return NULL;
        }
    }
    e->seq.cap = 1024;
    e->seq.hist = (int32_t*) malloc(sizeof(int32_t) * (size_t) e->seq.cap);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const char* fm[5] = {"bf16", "seed4", "q8", "q4", "seed4p4"};
    snprintf(e->desc, sizeof e->desc,
             "mova engine (CUDA, %s): experts %s/%s/%s, attention %s, value experts %s, embed %s, head %s; "
             "expert units %lld in VRAM, %lld in host memory",
             prop.name, fm[e->L[c->first_sparse].eg.fmt], fm[e->L[c->first_sparse].eu.fmt], fm[e->L[c->first_sparse].ed.fmt],
             fm[e->L[0].q.fmt], fm[e->L[c->first_sparse].vx.fmt], fm[e->embed.fmt], fm[e->head.fmt], (long long) e->experts_vram,
             (long long) e->experts_host);
    return e;
}

void eng_close(Eng* e) {
    if (!e) return;
    if (e->st) cudaStreamSynchronize(e->st);
    for (int i = 0; i < e->nalloc; ++i) cudaFree(e->allocs[i]);
    for (int i = 0; i < e->nhalloc; ++i) cudaFreeHost(e->hallocs[i]);
    free(e->allocs);
    free(e->hallocs);
    if (e->L) {
        for (int l = 0; l < e->c.n_layer; ++l) { free(e->L[l].vx.h); free(e->L[l].eg.h); free(e->L[l].eu.h); free(e->L[l].ed.h); }
        free(e->L);
    }
    for (int i = 0; i < e->nev; ++i) cudaEventDestroy(e->ev[i]);
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
    if (!e->timing_on || group == e->cur_group || e->nev >= 2048) return;
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
// Host bucketing of T x k selections by expert: perm = pair ids grouped by expert (ascending within an expert),
// tiles = runs of <= MM_BN pairs.  Reads the selections from the device (a sync point), uploads perm and tiles.
static int bucket(Eng* e, const int32_t* d_inds, int T, int k, int n, int32_t* d_perm, MmTile* d_tiles) {
    const int P = T * k;
    if (!CK(cudaMemcpyAsync(e->h_inds, d_inds, (size_t) P * 4, cudaMemcpyDeviceToHost, e->st)) || sync_stream(e)) return -1;
    int cnt[128] = {0}, off[129], pos[128];
    for (int p = 0; p < P; ++p) cnt[e->h_inds[p]]++;
    off[0] = 0;
    for (int x = 0; x < n; ++x) off[x + 1] = off[x] + cnt[x];
    memcpy(pos, off, sizeof(int) * (size_t) n);
    for (int p = 0; p < P; ++p) e->h_perm[pos[e->h_inds[p]]++] = p;
    int nt = 0;
    for (int x = 0; x < n; ++x)
        for (int s = off[x]; s < off[x + 1]; s += MM_BN) {
            MmTile t = {x, s, off[x + 1] - s < MM_BN ? off[x + 1] - s : MM_BN, 0};
            e->h_tiles[nt++] = t;
        }
    if (!CK(cudaMemcpyAsync(d_perm, e->h_perm, (size_t) P * 4, cudaMemcpyHostToDevice, e->st)) ||
        !CK(cudaMemcpyAsync(d_tiles, e->h_tiles, (size_t) nt * sizeof(MmTile), cudaMemcpyHostToDevice, e->st)))
        return -1;
    // the staging is reused by the next bucket(): wait for these copies
    return sync_stream(e) ? -1 : nt;
}

static int encode_forward(Eng* e, int T, int max_ctx, int h0) {
    const MovaCfg* g = &e->c;
    const int d = g->d, qd = g->n_head * g->head_dim, kvd = g->n_kv * g->head_dim;
    const int big = T > MV_MAXT;
    tgroup(e, MOVA_TG_EMBED_NORM);
    kc_embed(e->st, e->embed.fmt, e->embed.w0, d, e->ids, e->x, T);
    int ns = (max_ctx + 127) / 128;
    if (ns > MAX_SPLITS) ns = MAX_SPLITS;
    while (ns > 1 && T * ns > 1024) --ns;
    if (ns < 1) ns = 1;
    AttnArgs aa = {g->n_head, g->n_kv, 128, ns, 1.0f / sqrtf((float) g->head_dim)};
    const uint64_t kvl = (uint64_t) e->kv_cap * kvd;
    const char* dump = getenv("MOVA_DUMP");   // debugging: append x (T x d f32) at every layer start and at the end
    for (int l = 0; l <= g->n_layer; ++l) {
        if (dump) {
            float* hx = (float*) malloc(sizeof(float) * (size_t) T * d);
            cudaMemcpyAsync(hx, e->x, sizeof(float) * (size_t) T * d, cudaMemcpyDeviceToHost, e->st);
            cudaStreamSynchronize(e->st);
            FILE* f = fopen(dump, "ab");
            if (f) { fwrite(hx, 4, (size_t) T * d, f); fclose(f); }
            free(hx);
        }
        if (l == g->n_layer) break;
        Layer* L = &e->L[l];
        tgroup(e, MOVA_TG_EMBED_NORM);
        kc_gnorm(e->st, d, g->eps, e->x, (const uint16_t*) L->ln1.w0.p[0], e->xn, T);
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
            if (!big) kc_mv_sel(e->st, L->vx.fmt, L->vx.d, L->vx.cols, L->vx.rows, e->xn, d, e->V, kvd, vi, T * g->top_kv, g->top_kv, stab_for(e, &L->vx));
            else {
                const int nt = bucket(e, vi, T, g->top_kv, g->n_vexp, e->vperm, e->vtiles);
                if (nt < 0) return -1;
                kc_mm_grouped(e->st, L->vx.fmt, L->vx.d, L->vx.cols, L->vx.rows, e->xn, d, e->V, kvd, g->top_kv, e->vperm, e->vtiles, nt, stab_for(e, &L->vx));
            }
            kc_vcombine(e->st, e->V, vw, e->v, kvd, g->top_kv, T);
        }
        tgroup(e, MOVA_TG_ATTN);
        uint16_t* Kl = e->Kc + kvl * (uint64_t) l;
        uint16_t* Vl = e->Vc + kvl * (uint64_t) l;
        kc_rope_kv(e->st, e->q, e->k, e->v, Kl, Vl, e->ri, e->inv, g->n_head, g->n_kv, T);
        if (big) kc_attn_prefill(e->st, aa, e->q, Kl, Vl, e->ri, e->gq, e->ao, T);
        else kc_attn(e->st, aa, e->q, Kl, Vl, e->ri, e->part, e->gq, e->ao, T);
        tgroup(e, MOVA_TG_ATTN_PROJ);
        enc_dense(e, &L->o, e->ao, qd, e->x, d, T, 1);
        tgroup(e, MOVA_TG_EMBED_NORM);
        kc_gnorm(e->st, d, g->eps, e->x, (const uint16_t*) L->ln2.w0.p[0], e->xn, T);
        if (!L->sparse) {
            tgroup(e, MOVA_TG_SHARED);
            enc_dense(e, &L->mg, e->xn, d, e->ga, g->ff_dense, T, 0);
            enc_dense(e, &L->mu, e->xn, d, e->ua, g->ff_dense, T, 0);
            kc_swiglu(e->st, e->ga, e->ua, e->aa, T * g->ff_dense);
            enc_dense(e, &L->md, e->aa, g->ff_dense, e->x, d, T, 1);
            continue;
        }
        tgroup(e, MOVA_TG_ROUTER);
        int32_t* ri = e->inds + (size_t) l * MAX_ROWS * g->top_k;
        float* rw = e->wts + (size_t) l * MAX_ROWS * g->top_k;
        enc_router(e, &L->r, &L->rb, e->xn, g->n_exp, g->top_k, ri, rw, e->rsel + (size_t) l * MAX_ROWS * g->n_exp, T);
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
            const int nt = bucket(e, ri, T, g->top_k, g->n_exp, e->perm, e->tiles);
            if (nt < 0) return -1;
            kc_mm_grouped(e->st, L->eg.fmt, L->eg.d, L->eg.cols, L->eg.rows, e->xn, d, e->G, g->ff_exp, g->top_k, e->perm, e->tiles, nt, stab_for(e, &L->eg));
            kc_mm_grouped(e->st, L->eu.fmt, L->eu.d, L->eu.cols, L->eu.rows, e->xn, d, e->U, g->ff_exp, g->top_k, e->perm, e->tiles, nt, stab_for(e, &L->eu));
            kc_swiglu(e->st, e->G, e->U, e->A, P * g->ff_exp);
            kc_mm_grouped(e->st, L->ed.fmt, L->ed.d, L->ed.cols, L->ed.rows, e->A, g->ff_exp, e->D, d, 1, e->perm, e->tiles, nt, stab_for(e, &L->ed));
        }
        tgroup(e, MOVA_TG_SHARED);
        enc_dense(e, &L->sg, e->xn, d, e->ga, g->ff_exp, T, 0);
        enc_dense(e, &L->su, e->xn, d, e->ua, g->ff_exp, T, 0);
        kc_swiglu(e->st, e->ga, e->ua, e->aa, T * g->ff_exp);
        enc_dense(e, &L->sd, e->aa, g->ff_exp, e->sh, d, T, 0);
        tgroup(e, MOVA_TG_EXPERTS);
        kc_moe_combine(e->st, e->D, rw, e->sh, e->x, d, g->top_k, T);
    }
    if (h0 < T) {
        tgroup(e, MOVA_TG_HEAD);
        kc_gnorm(e->st, d, g->eps, e->x, (const uint16_t*) e->norm.w0.p[0], e->xn, T);
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
    if (T < 1 || T > MAX_ROWS) return -1;
    if (pos0 + T > e->kv_cap) { fprintf(stderr, "mova engine: KV capacity %lld exceeded\n", (long long) e->kv_cap); return -1; }
    memcpy(e->h_ids, tok, (size_t) T * 4);
    for (int t = 0; t < T; ++t) { memset(&e->h_ri[t], 0, sizeof e->h_ri[t]); e->h_ri[t].pos = pos0 + t; }
    if (!CK(cudaMemcpyAsync(e->ids, e->h_ids, (size_t) T * 4, cudaMemcpyHostToDevice, e->st)) ||
        !CK(cudaMemcpyAsync(e->ri, e->h_ri, (size_t) T * sizeof(RowInfo), cudaMemcpyHostToDevice, e->st)))
        return -1;
    if (encode_forward(e, T, pos0 + T, h0)) return -1;
    for (int r = h0; r < T; r += MAX_LOGIT_ROWS) {
        const int n = T - r < MAX_LOGIT_ROWS ? T - r : MAX_LOGIT_ROWS;
        encode_head(e, r, n);
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

// ---- sequences (one slot) -------------------------------------------------------------------------------------------

static void hist_push(Seq* s, int32_t t) {
    if (s->len == s->cap) { s->cap *= 2; s->hist = (int32_t*) realloc(s->hist, sizeof(int32_t) * (size_t) s->cap); }
    s->hist[s->len++] = t;
}

int eng_prefill(Eng* e, int seq, const int32_t* ids, int n) {
    if (seq != 0 || n < 1 || n > e->kv_cap) return -1;
    e->seq.len = 0;
    for (int i = 0; i < n; ++i) hist_push(&e->seq, ids[i]);
    for (int p = 0; p < n - 1; p += MAX_ROWS) {
        const int T = n - 1 - p < MAX_ROWS ? n - 1 - p : MAX_ROWS;
        if (forward(e, ids + p, T, p, T, NULL, NULL)) return -1;
    }
    return 0;
}
int eng_prefill_cached(Eng* e, int seq, const int32_t* ids, int n, int* reused) {
    if (seq != 0 || n < 1 || n > e->kv_cap) return -1;
    int c = 0;
    while (c < e->seq.len - 1 && c < n - 1 && e->seq.hist[c] == ids[c]) ++c;   // KV is valid for hist[0 .. len-2]
    e->seq.len = c;
    for (int i = c; i < n; ++i) hist_push(&e->seq, ids[i]);
    for (int p = c; p < n - 1; p += MAX_ROWS) {
        const int T = n - 1 - p < MAX_ROWS ? n - 1 - p : MAX_ROWS;
        if (forward(e, ids + p, T, p, T, NULL, NULL)) return -1;
    }
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
    for (int p = 0; p <= last; p += MAX_ROWS) {
        const int T = last + 1 - p < MAX_ROWS ? last + 1 - p : MAX_ROWS;
        int h0 = from - 1 - p;
        if (h0 < 0) h0 = 0;
        if (h0 > T) h0 = T;
        float* dst = h0 < T ? logits + (size_t) (p + h0 - (from - 1)) * e->c.vocab : NULL;
        if (forward(e, ids + p, T, p, h0, dst, NULL)) return -1;
    }
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
