// engine/qwen4_cuda.c - the Qwen3.8 CUDA engine (engine/qwen4_eng.h): the steps of nslm/qwen4_ref.h on the matrix
// kernels of engine/kernels_cuda.h and the Qwen3.8 kernels of engine/kernels_qwen4.h.
#include "qwen4_eng.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "kernels_cuda.h"
#include "kernels_qwen4.h"
#include "lfsr.h"
#include "model_st.h"
#include "qwen4_ops.h"
#include "rows.h"

#define MAXT Q4E_MAX_ROWS
#if Q4E_MAX_ROWS != MV_MAXT
#error Q4E_MAX_ROWS must equal MV_MAXT
#endif

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
    int max_pos, pos, host_experts;
    int32_t* ids;   // the context (host), for the PLE n-grams
    float *x, *H, *normed, *lo, *up, *inj, *br, *qp, *kk, *vv, *o, *ixo, *sc, *emb, *emb_h, *keys, *qs, *vals, *gated;
    float *lg, *wt, *A, *Dx, *sh, *sgv, *logits, *y, *cv, *qkvb, *zb, *bb, *ab;
    int32_t *d_ids, *ind, *sel, *done, *nsel, *zero;
    void** dev;   // allocations, freed at close
    int ndev, cdev;
    void** host;
    int nhost, chost;
    char err[256];
};

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
static void mv(Q4Eng* e, const W* w, const float* x, float* y, int T) {
    kc_mv(e->s, w->fmt, w->w, w->cols, w->rows, x, w->cols, y, w->rows, T, 0, G(e, w));
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
    load_w(e, &L->eg, e->host_experts, "%s.%d.mlp.switch_mlp.gate_proj.weight", p, l);
    load_w(e, &L->eu, e->host_experts, "%s.%d.mlp.switch_mlp.up_proj.weight", p, l);
    load_w(e, &L->ed, e->host_experts, "%s.%d.mlp.switch_mlp.down_proj.weight", p, l);
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

Q4Eng* q4e_open(const char* model_dir, const Q4EOpts* o, char* err, int errlen) {
    Q4Eng* e = (Q4Eng*) calloc(1, sizeof *e);
    if (qwen4_cfg_load(&e->c, model_dir, err, errlen) || ns_open(&e->nm, model_dir, err, errlen)) { free(e); return NULL; }
    const Qwen4Cfg* c = &e->c;
    e->max_pos = o && o->max_pos > 0 ? o->max_pos : 4096;
    e->host_experts = !(o && o->vram_experts);
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
    const size_t T = MAXT, HD = (size_t) c->hc * c->d, cd = (size_t) 2 * c->lin_hk * c->lin_dk + (size_t) c->lin_hv * c->lin_dv;
    const size_t vd = (size_t) c->lin_hv * c->lin_dv, iw = (size_t) (c->ix_heads + 1) * c->ix_dim, kf = (size_t) c->top_k * c->ff_exp;
    const size_t nb = (size_t) e->max_pos / c->ix_ratio + 1;
#define F(n) (float*) dmem(e, NULL, sizeof(float) * (n))
#define I(n) (int32_t*) dmem(e, NULL, sizeof(int32_t) * (n))
    e->x = F(T * c->d); e->H = F(T * HD); e->normed = F(T * HD); e->lo = F(T * c->hc_rank); e->up = F(T * HD); e->inj = F(T * c->hc);
    e->br = F(T * c->d); e->qp = F(T * c->n_head * 2 * c->head_dim); e->kk = F(T * c->n_kv * c->head_dim);
    e->vv = F(T * c->n_kv * c->head_dim); e->o = F(T * c->n_head * c->head_dim); e->ixo = F(T * iw); e->sc = F(T * nb);
    e->emb = F(T * c->ple_dim); e->keys = F(T * HD); e->qs = F(T * HD); e->vals = F(T * c->d); e->gated = F(T * HD);
    e->lg = F(T * c->n_exp); e->wt = F(T * c->top_k); e->A = F(T * (kf > (size_t) c->ff_shared ? kf : (size_t) c->ff_shared));
    e->Dx = F(T * c->top_k * c->d); e->sh = F(T * c->d); e->sgv = F(T); e->logits = F(T * c->vocab); e->y = F(T * vd);
    e->cv = F(T * cd); e->qkvb = F(T * cd); e->zb = F(T * vd); e->bb = F(T * c->lin_hv); e->ab = F(T * c->lin_hv);
    e->d_ids = I(T); e->ind = I(T * c->top_k); e->sel = I(T * (c->ix_budget / c->ix_ratio)); e->done = I(T); e->nsel = I(T); e->zero = I(T);
#undef F
#undef I
    e->emb_h = (float*) malloc(sizeof(float) * T * (size_t) c->ple_dim);
    e->ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) e->max_pos);
    if (!e->err[0] && cudaDeviceSynchronize() != cudaSuccess) fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    if (e->err[0]) { snprintf(err, (size_t) errlen, "%s", e->err); q4e_close(e); return NULL; }
    return e;
}

void q4e_close(Q4Eng* e) {
    if (!e) return;
    cudaDeviceSynchronize();
    for (int i = 0; i < e->ndev; ++i) cudaFree(e->dev[i]);
    for (int i = 0; i < e->nhost; ++i) cudaFreeHost(e->host[i]);
    free(e->dev); free(e->host); free(e->emb_h); free(e->ids);
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

static int ple(Q4Eng* e, int l, int T) {
    const Qwen4Cfg* c = &e->c;
    Layer* L = &e->L[l];
    const int HD = c->hc * c->d;
    for (int t = 0; t < T; ++t)
        if (qwen4_ple_embed(&e->nm, &e->rows, e->table, c, &e->ng[c->ple[l]], l, e->ids, e->pos + t, e->emb_h + (size_t) t * c->ple_dim))
            return fail(e, "layer %d: a PLE table shard is missing or does not hold a row", l);
    cudaMemcpyAsync(e->emb, e->emb_h, sizeof(float) * (size_t) T * c->ple_dim, cudaMemcpyHostToDevice, e->s);
    mv(e, &L->key, e->emb, e->keys, T);
    kq4_norm1p(e->s, e->keys, HD, c->d, L->nk, c->eps, e->keys, T);
    mv(e, &L->val, e->emb, e->vals, T);
    kq4_norm1p(e->s, e->H, HD, c->d, L->nq, c->eps, e->qs, T);
    kq4_ple_gate(e->s, c, e->keys, e->qs, e->vals, e->gated, T);
    kq4_norm1p(e->s, e->gated, HD, c->d, L->nc, c->eps, e->keys, T);   // keys now holds the conv input
    kq4_ple_conv(e->s, c, L->pw, e->keys, e->gated, L->hist, e->H, T);
    return 0;
}

static int attention(Q4Eng* e, Layer* L, int T) {
    const Qwen4Cfg* c = &e->c;
    const int hd = c->head_dim, kvw = c->n_kv * hd, D = c->ix_dim, R = c->ix_ratio, iw = (c->ix_heads + 1) * D, topk = c->ix_budget / R;
    mv(e, &L->q, e->x, e->qp, T);
    mv(e, &L->k, e->x, e->kk, T);
    mv(e, &L->v, e->x, e->vv, T);
    kq4_head_norm_rope(e->s, c, e->qp, c->n_head * 2 * hd, c->n_head, 2 * hd, hd, L->qn, e->pos, T);
    kq4_head_norm_rope(e->s, c, e->kk, kvw, c->n_kv, hd, hd, L->kn, e->pos, T);
    cudaMemcpyAsync(L->K + (size_t) e->pos * kvw, e->kk, sizeof(float) * (size_t) T * kvw, cudaMemcpyDeviceToDevice, e->s);
    cudaMemcpyAsync(L->V + (size_t) e->pos * kvw, e->vv, sizeof(float) * (size_t) T * kvw, cudaMemcpyDeviceToDevice, e->s);
    mv(e, &L->ix, e->x, e->ixo, T);
    cudaMemcpy2DAsync(L->raw + (size_t) e->pos * D, sizeof(float) * D, e->ixo + (size_t) c->ix_heads * D, sizeof(float) * (size_t) iw,
                      sizeof(float) * D, (size_t) T, cudaMemcpyDeviceToDevice, e->s);
    kq4_head_norm_rope(e->s, c, e->ixo, iw, c->ix_heads, D, D, L->iqn, e->pos, T);
    kq4_qsa_blocks(e->s, c, L->raw, L->ikn, e->pos / R, (e->pos + T) / R, L->blk);
    int32_t done[MAXT], nsel[MAXT], maxdone = 0;
    for (int t = 0; t < T; ++t) {
        done[t] = (e->pos + t + 1) / R;
        nsel[t] = done[t] > topk ? topk : 0;
        if (done[t] > maxdone) maxdone = done[t];
    }
    cudaMemcpyAsync(e->done, done, sizeof(int32_t) * (size_t) T, cudaMemcpyHostToDevice, e->s);
    cudaMemcpyAsync(e->nsel, nsel, sizeof(int32_t) * (size_t) T, cudaMemcpyHostToDevice, e->s);
    kq4_qsa_select(e->s, c, e->ixo, iw, L->blk, e->done, maxdone, e->sc, e->sel, T);
    if (kq4_attend(e->s, c, e->qp, L->K, L->V, e->pos, e->sel, e->nsel, e->o, T)) return fail(e, "head_dim %d not supported", hd);
    mv(e, &L->o, e->o, e->br, T);
    return 0;
}

static int gdn(Q4Eng* e, Layer* L, int T) {
    mv(e, &L->qkv, e->x, e->qkvb, T);
    mv(e, &L->z, e->x, e->zb, T);
    mv(e, &L->b, e->x, e->bb, T);
    mv(e, &L->a, e->x, e->ab, T);
    if (kq4_gdn(e->s, &e->c, L->gw, e->qkvb, e->zb, e->ab, e->bb, L->conv, L->S, e->cv, e->y, T)) return fail(e, "Gated DeltaNet head widths not supported");
    mv(e, &L->out, e->y, e->br, T);
    return 0;
}

static void moe(Q4Eng* e, Layer* L, int T) {
    const Qwen4Cfg* c = &e->c;
    const int k = c->top_k, P = T * k;
    mv(e, &L->gate, e->x, e->lg, T);
    kq4_route(e->s, c, e->lg, e->ind, e->wt, T);
    kc_mv_gu(e->s, L->eg.fmt, L->eg.tab, L->eu.tab, c->d, c->ff_exp, e->x, c->d, e->A, c->ff_exp, e->ind, P, k, G(e, &L->eg));
    kc_mv_sel(e->s, L->ed.fmt, L->ed.tab, c->ff_exp, c->d, e->A, c->ff_exp, e->Dx, c->d, e->ind, P, 1, G(e, &L->ed));
    kc_mv_gu(e->s, L->sh_g.fmt, L->sh_g.tab, L->sh_u.tab, c->d, c->ff_shared, e->x, c->d, e->A, c->ff_shared, e->zero, T, 1, G(e, &L->sh_g));
    mv(e, &L->sh_d, e->A, e->sh, T);
    mv(e, &L->sg, e->x, e->sgv, T);
    kq4_moe_combine(e->s, c->d, k, e->Dx, e->wt, e->sh, e->sgv, e->br, T);
}

static int forward_rows(Q4Eng* e, const int32_t* ids, int T, float* logits) {
    const Qwen4Cfg* c = &e->c;
    memcpy(e->ids + e->pos, ids, sizeof(int32_t) * (size_t) T);
    cudaMemcpyAsync(e->d_ids, ids, sizeof(int32_t) * (size_t) T, cudaMemcpyHostToDevice, e->s);
    kc_embed(e->s, e->embed.fmt, e->embed.w, c->d, e->d_ids, e->x, T);
    kq4_tile(e->s, c->hc, c->d, e->x, e->H, T);
    for (int l = 0; l < c->n_layer; ++l) {
        Layer* L = &e->L[l];
        if (c->ple[l] >= 0 && ple(e, l, T)) return -1;
        hc_mix(e, L->hc_norm[0], &L->hc_down[0], &L->hc_up[0], &L->hc_inj[0], T);
        if (c->attn[l] ? attention(e, L, T) : gdn(e, L, T)) return -1;
        kq4_hc_add(e->s, c->hc, c->d, e->H, e->br, e->inj, T);
        hc_mix(e, L->hc_norm[1], &L->hc_down[1], &L->hc_up[1], &L->hc_inj[1], T);
        moe(e, L, T);
        kq4_hc_add(e->s, c->hc, c->d, e->H, e->br, e->inj, T);
    }
    hc_mix(e, e->mix_norm, &e->mix_down, &e->mix_up, NULL, T);
    mv(e, &e->head, e->x, e->logits, T);
    if (logits) cudaMemcpyAsync(logits, e->logits, sizeof(float) * (size_t) T * c->vocab, cudaMemcpyDeviceToHost, e->s);
    if (cudaStreamSynchronize(e->s) != cudaSuccess || cudaGetLastError() != cudaSuccess)
        return fail(e, "CUDA: %s", cudaGetErrorString(cudaGetLastError()));
    e->pos += T;
    return 0;
}

int q4e_forward(Q4Eng* e, const int32_t* ids, int n, float* logits) {
    e->err[0] = 0;
    if (e->pos + n > e->max_pos) return fail(e, "the context is full (%d positions)", e->max_pos);
    for (int i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= e->embed.rows) return fail(e, "token %d out of range", ids[i]);
    for (int i = 0; i < n; i += MAXT) {
        const int T = n - i < MAXT ? n - i : MAXT;
        if (forward_rows(e, ids + i, T, logits ? logits + (size_t) i * e->c.vocab : NULL)) return -1;
    }
    return 0;
}
