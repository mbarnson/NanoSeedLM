// nslm/lib_qwen4_ref.c - the qwen4_exp CPU reference forward (nslm/qwen4_ref.h).  The whole sequence goes through one
// layer at a time, with the MLX folder's names (language_model.model.layers.L.*).
#include "qwen4_ref.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_st.h"
#include "moe.h"
#include "qwen4_ops.h"
#include "rows.h"

struct Qwen4Ref {
    Qwen4Cfg c;
    NsModel nm;
    NsRows rows;
    Qwen4Ngram ng[QWEN4_MAX_LAYERS];   // by PLE index
    char err[256];
};

static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
static float sigm(double x) { return (float) (1.0 / (1.0 + exp(-x))); }
static float silu(double x) { return (float) (x / (1.0 + exp(-x))); }
static void* zalloc(size_t n) { return calloc(n ? n : 1, 1); }

static const NsTensor* tens(Qwen4Ref* r, const char* fmt, ...) {
    char name[192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    const NsTensor* t = ns_find(&r->nm, name);
    if (!t && !r->err[0]) snprintf(r->err, sizeof r->err, "%s: missing", name);
    if (t && t->enc == NS_SEED6P8 && !r->err[0]) { snprintf(r->err, sizeof r->err, "%s: P = 8 seeds not supported", name); t = NULL; }
    return t;
}

// y[i][rows] = W[slice] x[i] for the n rows of x (stride cols)
static void lin(Qwen4Ref* r, const NsTensor* t, int slice, const float* x, int n, float* y) {
    if (!t) return;
    float* w = (float*) malloc(sizeof(float) * (size_t) t->cols);
    for (int o = 0; o < t->rows; ++o) {
        ns_row_f32(&r->rows, t, slice, o, 0, w);
        for (int i = 0; i < n; ++i) {
            const float* xi = x + (size_t) i * t->cols;
            double acc = 0;
            for (int c = 0; c < t->cols; ++c) acc += (double) w[c] * xi[c];
            y[(size_t) i * t->rows + o] = (float) acc;
        }
    }
    free(w);
}

// A BF16 vector of n values (any shape), or zeros and an error
static float* vec(Qwen4Ref* r, const NsTensor* t, int64_t n) {
    float* v = (float*) zalloc(sizeof(float) * (size_t) n);
    if (!t) return v;
    if (t->enc != NS_BF16 || (int64_t) t->slices * t->rows * t->cols != n) {
        if (!r->err[0]) snprintf(r->err, sizeof r->err, "%s: not BF16 with %lld values", t->name, (long long) n);
        return v;
    }
    const uint16_t* p = (const uint16_t*) t->s[0].p;
    for (int64_t i = 0; i < n; ++i) v[i] = bf2f(p[i]);
    return v;
}

static void norm1p(const float* x, int n, int len, int g, const float* w, float eps, float* y) {
    for (int i = 0; i < n; ++i) qwen4_norm1p(x + (size_t) i * len, len, g, w, eps, y + (size_t) i * len);
}

// ---- hyper-connections: mix the hc streams of H[n][hc * d] into one input, and the weights that add a branch back ----

static void hc_mix(Qwen4Ref* r, const char* pre, const float* H, int n, float* mixed, float* inj) {
    const Qwen4Cfg* c = &r->c;
    const int HD = c->hc * c->d;
    float* nw = vec(r, tens(r, "%s.hc_norm.weight", pre), HD);
    float* normed = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* lo = (float*) zalloc(sizeof(float) * (size_t) n * c->hc_rank);
    float* up = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    norm1p(H, n, HD, c->d, nw, c->eps, normed);
    lin(r, tens(r, "%s.input_mix_weight_down.weight", pre), 0, normed, n, lo);
    for (int t = 0; t < n; ++t) qwen4_hc_act(c, lo + (size_t) t * c->hc_rank);
    lin(r, tens(r, "%s.input_mix_weight_up.weight", pre), 0, lo, n, up);
    for (int t = 0; t < n; ++t) qwen4_hc_mix(c, normed + (size_t) t * HD, up + (size_t) t * HD, mixed + (size_t) t * c->d);
    if (inj) {
        lin(r, tens(r, "%s.block_inject_weight.weight", pre), 0, normed, n, inj);
        for (int t = 0; t < n; ++t) qwen4_hc_gate(c, inj + (size_t) t * c->hc);
    }
    free(nw); free(normed); free(lo); free(up);
}

static void hc_add(const Qwen4Cfg* c, float* H, int n, const float* branch, const float* inj) {
    for (int t = 0; t < n; ++t) qwen4_hc_add(c, H + (size_t) t * c->hc * c->d, branch + (size_t) t * c->d, inj + (size_t) t * c->hc);
}

// ---- Gated DeltaNet (nslm/qwen4_ops.h): the projections, the token steps, the output projection --------------------

static void gdn(Qwen4Ref* r, int l, const float* x, int n, float* out) {
    const Qwen4Cfg* c = &r->c;
    const int Hv = c->lin_hv, K = c->lin_conv, vd = Hv * c->lin_dv, cd = 2 * c->lin_hk * c->lin_dk + vd;
    char pre[96];
    snprintf(pre, sizeof pre, "language_model.model.layers.%d.linear_attn", l);
    float* qkv = (float*) zalloc(sizeof(float) * (size_t) n * cd);
    float* z = (float*) zalloc(sizeof(float) * (size_t) n * vd);
    float* bb = (float*) zalloc(sizeof(float) * (size_t) n * Hv);
    float* aa = (float*) zalloc(sizeof(float) * (size_t) n * Hv);
    float* y = (float*) zalloc(sizeof(float) * (size_t) n * vd);
    lin(r, tens(r, "%s.in_proj_qkv.weight", pre), 0, x, n, qkv);
    lin(r, tens(r, "%s.in_proj_z.weight", pre), 0, x, n, z);
    lin(r, tens(r, "%s.in_proj_b.weight", pre), 0, x, n, bb);
    lin(r, tens(r, "%s.in_proj_a.weight", pre), 0, x, n, aa);
    float* cw = vec(r, tens(r, "%s.conv1d.weight", pre), (int64_t) cd * K);
    float* alog = vec(r, tens(r, "%s.A_log", pre), Hv);
    float* dtb = vec(r, tens(r, "%s.dt_bias", pre), Hv);
    float* nw = vec(r, tens(r, "%s.norm.weight", pre), c->lin_dv);
    float* conv = (float*) zalloc(sizeof(float) * qwen4_gdn_conv_len(c));
    float* S = (float*) zalloc(sizeof(float) * qwen4_gdn_state_len(c));
    const Qwen4GdnW gw = {cw, alog, dtb, nw};
    for (int t = 0; t < n; ++t)
        qwen4_gdn_token(c, &gw, qkv + (size_t) t * cd, z + (size_t) t * vd, aa + (size_t) t * Hv, bb + (size_t) t * Hv, conv, S, y + (size_t) t * vd);
    lin(r, tens(r, "%s.out_proj.weight", pre), 0, y, n, out);
    free(qkv); free(z); free(bb); free(aa); free(y); free(cw); free(alog); free(dtb); free(nw); free(S); free(conv);
}

// ---- QSA attention: dense within the budget; past it, each query sees the indexer's top blocks and its own tail -------

static void attention(Qwen4Ref* r, const char* lp, const float* x, int n, float* out) {   // lp: the layer's name
    const Qwen4Cfg* c = &r->c;
    const int Hq = c->n_head, Hkv = c->n_kv, hd = c->head_dim, D = c->ix_dim, R = c->ix_ratio, iw = (c->ix_heads + 1) * D;
    char pre[112];
    snprintf(pre, sizeof pre, "%s.self_attn", lp);
    float* qp = (float*) zalloc(sizeof(float) * (size_t) n * Hq * 2 * hd);
    float* kp = (float*) zalloc(sizeof(float) * (size_t) n * Hkv * hd);
    float* vp = (float*) zalloc(sizeof(float) * (size_t) n * Hkv * hd);
    float* o = (float*) zalloc(sizeof(float) * (size_t) n * Hq * hd);
    float* ix = (float*) zalloc(sizeof(float) * (size_t) n * iw);
    float* raw = (float*) zalloc(sizeof(float) * (size_t) R * D);
    float* blk = (float*) zalloc(sizeof(float) * (size_t) (n / R + 1) * D);
    int32_t* sel = (int32_t*) zalloc(sizeof(int32_t) * (size_t) (c->ix_budget / R));
    lin(r, tens(r, "%s.q_proj.weight", pre), 0, x, n, qp);
    lin(r, tens(r, "%s.k_proj.weight", pre), 0, x, n, kp);
    lin(r, tens(r, "%s.v_proj.weight", pre), 0, x, n, vp);
    lin(r, tens(r, "%s.indexer.index_qk_proj.weight", pre), 0, x, n, ix);
    float* qn = vec(r, tens(r, "%s.q_norm.weight", pre), hd);
    float* kn = vec(r, tens(r, "%s.k_norm.weight", pre), hd);
    float* iqn = vec(r, tens(r, "%s.indexer.q_layernorm.weight", pre), D);
    float* ikn = vec(r, tens(r, "%s.indexer.k_layernorm.weight", pre), D);
    for (int t = 0; t < n; ++t) {   // q_proj: per head hd query values, then hd gate values
        qwen4_head_norm_rope(c, qp + (size_t) t * Hq * 2 * hd, Hq, 2 * hd, hd, qn, t);
        qwen4_head_norm_rope(c, kp + (size_t) t * Hkv * hd, Hkv, hd, hd, kn, t);
    }
    for (int j = 0; j < n / R; ++j) {   // block keys from the raw indexer keys (after the query heads)
        for (int u = 0; u < R; ++u) memcpy(raw + (size_t) u * D, ix + (size_t) (j * R + u) * iw + c->ix_heads * D, sizeof(float) * (size_t) D);
        qwen4_qsa_block(c, raw, ikn, j, blk + (size_t) j * D);
    }
    for (int t = 0; t < n; ++t) {
        float* iq = ix + (size_t) t * iw;
        qwen4_head_norm_rope(c, iq, c->ix_heads, D, D, iqn, t);
        const int nsel = qwen4_qsa_select(c, iq, blk, (t + 1) / R, sel);
        qwen4_attend(c, qp + (size_t) t * Hq * 2 * hd, kp, vp, t, sel, nsel, o + (size_t) t * Hq * hd);
    }
    lin(r, tens(r, "%s.o_proj.weight", pre), 0, o, n, out);
    free(qp); free(kp); free(vp); free(o); free(ix); free(raw); free(blk); free(sel); free(qn); free(kn); free(iqn); free(ikn);
}

// ---- MoE: softmax router, top-k renormalised, silu experts; a shared expert scaled by sigmoid(gate) ------------------

static void glu(Qwen4Ref* r, const char* pre, int slice, int ff, const float* x, float* y) {
    float* g = (float*) zalloc(sizeof(float) * (size_t) ff);
    float* u = (float*) zalloc(sizeof(float) * (size_t) ff);
    lin(r, tens(r, "%s.gate_proj.weight", pre), slice, x, 1, g);
    lin(r, tens(r, "%s.up_proj.weight", pre), slice, x, 1, u);
    for (int i = 0; i < ff; ++i) g[i] = silu(g[i]) * u[i];
    lin(r, tens(r, "%s.down_proj.weight", pre), slice, g, 1, y);
    free(g); free(u);
}

static void moe(Qwen4Ref* r, const char* lp, const float* x, int n, float* out) {
    const Qwen4Cfg* c = &r->c;
    const int E = c->n_exp, D = c->d;
    char pre[112], sw[128], sh[128];
    snprintf(pre, sizeof pre, "%s.mlp", lp);
    snprintf(sw, sizeof sw, "%s.switch_mlp", pre);
    snprintf(sh, sizeof sh, "%s.shared_expert", pre);
    float* lg = (float*) zalloc(sizeof(float) * (size_t) n * E);
    float* y = (float*) zalloc(sizeof(float) * (size_t) D);
    int32_t* ind = (int32_t*) zalloc(sizeof(int32_t) * (size_t) c->top_k);
    float* wt = (float*) zalloc(sizeof(float) * (size_t) c->top_k);
    lin(r, tens(r, "%s.gate.weight", pre), 0, x, n, lg);
    for (int t = 0; t < n; ++t) {
        const float* xt = x + (size_t) t * D;
        float* ot = out + (size_t) t * D;
        qwen4_route(c, lg + (size_t) t * E, ind, wt);
        memset(ot, 0, sizeof(float) * (size_t) D);
        for (int k = 0; k < c->top_k; ++k) {
            glu(r, sw, ind[k], c->ff_exp, xt, y);
            for (int i = 0; i < D; ++i) ot[i] += wt[k] * y[i];
        }
        float sg;
        lin(r, tens(r, "%s.shared_expert_gate.weight", pre), 0, xt, 1, &sg);
        glu(r, sh, 0, c->ff_shared, xt, y);
        for (int i = 0; i < D; ++i) ot[i] += sigm(sg) * y[i];
    }
    free(lg); free(y); free(ind); free(wt);
}

// ---- PLE: n-gram rows -> keys gating a value per stream, plus a dilated causal conv of the gated values --------------

static void ple(Qwen4Ref* r, int l, const int32_t* ids, float* H, int n) {
    const Qwen4Cfg* c = &r->c;
    const int D = c->d, HD = c->hc * D, P = c->ple_dim;
    char pre[96];
    snprintf(pre, sizeof pre, "language_model.model.layers.%d.ple", l);
    float* emb = (float*) zalloc(sizeof(float) * (size_t) n * P);
    for (int t = 0; t < n && !r->err[0]; ++t)
        if (qwen4_ple_embed(&r->nm, &r->rows, NULL, c, &r->ng[c->ple[l]], l, ids, t, emb + (size_t) t * P))
            snprintf(r->err, sizeof r->err, "%s: a PLE table shard is missing or does not hold a row", pre);
    float* keys = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* qs = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* vals = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* gated = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* hist = (float*) zalloc(sizeof(float) * qwen4_ple_hist_len(c));
    float* nk = vec(r, tens(r, "%s.norm_key.weight", pre), HD);
    float* nq = vec(r, tens(r, "%s.norm_query.weight", pre), HD);
    float* nc = vec(r, tens(r, "%s.norm_conv.weight", pre), HD);
    float* cw = vec(r, tens(r, "%s.conv1d.weight", pre), (int64_t) HD * c->ple_conv);
    lin(r, tens(r, "%s.key_proj.weight", pre), 0, emb, n, keys);
    lin(r, tens(r, "%s.value_proj.weight", pre), 0, emb, n, vals);
    norm1p(keys, n, HD, D, nk, c->eps, keys);
    norm1p(H, n, HD, D, nq, c->eps, qs);
    for (int t = 0; t < n; ++t) qwen4_ple_gate(c, keys + (size_t) t * HD, qs + (size_t) t * HD, vals + (size_t) t * D, gated + (size_t) t * HD);
    norm1p(gated, n, HD, D, nc, c->eps, keys);   // keys now holds the conv input
    for (int t = 0; t < n; ++t) qwen4_ple_conv(c, cw, keys + (size_t) t * HD, gated + (size_t) t * HD, hist, H + (size_t) t * HD);
    free(emb); free(keys); free(qs); free(vals); free(gated); free(hist); free(nk); free(nq); free(nc); free(cw);
}

// ---- the model -------------------------------------------------------------------------------------------------------

Qwen4Ref* qwen4_ref_open(const char* model_dir, char* err, int errlen) {
    Qwen4Ref* r = (Qwen4Ref*) calloc(1, sizeof *r);
    if (qwen4_cfg_load(&r->c, model_dir, err, errlen) || ns_open(&r->nm, model_dir, err, errlen)) { free(r); return NULL; }
    if (ns_add_banks(&r->nm, model_dir, err, errlen)) { ns_close(&r->nm); free(r); return NULL; }   // trellis experts (.blkt)
    ns_rows_init(&r->rows);
    for (int l = 0; l < r->c.n_layer; ++l) if (r->c.ple[l] >= 0) qwen4_ngram_init(&r->c, r->c.ple[l], &r->ng[r->c.ple[l]]);
    return r;
}
void qwen4_ref_close(Qwen4Ref* r) {
    if (!r) return;
    ns_close(&r->nm);
    free(r);
}
const Qwen4Cfg* qwen4_ref_cfg(const Qwen4Ref* r) { return &r->c; }
const char* qwen4_ref_error(const Qwen4Ref* r) { return r->err; }

// One decoder layer named lp on the streams H: the attention (or DeltaNet layer l) and the MoE, each through its HC
static void decoder(Qwen4Ref* r, const char* lp, int attn, int l, float* H, int n, float* x, float* b, float* inj) {
    char pre[128];
    snprintf(pre, sizeof pre, "%s.attn_hyper_connection", lp);
    hc_mix(r, pre, H, n, x, inj);
    if (attn) attention(r, lp, x, n, b);
    else gdn(r, l, x, n, b);
    hc_add(&r->c, H, n, b, inj);
    snprintf(pre, sizeof pre, "%s.mlp_hyper_connection", lp);
    hc_mix(r, pre, H, n, x, inj);
    moe(r, lp, x, n, b);
    hc_add(&r->c, H, n, b, inj);
}
static int embed(Qwen4Ref* r, const int32_t* ids, int n, float* x) {
    const NsTensor* E = tens(r, "language_model.model.embed_tokens.weight");
    for (int t = 0; E && t < n; ++t) {
        if (ids[t] < 0 || ids[t] >= E->rows) { snprintf(r->err, sizeof r->err, "token %d out of range", ids[t]); return -1; }
        ns_row_f32(&r->rows, E, 0, ids[t], 0, x + (size_t) t * r->c.d);
    }
    return E ? 0 : -1;
}

int qwen4_ref_forward_h(Qwen4Ref* r, const int32_t* ids, int n, float* logits, float* Hout) {
    const Qwen4Cfg* c = &r->c;
    const int D = c->d, HD = c->hc * D;
    r->err[0] = 0;
    float* H = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* x = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* b = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* inj = (float*) zalloc(sizeof(float) * (size_t) n * c->hc);
    if (!embed(r, ids, n, x))
        for (int t = 0; t < n; ++t)
            for (int s = 0; s < c->hc; ++s) memcpy(H + (size_t) t * HD + (size_t) s * D, x + (size_t) t * D, sizeof(float) * (size_t) D);
    char lp[64];
    for (int l = 0; l < c->n_layer && !r->err[0]; ++l) {
        if (c->ple[l] >= 0) ple(r, l, ids, H, n);
        snprintf(lp, sizeof lp, "language_model.model.layers.%d", l);
        decoder(r, lp, c->attn[l], l, H, n, x, b, inj);
    }
    if (!r->err[0]) {
        if (Hout) memcpy(Hout, H, sizeof(float) * (size_t) n * HD);
        hc_mix(r, "language_model.model.hyper_connection_mixer", H, n, x, NULL);
        lin(r, tens(r, "language_model.lm_head.weight"), 0, x, n, logits);
    }
    free(H); free(x); free(b); free(inj);
    return r->err[0] ? -1 : 0;
}
int qwen4_ref_forward(Qwen4Ref* r, const int32_t* ids, int n, float* logits) { return qwen4_ref_forward_h(r, ids, n, logits, NULL); }

int qwen4_ref_mtp(Qwen4Ref* r, const int32_t* tok, const float* Hin, int n, float* logits, float* Hout) {
    const Qwen4Cfg* c = &r->c;
    const int D = c->d, HD = c->hc * D;
    r->err[0] = 0;
    float* H = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* x = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* b = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* inj = (float*) zalloc(sizeof(float) * (size_t) n * c->hc);
    float* we = vec(r, tens(r, "mtp.pre_fc_norm_embedding.weight"), D);
    float* wh = vec(r, tens(r, "mtp.pre_fc_norm_hidden.weight"), HD);
    if (!embed(r, tok, n, x) && we && wh) {   // fc_embedding(norm(embed)) + fc_hidden(each stream of norm(H)): the layer's streams
        norm1p(x, n, D, D, we, c->eps, x);
        lin(r, tens(r, "mtp.fc_embedding.weight"), 0, x, n, b);
        norm1p(Hin, n, HD, HD, wh, c->eps, H);
        float* hp = (float*) zalloc(sizeof(float) * (size_t) n * HD);
        lin(r, tens(r, "mtp.fc_hidden.weight"), 0, H, n * c->hc, hp);
        for (int t = 0; t < n; ++t)
            for (int s = 0; s < c->hc; ++s)
                for (int i = 0; i < D; ++i) H[(size_t) t * HD + (size_t) s * D + i] = b[(size_t) t * D + i] + hp[(size_t) t * HD + (size_t) s * D + i];
        free(hp);
        decoder(r, "mtp.layers.0", 1, -1, H, n, x, b, inj);
    }
    if (!r->err[0]) {
        if (Hout) memcpy(Hout, H, sizeof(float) * (size_t) n * HD);
        hc_mix(r, "mtp.hyper_connection_mixer", H, n, x, NULL);
        lin(r, tens(r, "language_model.lm_head.weight"), 0, x, n, logits);
    }
    free(H); free(x); free(b); free(inj); free(we); free(wh);
    return r->err[0] ? -1 : 0;
}
