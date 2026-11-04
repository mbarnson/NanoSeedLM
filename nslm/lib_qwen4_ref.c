// nslm/lib_qwen4_ref.c - the qwen4_exp CPU reference forward (nslm/qwen4_ref.h).  The whole sequence goes through one
// layer at a time, with the MLX folder's names (language_model.model.layers.L.*).
#include "qwen4_ref.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_st.h"
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

// Qwen4 RMSNorm (weights centred at 0): each group of g values of x[n][len] * rsqrt(mean square + eps) * (1 + w)
static void norm1p(const float* x, int n, int len, int g, const float* w, float eps, float* y) {
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < len; a += g) {
            const float* xi = x + (size_t) i * len + a;
            double ss = 0;
            for (int k = 0; k < g; ++k) ss += (double) xi[k] * xi[k];
            const double s = 1.0 / sqrt(ss / g + eps);
            for (int k = 0; k < g; ++k) y[(size_t) i * len + a + k] = (float) (xi[k] * s * (1.0 + w[a + k]));
        }
}

// RoPE on the first rot values of x (halves rotated against each other), position pos
static void rope(float* x, int rot, float theta, int pos) {
    for (int i = 0; i < rot / 2; ++i) {
        const float inv = (float) pow((double) theta, -2.0 * i / rot), a = (float) pos * inv;   // f32 angles, as MLX
        const double cs = cos(a), sn = sin(a), x1 = x[i], x2 = x[i + rot / 2];
        x[i] = (float) (x1 * cs - x2 * sn);
        x[i + rot / 2] = (float) (x2 * cs + x1 * sn);
    }
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
    for (int i = 0; i < n * c->hc_rank; ++i) lo[i] = silu(lo[i] / c->hc);
    lin(r, tens(r, "%s.input_mix_weight_up.weight", pre), 0, lo, n, up);
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < c->d; ++i) {
            double m = 0;
            for (int s = 0; s < c->hc; ++s) m += (double) sigm(up[(size_t) t * HD + s * c->d + i]) * normed[(size_t) t * HD + s * c->d + i];
            mixed[(size_t) t * c->d + i] = (float) (m / c->hc);
        }
    if (inj) {
        lin(r, tens(r, "%s.block_inject_weight.weight", pre), 0, normed, n, inj);
        for (int i = 0; i < n * c->hc; ++i) inj[i] = 2.0f * sigm(inj[i] / c->hc);
    }
    free(nw); free(normed); free(lo); free(up);
}

static void hc_add(const Qwen4Cfg* c, float* H, int n, const float* branch, const float* inj) {
    for (int t = 0; t < n; ++t)
        for (int s = 0; s < c->hc; ++s)
            for (int i = 0; i < c->d; ++i) H[((size_t) t * c->hc + s) * c->d + i] += branch[(size_t) t * c->d + i] * inj[t * c->hc + s];
}

// ---- Gated DeltaNet: a causal conv, then per value head a [dv][dk] state decayed by g and corrected toward v (delta rule)

static void gdn(Qwen4Ref* r, int l, const float* x, int n, float* out) {
    const Qwen4Cfg* c = &r->c;
    const int Hk = c->lin_hk, Hv = c->lin_hv, Dk = c->lin_dk, Dv = c->lin_dv, K = c->lin_conv;
    const int kd = Hk * Dk, vd = Hv * Dv, cd = 2 * kd + vd, rep = Hv / Hk;
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
    float* nw = vec(r, tens(r, "%s.norm.weight", pre), Dv);
    double* S = (double*) zalloc(sizeof(double) * (size_t) Hv * Dv * Dk);
    float* cv = (float*) zalloc(sizeof(float) * (size_t) cd);
    for (int t = 0; t < n; ++t) {
        for (int ch = 0; ch < cd; ++ch) {   // depthwise causal conv over the last K inputs, then silu
            double acc = 0;
            for (int k = 0; k < K; ++k) {
                const int u = t - (K - 1) + k;
                if (u >= 0) acc += (double) cw[(size_t) ch * K + k] * qkv[(size_t) u * cd + ch];
            }
            cv[ch] = silu(acc);
        }
        for (int h = 0; h < 2 * Hk; ++h) {   // L2-normalise q and k heads; q also scaled by dk^-0.5
            float* v = cv + (size_t) h * Dk;
            double ss = 0;
            for (int j = 0; j < Dk; ++j) ss += (double) v[j] * v[j];
            const double s = 1.0 / sqrt(ss + 1e-6) * (h < Hk ? 1.0 / sqrt((double) Dk) : 1.0);
            for (int j = 0; j < Dk; ++j) v[j] = (float) (v[j] * s);
        }
        for (int hv = 0; hv < Hv; ++hv) {
            const float* q = cv + (size_t) (hv / rep) * Dk;
            const float* k = cv + kd + (size_t) (hv / rep) * Dk;
            const float* v = cv + 2 * kd + (size_t) hv * Dv;
            const double ap = aa[t * Hv + hv] + dtb[hv], sp = ap > 20 ? ap : log1p(exp(ap));
            const double g = exp(-exp((double) alog[hv]) * sp), beta = sigm(bb[t * Hv + hv]);
            double* St = S + (size_t) hv * Dv * Dk;
            float* yo = y + (size_t) t * vd + (size_t) hv * Dv;
            double ss = 0;
            for (int i = 0; i < Dv; ++i) {
                double* row = St + (size_t) i * Dk, kv = 0, yq = 0;
                for (int j = 0; j < Dk; ++j) { row[j] *= g; kv += row[j] * k[j]; }
                const double delta = (v[i] - kv) * beta;
                for (int j = 0; j < Dk; ++j) { row[j] += delta * k[j]; yq += row[j] * q[j]; }
                yo[i] = (float) yq;
                ss += yq * yq;
            }
            const double s = 1.0 / sqrt(ss / Dv + c->eps);   // RMSNorm (plain weight), gated by sigmoid(z)
            for (int i = 0; i < Dv; ++i) yo[i] = (float) (yo[i] * s * nw[i] * sigm(z[(size_t) t * vd + (size_t) hv * Dv + i]));
        }
    }
    lin(r, tens(r, "%s.out_proj.weight", pre), 0, y, n, out);
    free(qkv); free(z); free(bb); free(aa); free(y); free(cw); free(alog); free(dtb); free(nw); free(S); free(cv);
}

// ---- QSA attention: dense within the budget; past it, each query sees the indexer's top blocks and its own tail -------

// (score, index) pairs, best first.  Equal scores (ReLU zeros are common) go to the higher index, as MLX's argpartition.
static int cmp_desc(const void* a, const void* b) {
    const double* x = (const double*) a;
    const double* y = (const double*) b;
    return x[0] < y[0] ? 1 : x[0] > y[0] ? -1 : x[1] < y[1] ? 1 : x[1] > y[1] ? -1 : 0;
}

// visible[t * n + u] = 1 when query t attends to key u
static void qsa_select(Qwen4Ref* r, const char* pre, const float* x, int n, uint8_t* visible) {
    const Qwen4Cfg* c = &r->c;
    const int nh = c->ix_heads, D = c->ix_dim, R = c->ix_ratio, topk = c->ix_budget / c->ix_ratio, w = (nh + 1) * D, nb = n / R;
    float* ix = (float*) zalloc(sizeof(float) * (size_t) n * w);
    lin(r, tens(r, "%s.indexer.index_qk_proj.weight", pre), 0, x, n, ix);
    float* qw = vec(r, tens(r, "%s.indexer.q_layernorm.weight", pre), D);
    float* kw = vec(r, tens(r, "%s.indexer.k_layernorm.weight", pre), D);
    float* blk = (float*) zalloc(sizeof(float) * (size_t) (nb ? nb : 1) * D);
    for (int j = 0; j < nb; ++j) {   // a block key: the mean of its raw keys, normalised, at the block's first position
        float* bk = blk + (size_t) j * D;
        for (int i = 0; i < D; ++i) {
            double m = 0;
            for (int u = 0; u < R; ++u) m += ix[(size_t) (j * R + u) * w + nh * D + i];
            bk[i] = (float) (m / R);
        }
        norm1p(bk, 1, D, D, kw, c->eps, bk);
        rope(bk, c->rot_dim, c->rope_theta, j * R);
    }
    double* sc = (double*) zalloc(sizeof(double) * 2 * (size_t) (nb ? nb : 1));
    for (int t = 0; t < n; ++t) {
        const int done = (t + 1) / R;   // complete blocks visible to query t
        uint8_t* vis = visible + (size_t) t * n;
        memset(vis, 0, (size_t) n);
        if (done <= topk) { memset(vis, 1, (size_t) t + 1); continue; }
        float* q = ix + (size_t) t * w;
        for (int h = 0; h < nh; ++h) { norm1p(q + h * D, 1, D, D, qw, c->eps, q + h * D); rope(q + h * D, c->rot_dim, c->rope_theta, t); }
        for (int j = 0; j < done; ++j) {
            double s = 0;
            for (int h = 0; h < nh; ++h) {
                double d = 0;
                for (int i = 0; i < D; ++i) d += (double) q[h * D + i] * blk[(size_t) j * D + i];
                s += d > 0 ? d : 0;
            }
            sc[2 * j] = s;
            sc[2 * j + 1] = j;
        }
        qsort(sc, (size_t) done, 2 * sizeof(double), cmp_desc);
        for (int k = 0; k < topk; ++k) memset(vis + (int) sc[2 * k + 1] * R, 1, (size_t) R);
        memset(vis + done * R, 1, (size_t) (t + 1 - done * R));
    }
    free(ix); free(qw); free(kw); free(blk); free(sc);
}

static void attention(Qwen4Ref* r, int l, const float* x, int n, float* out) {
    const Qwen4Cfg* c = &r->c;
    const int Hq = c->n_head, Hkv = c->n_kv, hd = c->head_dim, rep = Hq / Hkv;
    char pre[96];
    snprintf(pre, sizeof pre, "language_model.model.layers.%d.self_attn", l);
    float* qp = (float*) zalloc(sizeof(float) * (size_t) n * Hq * 2 * hd);
    float* kp = (float*) zalloc(sizeof(float) * (size_t) n * Hkv * hd);
    float* vp = (float*) zalloc(sizeof(float) * (size_t) n * Hkv * hd);
    float* o = (float*) zalloc(sizeof(float) * (size_t) n * Hq * hd);
    uint8_t* vis = (uint8_t*) zalloc((size_t) n * n);
    double* p = (double*) zalloc(sizeof(double) * (size_t) n);
    lin(r, tens(r, "%s.q_proj.weight", pre), 0, x, n, qp);
    lin(r, tens(r, "%s.k_proj.weight", pre), 0, x, n, kp);
    lin(r, tens(r, "%s.v_proj.weight", pre), 0, x, n, vp);
    float* qn = vec(r, tens(r, "%s.q_norm.weight", pre), hd);
    float* kn = vec(r, tens(r, "%s.k_norm.weight", pre), hd);
    for (int t = 0; t < n; ++t) {
        for (int h = 0; h < Hq; ++h) {   // q_proj: per head hd query values, then hd gate values
            float* q = qp + ((size_t) t * Hq + h) * 2 * hd;
            norm1p(q, 1, hd, hd, qn, c->eps, q);
            rope(q, c->rot_dim, c->rope_theta, t);
        }
        for (int g = 0; g < Hkv; ++g) {
            float* k = kp + ((size_t) t * Hkv + g) * hd;
            norm1p(k, 1, hd, hd, kn, c->eps, k);
            rope(k, c->rot_dim, c->rope_theta, t);
        }
    }
    qsa_select(r, pre, x, n, vis);
    const double scale = 1.0 / sqrt((double) hd);
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < Hq; ++h) {
            const float* q = qp + ((size_t) t * Hq + h) * 2 * hd;
            const int g = h / rep;
            double mx = -INFINITY, sum = 0;
            for (int u = 0; u <= t; ++u) {
                if (!vis[(size_t) t * n + u]) continue;
                const float* k = kp + ((size_t) u * Hkv + g) * hd;
                double d = 0;
                for (int i = 0; i < hd; ++i) d += (double) q[i] * k[i];
                p[u] = d * scale;
                if (p[u] > mx) mx = p[u];
            }
            for (int u = 0; u <= t; ++u) if (vis[(size_t) t * n + u]) sum += (p[u] = exp(p[u] - mx));
            float* oo = o + ((size_t) t * Hq + h) * hd;
            for (int i = 0; i < hd; ++i) {
                double a = 0;
                for (int u = 0; u <= t; ++u) if (vis[(size_t) t * n + u]) a += p[u] * vp[((size_t) u * Hkv + g) * hd + i];
                oo[i] = (float) (a / sum) * sigm(q[hd + i]);
            }
        }
    lin(r, tens(r, "%s.o_proj.weight", pre), 0, o, n, out);
    free(qp); free(kp); free(vp); free(o); free(vis); free(p); free(qn); free(kn);
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

static void moe(Qwen4Ref* r, int l, const float* x, int n, float* out) {
    const Qwen4Cfg* c = &r->c;
    const int E = c->n_exp, D = c->d;
    char pre[96], sw[112], sh[112];
    snprintf(pre, sizeof pre, "language_model.model.layers.%d.mlp", l);
    snprintf(sw, sizeof sw, "%s.switch_mlp", pre);
    snprintf(sh, sizeof sh, "%s.shared_expert", pre);
    float* lg = (float*) zalloc(sizeof(float) * (size_t) n * E);
    float* y = (float*) zalloc(sizeof(float) * (size_t) D);
    double* pr = (double*) zalloc(sizeof(double) * 2 * (size_t) E);
    lin(r, tens(r, "%s.gate.weight", pre), 0, x, n, lg);
    for (int t = 0; t < n; ++t) {
        const float* xt = x + (size_t) t * D;
        float* ot = out + (size_t) t * D;
        double mx = -INFINITY, sum = 0, top = 0;
        for (int e = 0; e < E; ++e) if (lg[t * E + e] > mx) mx = lg[t * E + e];
        for (int e = 0; e < E; ++e) { pr[2 * e] = exp(lg[t * E + e] - mx); pr[2 * e + 1] = e; sum += pr[2 * e]; }
        qsort(pr, (size_t) E, 2 * sizeof(double), cmp_desc);
        for (int k = 0; k < c->top_k; ++k) top += pr[2 * k];
        memset(ot, 0, sizeof(float) * (size_t) D);
        for (int k = 0; k < c->top_k; ++k) {   // softmax then renormalise over the top k: p / sum of top-k p
            glu(r, sw, (int) pr[2 * k + 1], c->ff_exp, xt, y);
            for (int i = 0; i < D; ++i) ot[i] += (float) (pr[2 * k] / top) * y[i];
        }
        float sg;
        lin(r, tens(r, "%s.shared_expert_gate.weight", pre), 0, xt, 1, &sg);
        glu(r, sh, 0, c->ff_shared, xt, y);
        for (int i = 0; i < D; ++i) ot[i] += sigm(sg) * y[i];
        (void) sum;
    }
    free(lg); free(y); free(pr);
}

// ---- PLE: n-gram rows -> keys gating a value per stream, plus a dilated causal conv of the gated values --------------

static void ple(Qwen4Ref* r, int l, const int32_t* ids, float* H, int n) {
    const Qwen4Cfg* c = &r->c;
    const Qwen4Ngram* g = &r->ng[c->ple[l]];
    const int D = c->d, HD = c->hc * D, P = c->ple_dim, wd = P / g->heads, Kc = c->ple_conv, dil = c->ngram;
    char pre[96];
    snprintf(pre, sizeof pre, "language_model.model.layers.%d.ple", l);
    float* emb = (float*) zalloc(sizeof(float) * (size_t) n * P);
    const int64_t base = g->rows / c->ngram_parts, rem = g->rows % c->ngram_parts;   // shard k: base + (k < rem) rows
    for (int t = 0; t < n; ++t) {
        int64_t rows[QWEN4_MAX_PLE_HEADS];
        qwen4_ngram_rows(c, g, ids, t, rows);
        for (int h = 0; h < g->heads; ++h) {
            const int64_t big = rem * (base + 1), k = rows[h] < big ? rows[h] / (base + 1) : rem + (rows[h] - big) / base;
            const int64_t first = k < rem ? k * (base + 1) : big + (k - rem) * base;
            const NsTensor* s = tens(r, "%s.ple_embedding.ngram_embedding.shards.%lld.weight", pre, (long long) k);
            if (s && s->cols == wd && rows[h] - first < s->rows) ns_row_f32(&r->rows, s, 0, (int) (rows[h] - first), 0, emb + (size_t) t * P + h * wd);
            else if (!r->err[0]) snprintf(r->err, sizeof r->err, "%s: PLE shard %lld does not hold row %lld", pre, (long long) k, (long long) rows[h]);
        }
    }
    float* keys = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* qs = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* vals = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* gated = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* nk = vec(r, tens(r, "%s.norm_key.weight", pre), HD);
    float* nq = vec(r, tens(r, "%s.norm_query.weight", pre), HD);
    float* nc = vec(r, tens(r, "%s.norm_conv.weight", pre), HD);
    float* cw = vec(r, tens(r, "%s.conv1d.weight", pre), (int64_t) HD * Kc);
    lin(r, tens(r, "%s.key_proj.weight", pre), 0, emb, n, keys);
    lin(r, tens(r, "%s.value_proj.weight", pre), 0, emb, n, vals);
    norm1p(keys, n, HD, D, nk, c->eps, keys);
    norm1p(H, n, HD, D, nq, c->eps, qs);
    for (int t = 0; t < n; ++t)
        for (int s = 0; s < c->hc; ++s) {
            const size_t o = (size_t) t * HD + (size_t) s * D;
            double d = 0;
            for (int i = 0; i < D; ++i) d += (double) keys[o + i] * qs[o + i];
            d /= sqrt((double) D);
            const double gs = (d > 0 ? 1 : d < 0 ? -1 : 0) * sqrt(fabs(d) > 1e-6 ? fabs(d) : 1e-6);   // a signed square root
            for (int i = 0; i < D; ++i) gated[o + i] = sigm(gs) * vals[(size_t) t * D + i];
        }
    norm1p(gated, n, HD, D, nc, c->eps, keys);   // keys now holds the conv input
    for (int t = 0; t < n; ++t)
        for (int ch = 0; ch < HD; ++ch) {
            double acc = 0;
            for (int k = 0; k < Kc; ++k) {
                const int u = t - (Kc - 1 - k) * dil;
                if (u >= 0) acc += (double) cw[(size_t) ch * Kc + k] * keys[(size_t) u * HD + ch];
            }
            H[(size_t) t * HD + ch] += gated[(size_t) t * HD + ch] + silu(acc);
        }
    free(emb); free(keys); free(qs); free(vals); free(gated); free(nk); free(nq); free(nc); free(cw);
}

// ---- the model -------------------------------------------------------------------------------------------------------

Qwen4Ref* qwen4_ref_open(const char* model_dir, char* err, int errlen) {
    Qwen4Ref* r = (Qwen4Ref*) calloc(1, sizeof *r);
    if (qwen4_cfg_load(&r->c, model_dir, err, errlen) || ns_open(&r->nm, model_dir, err, errlen)) { free(r); return NULL; }
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

int qwen4_ref_forward(Qwen4Ref* r, const int32_t* ids, int n, float* logits) {
    const Qwen4Cfg* c = &r->c;
    const int D = c->d, HD = c->hc * D;
    r->err[0] = 0;
    float* H = (float*) zalloc(sizeof(float) * (size_t) n * HD);
    float* x = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* b = (float*) zalloc(sizeof(float) * (size_t) n * D);
    float* inj = (float*) zalloc(sizeof(float) * (size_t) n * c->hc);
    const NsTensor* E = tens(r, "language_model.model.embed_tokens.weight");
    for (int t = 0; E && t < n; ++t) {
        if (ids[t] < 0 || ids[t] >= E->rows) { snprintf(r->err, sizeof r->err, "token %d out of range", ids[t]); break; }
        ns_row_f32(&r->rows, E, 0, ids[t], 0, x);
        for (int s = 0; s < c->hc; ++s) memcpy(H + (size_t) t * HD + (size_t) s * D, x, sizeof(float) * (size_t) D);
    }
    char pre[112];
    for (int l = 0; l < c->n_layer && !r->err[0]; ++l) {
        if (c->ple[l] >= 0) ple(r, l, ids, H, n);
        snprintf(pre, sizeof pre, "language_model.model.layers.%d.attn_hyper_connection", l);
        hc_mix(r, pre, H, n, x, inj);
        if (c->attn[l]) attention(r, l, x, n, b);
        else gdn(r, l, x, n, b);
        hc_add(c, H, n, b, inj);
        snprintf(pre, sizeof pre, "language_model.model.layers.%d.mlp_hyper_connection", l);
        hc_mix(r, pre, H, n, x, inj);
        moe(r, l, x, n, b);
        hc_add(c, H, n, b, inj);
    }
    if (!r->err[0]) {
        hc_mix(r, "language_model.model.hyper_connection_mixer", H, n, x, NULL);
        lin(r, tens(r, "language_model.lm_head.weight"), 0, x, n, logits);
    }
    free(H); free(x); free(b); free(inj);
    return r->err[0] ? -1 : 0;
}
