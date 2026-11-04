// nslm/lib_qwen4_ops.c - the qwen4_exp operations of one token (nslm/qwen4_ops.h).
#include "qwen4_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }

void qwen4_norm1p(const float* x, int len, int g, const float* w, float eps, float* y) {
    for (int a = 0; a < len; a += g) {
        double ss = 0;
        for (int k = 0; k < g; ++k) ss += (double) x[a + k] * x[a + k];
        const double s = 1.0 / sqrt(ss / g + eps);
        for (int k = 0; k < g; ++k) y[a + k] = (float) (x[a + k] * s * (1.0 + w[a + k]));
    }
}

size_t qwen4_gdn_conv_len(const Qwen4Cfg* c) {
    return (size_t) (c->lin_conv - 1) * (size_t) (2 * c->lin_hk * c->lin_dk + c->lin_hv * c->lin_dv);
}
size_t qwen4_gdn_state_len(const Qwen4Cfg* c) { return (size_t) c->lin_hv * c->lin_dk * c->lin_dv; }

void qwen4_gdn_token(const Qwen4Cfg* c, const Qwen4GdnW* w, const float* qkv, const float* z, const float* a, const float* b,
                     float* conv, float* S, float* y) {
    const int Hk = c->lin_hk, Hv = c->lin_hv, Dk = c->lin_dk, Dv = c->lin_dv, K = c->lin_conv;
    const int kd = Hk * Dk, cd = 2 * kd + Hv * Dv, rep = Hv / Hk;
    float* cv = (float*) malloc(sizeof(float) * (size_t) cd);
    for (int ch = 0; ch < cd; ++ch) {   // depthwise causal conv over the last K inputs, then silu
        double acc = (double) w->conv_w[(size_t) ch * K + K - 1] * qkv[ch];
        for (int k = 0; k < K - 1; ++k) acc += (double) w->conv_w[(size_t) ch * K + k] * conv[(size_t) k * cd + ch];
        cv[ch] = (float) (acc * sigm(acc));
    }
    if (K > 1) {
        memmove(conv, conv + cd, sizeof(float) * (size_t) (K - 2) * cd);
        memcpy(conv + (size_t) (K - 2) * cd, qkv, sizeof(float) * (size_t) cd);
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
        const double ap = (double) a[hv] + w->dt_bias[hv], sp = ap > 20 ? ap : log1p(exp(ap));
        const double g = exp(-exp((double) w->A_log[hv]) * sp), beta = sigm(b[hv]);
        float* St = S + (size_t) hv * Dk * Dv;   // [dk][dv]
        float* yo = y + (size_t) hv * Dv;
        double ss = 0;
        for (int i = 0; i < Dv; ++i) {
            double kv = 0, yq = 0;
            for (int j = 0; j < Dk; ++j) kv += (double) St[(size_t) j * Dv + i] * k[j];
            const double delta = (v[i] - g * kv) * beta;
            for (int j = 0; j < Dk; ++j) {
                const float s = (float) (g * St[(size_t) j * Dv + i] + delta * k[j]);
                St[(size_t) j * Dv + i] = s;
                yq += (double) s * q[j];
            }
            yo[i] = (float) yq;
            ss += yq * yq;
        }
        const double s = 1.0 / sqrt(ss / Dv + c->eps);   // RMSNorm (plain weight), gated by sigmoid(z)
        for (int i = 0; i < Dv; ++i) yo[i] = (float) (yo[i] * s * w->norm_w[i] * sigm(z[(size_t) hv * Dv + i]));
    }
    free(cv);
}

void qwen4_hc_act(const Qwen4Cfg* c, float* lo) {
    for (int i = 0; i < c->hc_rank; ++i) { const double x = lo[i] / (double) c->hc; lo[i] = (float) (x * sigm(x)); }
}
void qwen4_hc_mix(const Qwen4Cfg* c, const float* normed, const float* up, float* mixed) {
    for (int i = 0; i < c->d; ++i) {
        double m = 0;
        for (int s = 0; s < c->hc; ++s) m += sigm(up[s * c->d + i]) * normed[s * c->d + i];
        mixed[i] = (float) (m / c->hc);
    }
}
void qwen4_hc_gate(const Qwen4Cfg* c, float* inj) {
    for (int s = 0; s < c->hc; ++s) inj[s] = (float) (2.0 * sigm(inj[s] / (double) c->hc));
}
void qwen4_hc_add(const Qwen4Cfg* c, float* H, const float* branch, const float* inj) {
    for (int s = 0; s < c->hc; ++s)
        for (int i = 0; i < c->d; ++i) H[s * c->d + i] += branch[i] * inj[s];
}

static void rope(float* x, int rot, float theta, int pos) {   // halves rotated against each other; f32 angles, as MLX
    for (int i = 0; i < rot / 2; ++i) {
        const float inv = (float) pow((double) theta, -2.0 * i / rot), a = (float) pos * inv;
        const double cs = cos(a), sn = sin(a), x1 = x[i], x2 = x[i + rot / 2];
        x[i] = (float) (x1 * cs - x2 * sn);
        x[i + rot / 2] = (float) (x2 * cs + x1 * sn);
    }
}
void qwen4_head_norm_rope(const Qwen4Cfg* c, float* x, int nh, int hs, int hd, const float* w, int pos) {
    for (int h = 0; h < nh; ++h) {
        qwen4_norm1p(x + (size_t) h * hs, hd, hd, w, c->eps, x + (size_t) h * hs);
        rope(x + (size_t) h * hs, c->rot_dim, c->rope_theta, pos);
    }
}

// (score, index) pairs, best first; equal scores go to the higher index, as MLX's argpartition
static int cmp_desc(const void* a, const void* b) {
    const double* x = (const double*) a;
    const double* y = (const double*) b;
    return x[0] < y[0] ? 1 : x[0] > y[0] ? -1 : x[1] < y[1] ? 1 : x[1] > y[1] ? -1 : 0;
}

void qwen4_route(const Qwen4Cfg* c, const float* logits, int32_t* ind, float* wt) {
    const int E = c->n_exp;
    double* pr = (double*) malloc(sizeof(double) * 2 * (size_t) E), mx = -INFINITY, top = 0;
    for (int e = 0; e < E; ++e) if (logits[e] > mx) mx = logits[e];
    for (int e = 0; e < E; ++e) { pr[2 * e] = exp(logits[e] - mx); pr[2 * e + 1] = e; }
    qsort(pr, (size_t) E, 2 * sizeof(double), cmp_desc);
    for (int k = 0; k < c->top_k; ++k) top += pr[2 * k];
    for (int k = 0; k < c->top_k; ++k) { ind[k] = (int32_t) pr[2 * k + 1]; wt[k] = (float) (pr[2 * k] / top); }
    free(pr);
}

int qwen4_ple_embed(const NsModel* m, const NsRows* rw, const PleTable* table, const Qwen4Cfg* c, const Qwen4Ngram* g, int layer,
                    const int32_t* ids, int t, float* emb) {
    const int wd = c->ple_dim / g->heads;
    const int64_t base = g->rows / c->ngram_parts, rem = g->rows % c->ngram_parts, big = rem * (base + 1);   // shard k: base + (k < rem) rows
    int64_t rows[QWEN4_MAX_PLE_HEADS];
    qwen4_ngram_rows(c, g, ids, t, rows);
    for (int h = 0; h < g->heads; ++h) {
        if (table) {
            if (table->width != wd || ple_row(table, rows[h], emb + (size_t) h * wd)) return -1;
            continue;
        }
        const int64_t k = rows[h] < big ? rows[h] / (base + 1) : rem + (rows[h] - big) / base;
        const int64_t first = k < rem ? k * (base + 1) : big + (k - rem) * base;
        char name[160];
        snprintf(name, sizeof name, "language_model.model.layers.%d.ple.ple_embedding.ngram_embedding.shards.%lld.weight", layer, (long long) k);
        const NsTensor* s = ns_find(m, name);
        if (!s || s->cols != wd || rows[h] - first >= s->rows || s->enc == NS_SEED6P8) return -1;
        ns_row_f32(rw, s, 0, (int) (rows[h] - first), 0, emb + (size_t) h * wd);
    }
    return 0;
}

void qwen4_ple_gate(const Qwen4Cfg* c, const float* keys, const float* qs, const float* vals, float* gated) {
    for (int s = 0; s < c->hc; ++s) {
        double d = 0;
        for (int i = 0; i < c->d; ++i) d += (double) keys[s * c->d + i] * qs[s * c->d + i];
        d /= sqrt((double) c->d);
        const double gs = sigm((d > 0 ? 1 : d < 0 ? -1 : 0) * sqrt(fabs(d) > 1e-6 ? fabs(d) : 1e-6));   // a signed square root
        for (int i = 0; i < c->d; ++i) gated[s * c->d + i] = (float) (gs * vals[i]);
    }
}

size_t qwen4_ple_hist_len(const Qwen4Cfg* c) { return (size_t) (c->ple_conv - 1) * c->ngram * c->hc * c->d; }
void qwen4_ple_conv(const Qwen4Cfg* c, const float* w, const float* cin, const float* gated, float* hist, float* H) {
    const int HD = c->hc * c->d, K = c->ple_conv, L = (K - 1) * c->ngram;
    for (int ch = 0; ch < HD; ++ch) {   // tap k reads the input (K - 1 - k) * ngram tokens back
        double acc = (double) w[(size_t) ch * K + K - 1] * cin[ch];
        for (int k = 0; k < K - 1; ++k) acc += (double) w[(size_t) ch * K + k] * hist[(size_t) (L - (K - 1 - k) * c->ngram) * HD + ch];
        H[ch] += gated[ch] + (float) (acc * sigm(acc));
    }
    if (L > 0) {
        memmove(hist, hist + HD, sizeof(float) * (size_t) (L - 1) * HD);
        memcpy(hist + (size_t) (L - 1) * HD, cin, sizeof(float) * (size_t) HD);
    }
}

void qwen4_qsa_block(const Qwen4Cfg* c, const float* raw, const float* w, int j, float* out) {
    for (int i = 0; i < c->ix_dim; ++i) {
        double m = 0;
        for (int u = 0; u < c->ix_ratio; ++u) m += raw[(size_t) u * c->ix_dim + i];
        out[i] = (float) (m / c->ix_ratio);
    }
    qwen4_head_norm_rope(c, out, 1, c->ix_dim, c->ix_dim, w, j * c->ix_ratio);
}

int qwen4_qsa_select(const Qwen4Cfg* c, const float* iq, const float* blk, int done, int32_t* sel) {
    const int topk = c->ix_budget / c->ix_ratio, D = c->ix_dim;
    if (done <= topk) return 0;
    double* sc = (double*) malloc(sizeof(double) * 2 * (size_t) done);
    for (int j = 0; j < done; ++j) {
        double s = 0;
        for (int h = 0; h < c->ix_heads; ++h) {
            double d = 0;
            for (int i = 0; i < D; ++i) d += (double) iq[h * D + i] * blk[(size_t) j * D + i];
            s += d > 0 ? d : 0;
        }
        sc[2 * j] = s;
        sc[2 * j + 1] = j;
    }
    qsort(sc, (size_t) done, 2 * sizeof(double), cmp_desc);
    for (int k = 0; k < topk; ++k) sel[k] = (int32_t) sc[2 * k + 1];
    free(sc);
    return topk;
}

void qwen4_attend(const Qwen4Cfg* c, const float* q, const float* K, const float* V, int pos, const int32_t* sel, int nsel,
                  float* o) {
    const int hd = c->head_dim, nkv = c->n_kv, rep = c->n_head / nkv, R = c->ix_ratio, tail = nsel ? (pos + 1) / R * R : 0;
    int32_t* keys = (int32_t*) malloc(sizeof(int32_t) * (size_t) (pos + 1));
    int n = 0;
    for (int k = 0; k < nsel; ++k) for (int u = 0; u < R; ++u) keys[n++] = sel[k] * R + u;
    for (int u = tail; u <= pos; ++u) keys[n++] = u;
    double* p = (double*) malloc(sizeof(double) * (size_t) n);
    const double scale = 1.0 / sqrt((double) hd);
    for (int h = 0; h < c->n_head; ++h) {
        const float* qh = q + (size_t) h * 2 * hd;
        const int g = h / rep;
        double mx = -INFINITY, sum = 0;
        for (int k = 0; k < n; ++k) {
            const float* kk = K + ((size_t) keys[k] * nkv + g) * hd;
            double d = 0;
            for (int i = 0; i < hd; ++i) d += (double) qh[i] * kk[i];
            p[k] = d * scale;
            if (p[k] > mx) mx = p[k];
        }
        for (int k = 0; k < n; ++k) sum += (p[k] = exp(p[k] - mx));
        for (int i = 0; i < hd; ++i) {
            double a = 0;
            for (int k = 0; k < n; ++k) a += p[k] * V[((size_t) keys[k] * nkv + g) * hd + i];
            o[(size_t) h * hd + i] = (float) (a / sum * sigm(qh[hd + i]));
        }
    }
    free(keys);
    free(p);
}

