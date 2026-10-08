// nslm/lib_mova_ref.c - the CPU reference forward (nslm/mova_ref.h).  The whole sequence goes through one layer at a
// time, so attention needs only the current layer's keys and values.
#include "mova_ref.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"
#include "model_st.h"
#include "mova_cfg.h"
#include "search4.h"

struct MovaRef {
    MovaCfg c;
    NsModel nm;
    int threads;
    uint32_t g24[65536], g32[65536];   // stream tables (lfsr_stream24 / lfsr_stream32)
    float inv[64];
};

static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
static float bfr(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float silu_bf(float g) { const float s = bfr(1.0f / (1.0f + expf(-g))); return bfr(g * s); }
static float softplus_gate(float g) {
    const float gx = g * 0.69314718055994531f;
    return bfr((fmaxf(gx, 0.0f) + logf(1.0f + expf(-fabsf(gx)))) / 0.69314718055994531f);
}

// ---- weights: one row of a tensor slice as f32 (the kernels' values) ------------------------------------------------

static void row_f32(const MovaRef* r, const NsTensor* t, int slice, int row, float* out) {
    const int K = t->cols;
    const uint64_t R = (uint64_t) t->rows, base = ((uint64_t) slice * R + (uint64_t) row) * (uint64_t) K;
    switch (t->enc) {
    case NS_BF16: {
        const uint16_t* w = (const uint16_t*) t->s[0].p + base;
        for (int c = 0; c < K; ++c) out[c] = bf2f(w[c]);
        break;
    }
    case NS_Q8:
    case NS_Q4: {
        const uint32_t* q = (const uint32_t*) t->s[0].p;
        const uint16_t* S = (const uint16_t*) t->s[1].p;
        const uint16_t* B = (const uint16_t*) t->s[2].p;
        for (int c = 0; c < K; ++c) {
            const uint64_t i = base + (uint64_t) c;
            const float s = bf2f(S[i / 64]), b = bf2f(B[i / 64]);
            const uint32_t v = t->enc == NS_Q8 ? (q[i / 4] >> (8 * (i % 4))) & 255u : (q[i / 8] >> (4 * (i % 8))) & 15u;
            out[c] = bfr(s * (float) v + b);
        }
        break;
    }
    case NS_SEED4P4:
    case NS_SEED4: {
        const int p4 = t->enc == NS_SEED4P4, P = p4 ? 4 : 3, nbk = K / 8;
        const uint16_t* seeds = (const uint16_t*) t->s[0].p;
        const uint16_t* cw = (const uint16_t*) t->s[1].p;
        int32_t eb;
        memcpy(&eb, t->s[2].p + 4 * (size_t) slice, 4);
        for (int j = 0; j < nbk; ++j) {
            const uint64_t k = base / 8 + (uint64_t) j;
            const uint32_t s = seeds[k], w = cw[k];
            const uint64_t st = (uint64_t) s | ((uint64_t) (p4 ? r->g32[s] : r->g24[s]) << 16);
            int e, q[4];
            if (p4) {
                e = eb + ((t->s[3].p[k >> 1] >> ((k & 1) * 4)) & 15);
                for (int p = 0; p < 4; ++p) q[p] = nslm4_q((uint16_t) w, p);
            } else {
                e = eb + (int) (w & 15u);
                for (int p = 0; p < 3; ++p) q[p] = nslm_q((uint16_t) w, p);
            }
            const float sc = nslm_scale(e);
            for (int cc = 0; cc < 8; ++cc) {
                int32_t isum = 0;
                for (int p = 0; p < P; ++p) isum += ((int32_t) ((st >> (cc * P + p + 1)) & 0xFFFFu) - 32768) * q[p];
                out[j * 8 + cc] = (float) isum * sc;
            }
        }
        break;
    }
    }
}

// ---- a pool of row jobs -----------------------------------------------------------------------------------------------

typedef struct {
    const NsTensor* t;
    int slice;
    const float* x;   // input rows (stride K)
    int n;            // rows of x
    float* y;         // output rows (stride ys)
    int ys, add;
} Lin;
#define CH 16   // weight rows per work item
typedef struct {
    const MovaRef* r;
    const Lin* L;
    int nl;
    int* first;   // first work item of each Lin (prefix), nl + 1 entries
    int next;
    pthread_mutex_t mu;
} Pool;

static void lin_item(Pool* p, int it) {
    int a = 0, b = p->nl;   // the Lin holding item it
    while (b - a > 1) { const int m = (a + b) / 2; if (p->first[m] <= it) a = m; else b = m; }
    const Lin* L = &p->L[a];
    const int K = L->t->cols, R = L->t->rows, r0 = (it - p->first[a]) * CH;
    float* w = (float*) malloc(sizeof(float) * (size_t) K);
    for (int r = r0; r < r0 + CH && r < R; ++r) {
        row_f32(p->r, L->t, L->slice, r, w);
        for (int i = 0; i < L->n; ++i) {
            const float* x = L->x + (size_t) i * K;
            double acc = 0;
            for (int c = 0; c < K; ++c) acc += (double) w[c] * (double) x[c];
            float* yo = L->y + (size_t) i * L->ys + r;
            *yo = L->add ? bfr(*yo + bfr((float) acc)) : bfr((float) acc);
        }
    }
    free(w);
}
static void* lin_worker(void* arg) {
    Pool* p = (Pool*) arg;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        const int it = p->next++;
        pthread_mutex_unlock(&p->mu);
        if (it >= p->first[p->nl]) return NULL;
        lin_item(p, it);
    }
}
static void run_lins(const MovaRef* r, const Lin* L, int nl) {
    if (!nl) return;
    Pool p;
    memset(&p, 0, sizeof p);
    p.r = r; p.L = L; p.nl = nl;
    p.first = (int*) malloc(sizeof(int) * (size_t) (nl + 1));
    p.first[0] = 0;
    for (int i = 0; i < nl; ++i) p.first[i + 1] = p.first[i] + (L[i].t->rows + CH - 1) / CH;
    pthread_mutex_init(&p.mu, NULL);
    const int nt = r->threads < 64 ? r->threads : 64;
    pthread_t th[64];
    for (int i = 0; i < nt; ++i) pthread_create(&th[i], NULL, lin_worker, &p);
    for (int i = 0; i < nt; ++i) pthread_join(th[i], NULL);
    pthread_mutex_destroy(&p.mu);
    free(p.first);
}
static void lin1(const MovaRef* r, const NsTensor* t, int slice, const float* x, int n, float* y, int ys, int add) {
    Lin L = {t, slice, x, n, y, ys, add};
    run_lins(r, &L, 1);
}

// ---- open -------------------------------------------------------------------------------------------------------------

MovaRef* mova_ref_open(const char* dir, int threads, char* err, int errlen) {
    MovaRef* r = (MovaRef*) calloc(1, sizeof *r);
    if (mova_cfg_load(&r->c, dir, err, errlen) || ns_open(&r->nm, dir, err, errlen)) { free(r); return NULL; }
    r->threads = threads > 0 ? threads : 1;
    for (uint32_t s = 0; s < 65536; ++s) { r->g24[s] = lfsr_stream24((uint16_t) s); r->g32[s] = lfsr_stream32((uint16_t) s); }
    for (int p = 0; p < 64; ++p) r->inv[p] = (float) pow((double) r->c.rope_theta, -2.0 * p / (double) r->c.head_dim);
    return r;
}
void mova_ref_close(MovaRef* r) {
    if (!r) return;
    ns_close(&r->nm);
    free(r);
}
int mova_ref_vocab(const MovaRef* r) { return r->c.vocab; }

static const NsTensor* T_(const MovaRef* r, const char* fmt, int l) {
    char nm[160];
    snprintf(nm, sizeof nm, fmt, l);
    return ns_find(&r->nm, nm);
}

static void gnorm(const MovaRef* r, const float* x, const NsTensor* w, float* y, int n) {
    const int d = r->c.d, hw = d / 2;
    const uint16_t* wv = (const uint16_t*) w->s[0].p;
    for (int t = 0; t < n; ++t) {
        const float* xr = x + (size_t) t * d;
        double a = 0, b = 0;
        for (int c = 0; c < d; ++c) { if (c < hw) a += (double) xr[c] * xr[c]; else b += (double) xr[c] * xr[c]; }
        const float r0 = (float) (1.0 / sqrt(a / hw + r->c.eps)), r1 = (float) (1.0 / sqrt(b / hw + r->c.eps));
        for (int c = 0; c < d; ++c) y[(size_t) t * d + c] = bfr(bf2f(wv[c]) * (xr[c] * (c < hw ? r0 : r1)));
    }
}

// Router of n rows: top k by sigmoid + bias (ties to the lowest id), weights = sigmoid / sum * scale.
static void router(const MovaRef* r, const NsTensor* W, const NsTensor* bias, const float* x, int n, int ne, int k,
                   int32_t* inds, float* wts) {
    const int d = r->c.d, hw = d / r->c.router_parts;
    const uint16_t* w = (const uint16_t*) W->s[0].p;
    const uint16_t* b = (const uint16_t*) bias->s[0].p;
    float* sc = (float*) malloc(sizeof(float) * (size_t) ne);
    float* se = (float*) malloc(sizeof(float) * (size_t) ne);
    for (int t = 0; t < n; ++t) {
        const float* xr = x + (size_t) t * d;
        for (int e = 0; e < ne; ++e) {
            double s0 = 0, s1 = 0;
            for (int c = 0; c < d; ++c) {
                const double v = (double) bf2f(w[(size_t) e * d + c]) * xr[c];
                if (c < hw) s0 += v; else s1 += v;
            }
            sc[e] = 1.0f / (1.0f + expf(-(bfr((float) s0) + bfr((float) s1))));
            se[e] = sc[e] + bf2f(b[e]);
        }
        float sum = 0;
        for (int j = 0; j < k; ++j) {
            int best = -1;
            for (int e = 0; e < ne; ++e) if (se[e] != -INFINITY && (best < 0 || se[e] > se[best])) best = e;
            inds[t * k + j] = best;
            sum += sc[best];
            se[best] = -INFINITY;
        }
        for (int j = 0; j < k; ++j) wts[t * k + j] = sc[inds[t * k + j]] / sum * r->c.route_scale;
    }
    free(sc);
    free(se);
}

// Experts of one layer: out[p] = W_{inds[p]} x[p / xdiv] for the P = n * k pairs (grouped per expert).
static void experts(const MovaRef* r, const NsTensor* t, const int32_t* inds, int P, int xdiv, const float* x, int K, float* out) {
    const int ne = t->slices, R = t->rows;
    Lin* L = (Lin*) calloc((size_t) ne, sizeof(Lin));
    float** xs = (float**) calloc((size_t) ne, sizeof(float*));
    float** ys = (float**) calloc((size_t) ne, sizeof(float*));
    int** who = (int**) calloc((size_t) ne, sizeof(int*));
    int* cnt = (int*) calloc((size_t) ne, sizeof(int));
    for (int p = 0; p < P; ++p) cnt[inds[p]]++;
    int nl = 0;
    for (int e = 0; e < ne; ++e) {
        if (!cnt[e]) continue;
        xs[e] = (float*) malloc(sizeof(float) * (size_t) cnt[e] * K);
        ys[e] = (float*) malloc(sizeof(float) * (size_t) cnt[e] * R);
        who[e] = (int*) malloc(sizeof(int) * (size_t) cnt[e]);
        int m = 0;
        for (int p = 0; p < P; ++p)
            if (inds[p] == e) { memcpy(xs[e] + (size_t) m * K, x + (size_t) (p / xdiv) * K, sizeof(float) * (size_t) K); who[e][m++] = p; }
        L[nl++] = (Lin) {t, e, xs[e], cnt[e], ys[e], R, 0};
    }
    run_lins(r, L, nl);
    for (int e = 0; e < ne; ++e) {
        for (int m = 0; m < cnt[e]; ++m) memcpy(out + (size_t) who[e][m] * R, ys[e] + (size_t) m * R, sizeof(float) * (size_t) R);
        free(xs[e]); free(ys[e]); free(who[e]);
    }
    free(L); free(xs); free(ys); free(who); free(cnt);
}

// ---- attention (one layer, all rows): causal, the softplus output gate -------------------------------------------------
typedef struct {
    const MovaRef* r;
    const float* q;      // [n][n_head * 128], roped, BF16 values
    const float* kc;     // [n][n_kv * 128]
    const float* vc;
    const float* g;      // gate pre-activations [n][n_head * 128]
    float* o;
    int n, next;
    pthread_mutex_t mu;
} Att;
static void* att_worker(void* arg) {
    Att* a = (Att*) arg;
    const MovaCfg* c = &a->r->c;
    const int qd = c->n_head * 128, kvd = c->n_kv * 128, grp = c->n_head / c->n_kv;
    const double scale = 1.0 / sqrt((double) c->head_dim);
    double* s = (double*) malloc(sizeof(double) * (size_t) a->n);
    for (;;) {
        pthread_mutex_lock(&a->mu);
        const int it = a->next++;
        pthread_mutex_unlock(&a->mu);
        if (it >= a->n * c->n_head) break;
        const int t = it / c->n_head, h = it % c->n_head, kh = h / grp;
        const float* qr = a->q + (size_t) t * qd + h * 128;
        double m = -INFINITY;
        for (int p = 0; p <= t; ++p) {
            const float* kr = a->kc + (size_t) p * kvd + kh * 128;
            double v = 0;
            for (int d = 0; d < 128; ++d) v += (double) qr[d] * kr[d];
            s[p] = v * scale;
            if (s[p] > m) m = s[p];
        }
        double l = 0;
        for (int p = 0; p <= t; ++p) { s[p] = exp(s[p] - m); l += s[p]; }
        for (int d = 0; d < 128; ++d) {
            double acc = 0;
            for (int p = 0; p <= t; ++p) acc += s[p] * a->vc[(size_t) p * kvd + kh * 128 + d];
            const size_t i = (size_t) t * qd + h * 128 + d;
            a->o[i] = bfr(bfr((float) (acc / l)) * softplus_gate(a->g[i]));
        }
    }
    free(s);
    return NULL;
}

// ---- forward ----------------------------------------------------------------------------------------------------------

int mova_ref_forward(MovaRef* r, const int32_t* ids, int n, int h0, float* logits, int32_t* mlp_sel, int32_t* val_sel) {
    const MovaCfg* c = &r->c;
    const int d = c->d, qd = c->n_head * 128, kvd = c->n_kv * 128, ff = c->ff_dense > c->ff_exp ? c->ff_dense : c->ff_exp;
    const int ns = c->n_layer - c->first_sparse;
    float* x = (float*) calloc((size_t) n * d, 4);
    float* xn = (float*) calloc((size_t) n * d, 4);
    float* q = (float*) calloc((size_t) n * qd, 4);
    float* g = (float*) calloc((size_t) n * qd, 4);
    float* o = (float*) calloc((size_t) n * qd, 4);
    float* k = (float*) calloc((size_t) n * kvd, 4);
    float* v = (float*) calloc((size_t) n * kvd, 4);
    float* ga = (float*) calloc((size_t) n * ff, 4);
    float* ua = (float*) calloc((size_t) n * ff, 4);
    float* aa = (float*) calloc((size_t) n * ff, 4);
    float* sh = (float*) calloc((size_t) n * d, 4);
    const int PK = n * (c->top_k > c->top_kv ? c->top_k : c->top_kv);
    float* G = (float*) calloc((size_t) PK * c->ff_exp, 4);
    float* U = (float*) calloc((size_t) PK * c->ff_exp, 4);
    float* A = (float*) calloc((size_t) PK * c->ff_exp, 4);
    float* D = (float*) calloc((size_t) PK * d, 4);
    float* V = (float*) calloc((size_t) PK * kvd, 4);
    int32_t* inds = (int32_t*) calloc((size_t) PK, 4);
    float* wts = (float*) calloc((size_t) PK, 4);
    // embedding
    {
        const NsTensor* E = ns_find(&r->nm, "model.embed_tokens.weight");
        float* row = (float*) malloc(sizeof(float) * (size_t) d);
        for (int t = 0; t < n; ++t) { row_f32(r, E, 0, ids[t], row); memcpy(x + (size_t) t * d, row, sizeof(float) * (size_t) d); }
        free(row);
    }
    for (int l = 0; l < c->n_layer; ++l) {
        const int sparse = l >= c->first_sparse;
        gnorm(r, x, T_(r, "model.layers.%d.input_layernorm.weight", l), xn, n);
        {
            Lin L[4] = {{T_(r, "model.layers.%d.self_attn.q_proj.weight", l), 0, xn, n, q, qd, 0},
                        {T_(r, "model.layers.%d.self_attn.k_proj.weight", l), 0, xn, n, k, kvd, 0},
                        {T_(r, "model.layers.%d.self_attn.gate_proj.weight", l), 0, xn, n, g, qd, 0},
                        {T_(r, "model.layers.%d.self_attn.v_proj.weight", l), 0, xn, n, v, kvd, 0}};
            run_lins(r, L, sparse ? 3 : 4);
        }
        if (sparse) {
            router(r, T_(r, "model.layers.%d.self_attn.v_router.weight", l), T_(r, "model.layers.%d.self_attn.v_router.bias", l), xn, n,
                   c->n_vexp, c->top_kv, inds, wts);
            if (val_sel)
                for (int t = 0; t < n; ++t)
                    memcpy(val_sel + ((size_t) t * ns + (l - c->first_sparse)) * c->top_kv, inds + t * c->top_kv, sizeof(int32_t) * (size_t) c->top_kv);
            experts(r, T_(r, "model.layers.%d.self_attn.v_experts.weight", l), inds, n * c->top_kv, c->top_kv, xn, d, V);
            for (int t = 0; t < n; ++t)
                for (int ci = 0; ci < kvd; ++ci) {
                    float s = 0;
                    for (int j = 0; j < c->top_kv; ++j)
                        s += bfr(silu_bf(V[((size_t) t * c->top_kv + j) * kvd + ci]) * bfr(wts[t * c->top_kv + j]));
                    v[(size_t) t * kvd + ci] = bfr(s);
                }
        }
        // RoPE: q in place (BF16), k and v as the cache stores them (BF16)
        for (int t = 0; t < n; ++t)
            for (int p = 0; p < 64; ++p) {
                const float th = (float) t * r->inv[p];
                const float cs = (float) cos((double) th), sn = (float) sin((double) th);
                for (int h = 0; h < c->n_head; ++h) {
                    float* qh = q + (size_t) t * qd + h * 128;
                    const float a = qh[p], b = qh[p + 64];
                    qh[p] = bfr(a * cs - b * sn);
                    qh[p + 64] = bfr(b * cs + a * sn);
                }
                for (int h = 0; h < c->n_kv; ++h) {
                    float* kh = k + (size_t) t * kvd + h * 128;
                    const float a = kh[p], b = kh[p + 64];
                    kh[p] = bfr(a * cs - b * sn);
                    kh[p + 64] = bfr(b * cs + a * sn);
                }
            }
        for (size_t i = 0; i < (size_t) n * kvd; ++i) v[i] = bfr(v[i]);
        {
            Att at;
            memset(&at, 0, sizeof at);
            at.r = r; at.q = q; at.kc = k; at.vc = v; at.g = g; at.o = o; at.n = n;
            pthread_mutex_init(&at.mu, NULL);
            const int nt = r->threads < 64 ? r->threads : 64;
            pthread_t th[64];
            for (int i = 0; i < nt; ++i) pthread_create(&th[i], NULL, att_worker, &at);
            for (int i = 0; i < nt; ++i) pthread_join(th[i], NULL);
            pthread_mutex_destroy(&at.mu);
        }
        lin1(r, T_(r, "model.layers.%d.self_attn.o_proj.weight", l), 0, o, n, x, d, 1);
        gnorm(r, x, T_(r, "model.layers.%d.post_attention_layernorm.weight", l), xn, n);
        if (!sparse) {
            Lin L[2] = {{T_(r, "model.layers.%d.mlp.gate_proj.weight", l), 0, xn, n, ga, c->ff_dense, 0},
                        {T_(r, "model.layers.%d.mlp.up_proj.weight", l), 0, xn, n, ua, c->ff_dense, 0}};
            run_lins(r, L, 2);
            for (size_t i = 0; i < (size_t) n * c->ff_dense; ++i) aa[i] = bfr(silu_bf(ga[i]) * ua[i]);
            lin1(r, T_(r, "model.layers.%d.mlp.down_proj.weight", l), 0, aa, n, x, d, 1);
            continue;
        }
        router(r, T_(r, "model.layers.%d.mlp.gate.weight", l), T_(r, "model.layers.%d.mlp.gate.bias", l), xn, n, c->n_exp, c->top_k, inds, wts);
        if (mlp_sel)
            for (int t = 0; t < n; ++t)
                memcpy(mlp_sel + ((size_t) t * ns + (l - c->first_sparse)) * c->top_k, inds + t * c->top_k, sizeof(int32_t) * (size_t) c->top_k);
        const int P = n * c->top_k;
        experts(r, T_(r, "model.layers.%d.mlp.experts.gate_proj.weight", l), inds, P, c->top_k, xn, d, G);
        experts(r, T_(r, "model.layers.%d.mlp.experts.up_proj.weight", l), inds, P, c->top_k, xn, d, U);
        for (size_t i = 0; i < (size_t) P * c->ff_exp; ++i) A[i] = bfr(silu_bf(bfr(G[i])) * bfr(U[i]));
        experts(r, T_(r, "model.layers.%d.mlp.experts.down_proj.weight", l), inds, P, 1, A, c->ff_exp, D);
        {
            Lin L[2] = {{T_(r, "model.layers.%d.mlp.shared_experts.gate_proj.weight", l), 0, xn, n, ga, c->ff_exp, 0},
                        {T_(r, "model.layers.%d.mlp.shared_experts.up_proj.weight", l), 0, xn, n, ua, c->ff_exp, 0}};
            run_lins(r, L, 2);
        }
        for (size_t i = 0; i < (size_t) n * c->ff_exp; ++i) aa[i] = bfr(silu_bf(ga[i]) * ua[i]);
        lin1(r, T_(r, "model.layers.%d.mlp.shared_experts.down_proj.weight", l), 0, aa, n, sh, d, 0);
        for (int t = 0; t < n; ++t)
            for (int ci = 0; ci < d; ++ci) {
                float s = 0;
                for (int j = 0; j < c->top_k; ++j) s += bfr(D[((size_t) t * c->top_k + j) * d + ci] * bfr(wts[t * c->top_k + j]));
                const size_t i = (size_t) t * d + ci;
                x[i] = bfr(x[i] + bfr(bfr(s) + sh[i]));
            }
    }
    if (h0 < n) {
        gnorm(r, x + (size_t) h0 * d, ns_find(&r->nm, "model.norm.weight"), xn, n - h0);
        lin1(r, ns_find(&r->nm, "lm_head.weight"), 0, xn, n - h0, logits, c->vocab, 0);
    }
    free(x); free(xn); free(q); free(g); free(o); free(k); free(v); free(ga); free(ua); free(aa); free(sh);
    free(G); free(U); free(A); free(D); free(V); free(inds); free(wts);
    return 0;
}
