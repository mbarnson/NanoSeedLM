// tests/test_qwen4_kernels.c - the qwen4 kernels against nslm/qwen4_ops.h on each GPU backend that has them.
// f32 kernels: within 1e-4 of the largest reference value; Q4 group 32 within BF16 rounding.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_backend.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static unsigned g_seed = 7;
static float frand(float lo, float hi) { g_seed = g_seed * 1103515245u + 12345u; return lo + (hi - lo) * (float) ((g_seed >> 8) & 0xFFFF) / 65536.0f; }
static float* rnd(size_t n, float lo, float hi) {
    float* v = (float*) malloc(sizeof(float) * (n ? n : 1));
    for (size_t i = 0; i < n; ++i) v[i] = frand(lo, hi);
    return v;
}
static float* fdup(const float* v, size_t n) { float* d = (float*) malloc(sizeof(float) * (n ? n : 1)); memcpy(d, v, sizeof(float) * n); return d; }
// max |a - b| over the largest |b|
static double rel(const float* a, const float* b, size_t n) {
    double d = 0, m = 1e-30;
    for (size_t i = 0; i < n; ++i) { d = fmax(d, fabs((double) a[i] - b[i])); m = fmax(m, fabs((double) b[i])); }
    return d / m;
}

static void gdn_case(int hk, int hv, int dk, int dv, int T) {
    Qwen4Cfg c;
    memset(&c, 0, sizeof c);
    c.lin_hk = hk; c.lin_hv = hv; c.lin_dk = dk; c.lin_dv = dv; c.lin_conv = 4; c.eps = 1e-6f;
    const size_t cd = (size_t) 2 * hk * dk + (size_t) hv * dv, vd = (size_t) hv * dv;
    const size_t nc = qwen4_gdn_conv_len(&c), ns = qwen4_gdn_state_len(&c);
    float *cw = rnd(cd * 4, -0.5f, 0.5f), *al = rnd((size_t) hv, -1, 1), *dt = rnd((size_t) hv, -1, 1), *nw = rnd((size_t) dv, 0.5f, 1.5f);
    const Qwen4GdnW w = {cw, al, dt, nw};
    float *qkv = rnd(cd * T, -1, 1), *z = rnd(vd * T, -2, 2), *a = rnd((size_t) hv * T, -2, 2), *b = rnd((size_t) hv * T, -2, 2);
    float *conv = rnd(nc, -1, 1), *S = rnd(ns, -0.5f, 0.5f);
    float *conv_k = fdup(conv, nc), *S_k = fdup(S, ns), *y = rnd(vd * T, 0, 0), *y_k = rnd(vd * T, 0, 0);
    for (int t = 0; t < T; ++t) qwen4_gdn_token(&c, &w, qkv + cd * t, z + vd * t, a + (size_t) hv * t, b + (size_t) hv * t, conv, S, y + vd * t);
    const int rc = kt_q4_gdn(&c, &w, qkv, z, a, b, conv_k, S_k, y_k, T);
    if (rc == 1) printf("  gdn: not on %s\n", kt_name());
    else {
        CHECK(rc == 0, "gdn %d/%d heads, %d/%d wide: kernel failed", hk, hv, dk, dv);
        const double ey = rel(y_k, y, vd * T), es = rel(S_k, S, ns), ec = rel(conv_k, conv, nc);
        printf("  gdn hk %d hv %d dk %d dv %d, %d tokens: y %.1e, state %.1e, conv %.1e\n", hk, hv, dk, dv, T, ey, es, ec);
        CHECK(ey < 1e-4 && es < 1e-4 && ec == 0, "gdn %d/%d heads, %d/%d wide: outputs differ", hk, hv, dk, dv);
    }
    free(cw); free(al); free(dt); free(nw); free(qkv); free(z); free(a); free(b); free(conv); free(S); free(conv_k); free(S_k); free(y); free(y_k);
}

static void hc_case(int hc, int d, int rank, int T) {
    Qwen4Cfg c;
    memset(&c, 0, sizeof c);
    c.hc = hc; c.d = d; c.hc_rank = rank; c.eps = 1e-6f;
    const size_t HD = (size_t) hc * d;
    float *x = rnd(HD * T, -3, 3), *w = rnd(HD, -0.3f, 0.3f), *y = rnd(HD * T, 0, 0), *y_k = rnd(HD * T, 0, 0);
    for (int t = 0; t < T; ++t) qwen4_norm1p(x + HD * t, (int) HD, d, w, c.eps, y + HD * t);
    int rc = kt_q4_norm1p(x, (int) HD, d, w, c.eps, y_k, T);
    if (rc == 1) printf("  norm1p: not on %s\n", kt_name());
    else {
        CHECK(rc == 0, "norm1p failed");
        printf("  norm1p %zu in groups of %d, %d tokens: %.1e\n", HD, d, T, rel(y_k, y, HD * T));
        CHECK(rel(y_k, y, HD * T) < 1e-5, "norm1p differs");
    }
    float *lo = rnd((size_t) rank * T, -4, 4), *up = rnd(HD * T, -4, 4), *inj = rnd((size_t) hc * T, -4, 4), *H = rnd(HD * T, -1, 1);
    float *br = rnd((size_t) d * T, -1, 1), *mix = rnd((size_t) d * T, 0, 0);
    float *lo_k = fdup(lo, (size_t) rank * T), *inj_k = fdup(inj, (size_t) hc * T), *H_k = fdup(H, HD * T), *mix_k = fdup(mix, (size_t) d * T);
    for (int t = 0; t < T; ++t) {
        qwen4_hc_act(&c, lo + (size_t) rank * t);
        qwen4_hc_mix(&c, y + HD * t, up + HD * t, mix + (size_t) d * t);
        qwen4_hc_gate(&c, inj + (size_t) hc * t);
        qwen4_hc_add(&c, H + HD * t, br + (size_t) d * t, inj + (size_t) hc * t);
    }
    rc = kt_q4_hc(&c, lo_k, y, up, mix_k, inj_k, H_k, br, T);
    if (rc == 1) printf("  hyper-connections: not on %s\n", kt_name());
    else {
        CHECK(rc == 0, "hyper-connection steps failed");
        const double e[4] = {rel(lo_k, lo, (size_t) rank * T), rel(mix_k, mix, (size_t) d * T), rel(inj_k, inj, (size_t) hc * T), rel(H_k, H, HD * T)};
        printf("  hyper-connections hc %d d %d rank %d: act %.1e, mix %.1e, gate %.1e, add %.1e\n", hc, d, rank, e[0], e[1], e[2], e[3]);
        CHECK(e[0] < 1e-5 && e[1] < 1e-5 && e[2] < 1e-5 && e[3] < 1e-5, "hyper-connection steps differ");
    }
    free(x); free(w); free(y); free(y_k); free(lo); free(up); free(inj); free(H); free(br); free(mix); free(lo_k); free(inj_k); free(H_k); free(mix_k);
}

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

// MLX affine Q4 with group 32 (the 4-bit qwen4 pack's projections) through the matvec (T <= 8) and the GEMM (T = 16):
// within BF16 rounding of the exact s * q + b products
static void q4g32_case(int R, int K) {
    const size_t n = (size_t) R * K, ng = n / 32;
    uint32_t* q = (uint32_t*) calloc(n / 8, 4);
    uint16_t *sc = (uint16_t*) malloc(2 * ng), *bi = (uint16_t*) malloc(2 * ng);
    for (size_t i = 0; i < n; ++i) q[i / 8] |= (uint32_t) (frand(0, 16) > 15.99f ? 15 : (int) frand(0, 16)) << (4 * (i % 8));
    for (size_t g = 0; g < ng; ++g) { sc[g] = f2bf(frand(0.005f, 0.05f)); bi[g] = f2bf(-8 * bf2f(sc[g]) + frand(-0.01f, 0.01f)); }
    KtWeight w;
    memset(&w, 0, sizeof w);
    w.fmt = MF_Q4; w.S = 1; w.R = R; w.K = K; w.w = q; w.s = sc; w.b = bi; w.wn = n / 2; w.sn = w.bn = 2 * ng; w.group = 32;
    const int Ts[2] = {3, 16};
    for (int c = 0; c < 2; ++c) {
        const int T = Ts[c];
        float *x = rnd((size_t) T * K, -1, 1), *y = rnd((size_t) T * R, 0, 0), *y_k = rnd((size_t) T * R, 0, 0);
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < R; ++r) {
                double acc = 0;
                for (int k = 0; k < K; ++k) {
                    const size_t i = (size_t) r * K + k;
                    acc += ((double) bf2f(sc[i / 32]) * ((q[i / 8] >> (4 * (i % 8))) & 15u) + bf2f(bi[i / 32])) * x[(size_t) t * K + k];
                }
                y[(size_t) t * R + r] = (float) acc;
            }
        const int rc = T <= 8 ? kt_mv(&w, T, 0, x, y_k) : kt_mm(&w, T, x, y_k);
        if (rc == 1) printf("  q4 group 32: not on %s\n", kt_name());
        else {
            CHECK(rc == 0, "q4 group 32 %s failed", T <= 8 ? "matvec" : "GEMM");
            printf("  q4 group 32 %s %d x %d, %d tokens: %.1e\n", T <= 8 ? "matvec" : "GEMM", R, K, T, rel(y_k, y, (size_t) T * R));
            CHECK(rel(y_k, y, (size_t) T * R) < 1e-2, "q4 group 32 %s differs", T <= 8 ? "matvec" : "GEMM");
        }
        free(x); free(y); free(y_k);
    }
    free(q); free(sc); free(bi);
}

static int cmp_i32(const void* a, const void* b) { return *(const int32_t*) a - *(const int32_t*) b; }
static Qwen4Cfg big_cfg(void) {   // the Qwen3.8-Flash-Next sizes the kernels below need
    Qwen4Cfg c;
    memset(&c, 0, sizeof c);
    c.d = 2560; c.hc = 4; c.eps = 1e-6f; c.rot_dim = 64; c.rope_theta = 10000000.0f; c.n_head = 24; c.n_kv = 2; c.head_dim = 256;
    c.ix_heads = 4; c.ix_dim = 128; c.ix_ratio = 4; c.ix_budget = 32; c.n_exp = 512; c.top_k = 10; c.ple_conv = 4; c.ngram = 3;
    return c;
}

static void norm_rope_route_case(void) {
    const Qwen4Cfg c = big_cfg();
    const int T = 3, rs = c.n_head * 2 * c.head_dim;
    float *x = rnd((size_t) rs * T, -2, 2), *w = rnd((size_t) c.head_dim, -0.3f, 0.3f), *x_k = fdup(x, (size_t) rs * T);
    for (int t = 0; t < T; ++t) qwen4_head_norm_rope(&c, x + (size_t) rs * t, c.n_head, 2 * c.head_dim, c.head_dim, w, 70000 + t);
    int rc = kt_q4_head_norm_rope(&c, x_k, rs, c.n_head, 2 * c.head_dim, c.head_dim, w, 70000, T);
    if (rc == 1) printf("  head norm + RoPE: not on %s\n", kt_name());
    else {
        printf("  head norm + RoPE, 24 heads of 256 (rot 64) at 70000: %.1e\n", rel(x_k, x, (size_t) rs * T));
        CHECK(rc == 0 && rel(x_k, x, (size_t) rs * T) < 1e-5, "head norm + RoPE differs");
    }
    float* lg = rnd((size_t) c.n_exp * T, -4, 4);
    lg[7] = lg[300] = 9.0f;   // a tie: the higher expert first
    int32_t ind[3 * 10], ind_k[3 * 10];
    float wt[3 * 10], wt_k[3 * 10];
    for (int t = 0; t < T; ++t) qwen4_route(&c, lg + (size_t) c.n_exp * t, ind + c.top_k * t, wt + c.top_k * t);
    rc = kt_q4_route(&c, lg, ind_k, wt_k, T);
    if (rc == 1) printf("  router: not on %s\n", kt_name());
    else {
        CHECK(rc == 0 && !memcmp(ind, ind_k, sizeof ind) && ind[0] == 300, "router experts differ");
        printf("  router 512 experts, top 10: experts %s, weights %.1e\n", memcmp(ind, ind_k, sizeof ind) ? "differ" : "equal",
               rel(wt_k, wt, (size_t) c.top_k * T));
        CHECK(rel(wt_k, wt, (size_t) c.top_k * T) < 1e-5, "router weights differ");
    }
    free(x); free(w); free(x_k); free(lg);
}

static void ple_case(void) {
    const Qwen4Cfg c = big_cfg();
    const int T = 5;
    const size_t HD = (size_t) c.hc * c.d, nh = qwen4_ple_hist_len(&c);
    float *keys = rnd(HD * T, -1, 1), *qs = rnd(HD * T, -1, 1), *vals = rnd((size_t) c.d * T, -1, 1), *w = rnd(HD * c.ple_conv, -0.5f, 0.5f);
    float *cin = rnd(HD * T, -1, 1), *hist = rnd(nh, -1, 1), *H = rnd(HD * T, -1, 1), *g = rnd(HD * T, 0, 0);
    float *hist_k = fdup(hist, nh), *H_k = fdup(H, HD * T), *g_k = rnd(HD * T, 0, 0);
    for (int t = 0; t < T; ++t) {
        qwen4_ple_gate(&c, keys + HD * t, qs + HD * t, vals + (size_t) c.d * t, g + HD * t);
        qwen4_ple_conv(&c, w, cin + HD * t, g + HD * t, hist, H + HD * t);
    }
    const int rc = kt_q4_ple(&c, keys, qs, vals, g_k, w, cin, hist_k, H_k, T);
    if (rc == 1) printf("  PLE gate + conv: not on %s\n", kt_name());
    else {
        const double e[3] = {rel(g_k, g, HD * T), rel(hist_k, hist, nh), rel(H_k, H, HD * T)};
        printf("  PLE gate + conv, %d tokens: gate %.1e, history %.1e, H %.1e\n", T, e[0], e[1], e[2]);
        CHECK(rc == 0 && e[0] < 1e-5 && e[1] == 0 && e[2] < 1e-5, "PLE gate + conv differs");
    }
    free(keys); free(qs); free(vals); free(w); free(cin); free(hist); free(H); free(g); free(hist_k); free(H_k); free(g_k);
}

static void qsa_attend_case(void) {
    const Qwen4Cfg c = big_cfg();
    const int nb = 40, D = c.ix_dim, k = c.ix_budget / c.ix_ratio, T = 3, rs = (c.ix_heads + 1) * D;
    float *raw = rnd((size_t) nb * c.ix_ratio * D, -1, 1), *w = rnd((size_t) D, -0.3f, 0.3f), *blk = rnd((size_t) nb * D, 0, 0);
    float *blk_k = rnd((size_t) nb * D, 0, 0), *iq = rnd((size_t) rs * T, -1, 1);
    for (int j = 0; j < nb; ++j) qwen4_qsa_block(&c, raw + (size_t) j * c.ix_ratio * D, w, j, blk + (size_t) j * D);
    const int32_t done[3] = {40, 20, 9};
    int32_t sel[3 * 8], sel_k[3 * 8];
    int ns[3];
    for (int t = 0; t < T; ++t) ns[t] = qwen4_qsa_select(&c, iq + (size_t) rs * t, blk, done[t], sel + k * t);
    int rc = kt_q4_qsa(&c, raw, w, nb, blk_k, iq, rs, done, sel_k, T);
    if (rc == 1) printf("  QSA blocks + select: not on %s\n", kt_name());
    else {
        int same = 1;
        for (int t = 0; t < T; ++t) {
            qsort(sel + k * t, (size_t) k, 4, cmp_i32);
            qsort(sel_k + k * t, (size_t) k, 4, cmp_i32);
            same &= ns[t] == k && !memcmp(sel + k * t, sel_k + k * t, 4 * (size_t) k);
        }
        printf("  QSA 40 block keys: %.1e; top 8 of 40 / 20 / 9 blocks: %s\n", rel(blk_k, blk, (size_t) nb * D), same ? "equal" : "differ");
        CHECK(rc == 0 && rel(blk_k, blk, (size_t) nb * D) < 1e-5 && same, "QSA differs");
    }
    free(raw); free(w); free(blk); free(blk_k); free(iq);
    for (int AT = 3; AT <= 12; AT += 9) {   // decode rows, and prompt rows (more than 8: a block per KV group)
        const int npos = 112, pos0 = 97, no = c.n_head * c.head_dim;
        const size_t kv = (size_t) npos * c.n_kv * c.head_dim;
        float *q = rnd((size_t) 2 * no * AT, -1, 1), *K = rnd(kv, -1, 1), *V = rnd(kv, -1, 1), *o = rnd((size_t) no * AT, 0, 0);
        float *o_k = rnd((size_t) no * AT, 0, 0);
        int32_t asel[12 * 8], ansel[12];
        for (int t = 0; t < AT; ++t) ansel[t] = t % 2 ? 0 : k;   // sparse and dense rows in turn
        for (int i = 0; i < AT * k; ++i) asel[i] = (i * 7 + i / k) % 24;   // distinct complete blocks of the sparse rows
        for (int t = 0; t < AT; ++t) qwen4_attend(&c, q + (size_t) 2 * no * t, K, V, pos0 + t, asel + k * t, ansel[t], o + (size_t) no * t);
        rc = kt_q4_attend(&c, q, K, V, npos, pos0, asel, ansel, o_k, AT);
        if (rc == 1) printf("  attention: not on %s\n", kt_name());
        else {
            printf("  attention 24 heads of 256, 2 KV heads, %d sparse and dense rows: %.1e\n", AT, rel(o_k, o, (size_t) no * AT));
            CHECK(rc == 0 && rel(o_k, o, (size_t) no * AT) < 1e-5, "attention differs at %d rows", AT);
        }
        free(q); free(K); free(V); free(o); free(o_k);
    }
}

int main(void) {
    char err[256];
    if (kt_open(err, sizeof err)) { printf("SKIP: %s\n", err); return 77; }
    printf("test_qwen4_kernels on %s\n", kt_name());
    gdn_case(16, 48, 128, 128, 6);   // Qwen3.8-Flash-Next
    gdn_case(16, 48, 128, 128, 40);  // tokens in more than one stage of the step kernel
    gdn_case(4, 8, 64, 64, 3);
    gdn_case(2, 4, 32, 32, 9);       // tests/data/qwen4_tiny
    hc_case(4, 2560, 320, 3);        // Qwen3.8-Flash-Next
    hc_case(4, 64, 32, 5);
    q4g32_case(96, 2560);
    norm_rope_route_case();
    ple_case();
    qsa_attend_case();
    kt_close();
    printf("test_qwen4_kernels: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
