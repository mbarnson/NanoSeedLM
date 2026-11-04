// tests/test_qwen4_kernels.c - the Qwen3.8 kernels against nslm/qwen4_ops.h, on every GPU backend that has them
// (tests/kernel_backend.h): Gated DeltaNet over several tokens at three head widths, the (1 + w) RMSNorm and the
// hyper-connection steps, and affine Q4 with group 32.  f32 kernels: outputs within 1e-4 of the largest reference value.
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
static float* dup(const float* v, size_t n) { float* d = (float*) malloc(sizeof(float) * (n ? n : 1)); memcpy(d, v, sizeof(float) * n); return d; }
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
    float *conv_k = dup(conv, nc), *S_k = dup(S, ns), *y = rnd(vd * T, 0, 0), *y_k = rnd(vd * T, 0, 0);
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
    float *lo_k = dup(lo, (size_t) rank * T), *inj_k = dup(inj, (size_t) hc * T), *H_k = dup(H, HD * T), *mix_k = dup(mix, (size_t) d * T);
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

// MLX affine Q4 with group 32 (the 4-bit Qwen3.8 pack's projections) through the matvec (T <= 8) and the GEMM (T = 16):
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

int main(void) {
    char err[256];
    if (kt_open(err, sizeof err)) { printf("SKIP: %s\n", err); return 77; }
    printf("test_qwen4_kernels on %s\n", kt_name());
    gdn_case(16, 48, 128, 128, 6);   // Qwen3.8-Flash-Next
    gdn_case(4, 8, 64, 64, 3);
    gdn_case(2, 4, 32, 32, 9);       // tests/data/qwen4_tiny
    hc_case(4, 2560, 320, 3);        // Qwen3.8-Flash-Next
    hc_case(4, 64, 32, 5);
    q4g32_case(96, 2560);
    kt_close();
    printf("test_qwen4_kernels: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
