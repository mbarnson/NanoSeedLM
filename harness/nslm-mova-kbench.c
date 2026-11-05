// harness/nslm-mova-kbench.c - micro-benchmark of the CUDA MoVA kernels (engine/kernels_moe.cu) at MoVA's shapes: GPU
// time per call and effective bandwidth (weight bytes read per second).  Random weights; timing only.  The Metal build
// has its own (harness/nslm-mova-kbench.m): the kernels differ, the shapes and the report are the same.  The CUDA build
// also times the decode expert gathers at Qwen3.8-Flash-Next's shapes, in its three expert formats.
//
//   nslm-mova-kbench [--iters 200] [--pos 4096]
//
// Time per call = CUDA event time over --iters back-to-back launches, best of 3, after a warm-up.  Gather cases rotate
// through the slices so successive calls do not hit the same expert in L2.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "kernels_cuda.h"
#include "lfsr.h"
#include "platform.h"

static void* dev_rand(size_t bytes, unsigned seed) {
    uint8_t* h = (uint8_t*) malloc(bytes);
    uint32_t s = seed * 2654435761u + 1;
    for (size_t i = 0; i < bytes; ++i) { s = s * 1664525u + 1013904223u; h[i] = (uint8_t) (s >> 24); }
    void* d = NULL;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    free(h);
    return d;
}
// BF16 scales of a moderate size (random bits would give infinities and NaNs)
static void* dev_scales(size_t n) {
    uint16_t* h = (uint16_t*) malloc(n * 2);
    for (size_t i = 0; i < n; ++i) h[i] = nslm_f2bf(0.001f * (float) (1 + i % 7));
    void* d = NULL;
    cudaMalloc(&d, n * 2);
    cudaMemcpy(d, h, n * 2, cudaMemcpyHostToDevice);
    free(h);
    return d;
}
static float* dev_act(size_t n) {   // BF16-valued activations
    float* h = (float*) malloc(n * 4);
    for (size_t i = 0; i < n; ++i) h[i] = nslm_bf2f(nslm_f2bf((float) ((int) (i % 2001) - 1000) / 1000.0f));
    float* d = NULL;
    cudaMalloc((void**) &d, n * 4);
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    free(h);
    return d;
}

// A stacked tensor of `slices` x R x K in one format, as a device slice table.
typedef struct {
    WSlice* d;
    WSlice h0;
    double bytes_per_slice;
} Tensor;
// group: the affine group of Q4 / Q8 (64 or 32)
static Tensor make_tensor_g(int fmt, int slices, int R, int K, unsigned seed, int group) {
    Tensor t;
    memset(&t, 0, sizeof t);
    WSlice* h = (WSlice*) calloc((size_t) slices, sizeof(WSlice));
    const size_t n = (size_t) R * K, ng = n / (size_t) group;
    if (fmt == MF_SEED4) {   // P = 3: seeds, nibble words, an exponent bias for each slice
        uint8_t* s = (uint8_t*) dev_rand(n / 8 * 2 * slices, seed), *c = (uint8_t*) dev_rand(n / 8 * 2 * slices, seed + 1);
        for (int i = 0; i < slices; ++i) { h[i].p[0] = s + n / 8 * 2 * i; h[i].p[1] = c + n / 8 * 2 * i; h[i].eb = -24; }
        t.bytes_per_slice = (double) n / 8 * 4;
    } else if (fmt == MF_SEED4P4) {
        uint8_t* s = (uint8_t*) dev_rand(n / 8 * 2 * slices, seed), *c = (uint8_t*) dev_rand(n / 8 * 2 * slices, seed + 1);
        uint8_t* e = (uint8_t*) dev_rand(n / 16 * slices, seed + 2);
        for (int i = 0; i < slices; ++i) {
            h[i].p[0] = s + n / 8 * 2 * i; h[i].p[1] = c + n / 8 * 2 * i; h[i].p[3] = e + n / 16 * i; h[i].eb = -24;
        }
        t.bytes_per_slice = (double) n / 8 * 4 + (double) n / 16;
    } else if (fmt == MF_Q4 || fmt == MF_Q8) {
        const size_t wb = fmt == MF_Q4 ? n / 2 : n;
        uint8_t* w = (uint8_t*) dev_rand(wb * slices, seed);
        uint8_t* sc = (uint8_t*) dev_scales(ng * slices), *bi = (uint8_t*) dev_scales(ng * slices);
        for (int i = 0; i < slices; ++i) {
            h[i].p[0] = w + wb * i; h[i].p[1] = sc + ng * 2 * i; h[i].p[2] = bi + ng * 2 * i; h[i].gs = group == 32 ? 5 : 6;
        }
        t.bytes_per_slice = (double) wb + (double) ng * 4;
    }
    cudaMalloc((void**) &t.d, sizeof(WSlice) * (size_t) slices);
    cudaMemcpy(t.d, h, sizeof(WSlice) * (size_t) slices, cudaMemcpyHostToDevice);
    t.h0 = h[0];
    free(h);
    return t;
}
static Tensor make_tensor(int fmt, int slices, int R, int K, unsigned seed) { return make_tensor_g(fmt, slices, R, K, seed, 64); }

typedef void (*Body)(void* ctx, int it);
static double time_us(Body f, void* ctx, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 10; ++i) f(ctx, i);
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        cudaEventRecord(a, 0);
        for (int i = 0; i < iters; ++i) f(ctx, i);
        cudaEventRecord(b, 0);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (ms * 1e3 / iters < best) best = ms * 1e3 / iters;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

typedef struct {
    int fmt, R, K, P, slices, xdiv;
    Tensor w, u;
    float *x, *y;
    int32_t* sel;   // [64][P]: rotating selections
    const uint32_t* G;
} Case;
static void run_dense(void* c_, int it) {
    Case* c = (Case*) c_;
    (void) it;
    kc_mv(0, c->fmt, c->w.h0, c->K, c->R, c->x, c->K, c->y, c->R, c->P, 0, c->G);
}
static void run_gather(void* c_, int it) {
    Case* c = (Case*) c_;
    kc_mv_sel(0, c->fmt, c->w.d, c->K, c->R, c->x, c->K, c->y, c->R, c->sel + (it % 64) * c->P, c->P, c->xdiv, c->G);
}
static void run_gu(void* c_, int it) {
    Case* c = (Case*) c_;
    kc_mv_gu(0, c->fmt, c->w.d, c->u.d, c->K, c->R, c->x, c->K, c->y, c->R, c->sel + (it % 64) * c->P, c->P, c->xdiv, c->G);
}
static void run_mm(void* c_, int it) {
    Case* c = (Case*) c_;
    (void) it;
    kc_mm(0, c->fmt, c->w.h0, c->K, c->R, c->x, c->K, c->y, c->R, c->P, 0, c->G);
}

typedef struct {
    AttnArgs a;
    float *q, *part, *g, *o;
    KvView kv;
    RowInfo* ri;
} AttnCase;
static void run_attn(void* c_, int it) {
    AttnCase* c = (AttnCase*) c_;
    (void) it;
    kc_attn(0, c->a, c->q, c->kv, c->ri, c->part, c->g, c->o, 1);
}

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const int iters = atoi(opt(argc, argv, "--iters", "200")), pos = atoi(opt(argc, argv, "--pos", "4096"));
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { fprintf(stderr, "no CUDA device\n"); return 1; }
    printf("device=%s\n", prop.name);
    uint32_t* g32h = (uint32_t*) malloc(65536 * 4);
    for (uint32_t s = 0; s < 65536; ++s) g32h[s] = lfsr_stream32((uint16_t) s);
    uint32_t* G = NULL;
    cudaMalloc((void**) &G, 65536 * 4);
    cudaMemcpy(G, g32h, 65536 * 4, cudaMemcpyHostToDevice);
    free(g32h);
    float* x = dev_act((size_t) 512 * 6144);
    float* y = NULL;
    cudaMalloc((void**) &y, (size_t) 512 * 250624 * 4 > (size_t) 64 * 250624 * 4 ? (size_t) 64 * 250624 * 4 : 0);
    cudaFree(y);
    cudaMalloc((void**) &y, (size_t) 64 * 250624 * 4);
    int32_t* selh = (int32_t*) malloc(64 * 8 * 4);
    for (int i = 0; i < 64 * 8; ++i) selh[i] = (i * 37 + i / 8 * 11) % 100;
    int32_t* sel = NULL;
    cudaMalloc((void**) &sel, 64 * 8 * 4);
    cudaMemcpy(sel, selh, 64 * 8 * 4, cudaMemcpyHostToDevice);
    int32_t* selv = NULL;   // value experts: 64 slices, 4 per token
    for (int i = 0; i < 64 * 8; ++i) selh[i] = (i * 29 + i / 4 * 7) % 64;
    cudaMalloc((void**) &selv, 64 * 8 * 4);
    cudaMemcpy(selv, selh, 64 * 8 * 4, cudaMemcpyHostToDevice);
    free(selh);

    printf("%-44s %10s %10s\n", "kernel (shape)", "us/call", "GB/s");
    const struct { const char* name; int fmt, R, K; } dense[] = {
        {"k_mv q8 q_proj (4096 x 2560)", MF_Q8, 4096, 2560},
        {"k_mv q8 o_proj (2560 x 4096)", MF_Q8, 2560, 4096},
        {"k_mv q8 k_proj (1024 x 2560)", MF_Q8, 1024, 2560},
        {"k_mv q8 shared down (2560 x 768)", MF_Q8, 2560, 768},
        {"k_mv q8 lm_head (250624 x 2560)", MF_Q8, 250624, 2560},
    };
    for (size_t i = 0; i < sizeof dense / sizeof dense[0]; ++i) {
        Case c;
        memset(&c, 0, sizeof c);
        c.fmt = dense[i].fmt; c.R = dense[i].R; c.K = dense[i].K; c.P = 1; c.x = x; c.y = y; c.G = G;
        c.w = make_tensor(c.fmt, 1, c.R, c.K, 3 + (unsigned) i);
        const double us = time_us(run_dense, &c, iters);
        printf("%-44s %10.2f %10.1f\n", dense[i].name, us, c.w.bytes_per_slice / us / 1e3);
    }
    {   // routed experts: 8 (token, expert) pairs of 100 experts; gate + up fused, then down
        Case c;
        memset(&c, 0, sizeof c);
        c.fmt = MF_SEED4P4; c.R = 768; c.K = 2560; c.P = 8; c.slices = 100; c.xdiv = 8; c.x = x; c.y = y; c.sel = sel; c.G = G;
        c.w = make_tensor(MF_SEED4P4, 100, 768, 2560, 20);
        c.u = make_tensor(MF_SEED4P4, 100, 768, 2560, 21);
        double us = time_us(run_gu, &c, iters);
        printf("%-44s %10.2f %10.1f\n", "k_mv_gu seed4p4 gate+up (8 x 768 x 2560)", us, 2 * 8 * c.w.bytes_per_slice / us / 1e3);
        Case d = c;
        d.R = 2560; d.K = 768; d.xdiv = 1;
        d.w = make_tensor(MF_SEED4P4, 100, 2560, 768, 22);
        us = time_us(run_gather, &d, iters);
        printf("%-44s %10.2f %10.1f\n", "k_mv seed4p4 down (8 x 2560 x 768)", us, 8 * d.w.bytes_per_slice / us / 1e3);
        Case v = c;
        v.fmt = MF_Q4; v.R = 1024; v.K = 2560; v.P = 4; v.xdiv = 4; v.sel = selv;
        v.w = make_tensor(MF_Q4, 64, 1024, 2560, 23);
        us = time_us(run_gather, &v, iters);
        printf("%-44s %10.2f %10.1f\n", "k_mv q4 value experts (4 x 1024 x 2560)", us, 4 * v.w.bytes_per_slice / us / 1e3);
    }
    {   // decode attention at --pos keys, BF16 KV in VRAM
        AttnCase c;
        memset(&c, 0, sizeof c);
        int ns = (pos + 127) / 128;
        if (ns > 32) ns = 32;
        AttnArgs a = {32, 8, 128, ns, 0.0883883f};
        c.a = a;
        c.q = dev_act(32 * 128);
        c.g = dev_act(32 * 128);
        cudaMalloc((void**) &c.o, 32 * 128 * 4);
        cudaMalloc((void**) &c.part, (size_t) 32 * 32 * 130 * 4);
        c.kv.fmt = KV_BF16;
        c.kv.nv = pos + 1;
        c.kv.a.k = dev_scales((size_t) (pos + 1) * 1024);
        c.kv.a.v = dev_scales((size_t) (pos + 1) * 1024);
        RowInfo ri;
        memset(&ri, 0, sizeof ri);
        ri.pos = pos;
        cudaMalloc((void**) &c.ri, sizeof ri);
        cudaMemcpy(c.ri, &ri, sizeof ri, cudaMemcpyHostToDevice);
        const double us = time_us(run_attn, &c, iters);
        char nm[64];
        snprintf(nm, sizeof nm, "k_attn + reduce, decode (%d keys)", pos);
        printf("%-44s %10.2f %10.1f\n", nm, us, (double) (pos + 1) * 1024 * 2 * 2 / us / 1e3);
    }
    {   // prefill GEMM: 512 tokens through a q8 projection
        Case c;
        memset(&c, 0, sizeof c);
        c.fmt = MF_Q8; c.R = 4096; c.K = 2560; c.P = 512; c.x = x; c.y = y; c.G = G;
        c.w = make_tensor(MF_Q8, 1, 4096, 2560, 30);
        const double us = time_us(run_mm, &c, iters / 4 > 10 ? iters / 4 : 10);
        printf("%-44s %10.2f %10.1f  (%.1f TFLOPS)\n", "k_mm q8 q_proj, 512 tokens (4096 x 2560)", us, c.w.bytes_per_slice / us / 1e3,
               2.0 * 512 * 4096 * 2560 / us / 1e6);
    }
    {   // Qwen3.8-Flash-Next decode: 10 (token, expert) pairs of 512 experts, gate + up fused (640 x 2560), then down
        uint32_t* g24h = (uint32_t*) malloc(65536 * 4);
        for (uint32_t s = 0; s < 65536; ++s) g24h[s] = lfsr_stream24((uint16_t) s);
        uint32_t* G24 = NULL;
        cudaMalloc((void**) &G24, 65536 * 4);
        cudaMemcpy(G24, g24h, 65536 * 4, cudaMemcpyHostToDevice);
        free(g24h);
        int32_t* s10h = (int32_t*) malloc(64 * 10 * 4);
        for (int i = 0; i < 64 * 10; ++i) s10h[i] = (i * 97 + i / 10 * 13) % 512;
        int32_t* sel10 = NULL;
        cudaMalloc((void**) &sel10, 64 * 10 * 4);
        cudaMemcpy(sel10, s10h, 64 * 10 * 4, cudaMemcpyHostToDevice);
        free(s10h);
        const struct { const char* name; int fmt, group; } fm[] = {{"q4 g32", MF_Q4, 32}, {"seed4 (P3)", MF_SEED4, 0}, {"seed4p4 (P4)", MF_SEED4P4, 0}};
        for (int f = 0; f < 3; ++f) {
            Case c;
            memset(&c, 0, sizeof c);
            c.fmt = fm[f].fmt; c.R = 640; c.K = 2560; c.P = 10; c.slices = 512; c.xdiv = 10; c.x = x; c.y = y; c.sel = sel10;
            c.G = fm[f].fmt == MF_SEED4 ? G24 : G;
            c.w = make_tensor_g(c.fmt, 512, 640, 2560, 40 + (unsigned) f, fm[f].group ? fm[f].group : 64);
            c.u = make_tensor_g(c.fmt, 512, 640, 2560, 50 + (unsigned) f, fm[f].group ? fm[f].group : 64);
            char nm[64];
            double us = time_us(run_gu, &c, iters);
            snprintf(nm, sizeof nm, "qwen4 k_mv_gu %s (10 x 640 x 2560)", fm[f].name);
            printf("%-44s %10.2f %10.1f\n", nm, us, 2 * 10 * c.w.bytes_per_slice / us / 1e3);
            Case d = c;
            d.R = 2560; d.K = 640; d.xdiv = 1;
            d.w = make_tensor_g(d.fmt, 512, 2560, 640, 60 + (unsigned) f, fm[f].group ? fm[f].group : 64);
            us = time_us(run_gather, &d, iters);
            snprintf(nm, sizeof nm, "qwen4 k_mv down %s (10 x 2560 x 640)", fm[f].name);
            printf("%-44s %10.2f %10.1f\n", nm, us, 10 * d.w.bytes_per_slice / us / 1e3);
        }
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    return 0;
}
