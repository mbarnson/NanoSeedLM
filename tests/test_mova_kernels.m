// tests/test_mova_kernels.m - the MoVA engine's kernels (engine/kernels_moe.metal) against C scalar references.
// References compute in double with BF16 rounding at the kernels' points; outputs match within one BF16 ulp plus the
// f32 accumulation bound.  Seed formats are checked against the exact (unrounded) seed weights.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"
#include "kernels_moe.metal"
#include "lfsr.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;

static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static float bfr(double x) { return bf2f(f2bf((float) x)); }
static double frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 65536.0 * 2 - 1; }

static id<MTLBuffer> buf(const void* p, size_t n) {
    id<MTLBuffer> b = [dev newBufferWithLength:n > 0 ? n : 16 options:MTLResourceStorageModeShared];
    if (p) memcpy(b.contents, p, n);
    return b;
}
static id<MTLComputePipelineState> pipe_(const char* name, int fmt, int T) {
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    short f = (short) fmt, t = (short) T;
    [cv setConstantValue:&f type:MTLDataTypeShort atIndex:0];
    [cv setConstantValue:&t type:MTLDataTypeShort atIndex:1];
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&err];
    return [dev newComputePipelineStateWithFunction:fn error:&err];
}
typedef void (^Enc)(id<MTLComputeCommandEncoder>);
static void run(Enc enc) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc(e);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}
// |a - b| within one BF16 ulp of b, plus extra
static int close_bf(float a, double b, double extra) {
    const double ulp = fabs(b) * (1.0 / 128.0) + 1e-30;
    return fabs(a - b) <= ulp + extra;
}

// SEED4P4 (4.5-bit, P = 4) data for S slices of R x K: random seeds, coefficient words and exponent codes, the
// per-slice biases, the exponent nibbles (two blocks per byte, low first) and the exact weights by the decode spec
// (search4.h).
typedef struct { uint16_t* seeds, *coefs; uint8_t* nib; int32_t bias[3]; double* w; size_t nb; } P4;
static P4 p4_make(int S, int R, int K, unsigned* sd) {
    P4 d;
    d.nb = (size_t) S * R * K / 8;
    d.seeds = malloc(2 * d.nb); d.coefs = malloc(2 * d.nb); d.nib = calloc(d.nb / 2 + 1, 1); d.w = malloc(sizeof(double) * d.nb * 8);
    d.bias[0] = -19; d.bias[1] = -17; d.bias[2] = -21;
    for (size_t b = 0; b < d.nb; ++b) {
        *sd = *sd * 1103515245u + 12345u;
        d.seeds[b] = (uint16_t) (1 + (*sd >> 8) % 65535);
        *sd = *sd * 1103515245u + 12345u;
        d.coefs[b] = (uint16_t) (*sd >> 9);
        const int ec = (int) ((*sd >> 3) % 6);
        d.nib[b / 2] |= (uint8_t) (ec << (4 * (b & 1)));
        const int s = (int) (b / ((size_t) R * K / 8));
        uint16_t st[32];
        lfsr_states(d.seeds[b], 32, st);
        const double sc = (double) NSLM_R32 * pow(2.0, d.bias[s] + ec);
        for (int c = 0; c < 8; ++c) {
            double v = 0;
            for (int p = 0; p < 4; ++p) v += ((double) st[4 * c + p] - 32768) * (double) ((int32_t) ((uint32_t) d.coefs[b] << (28 - 4 * p)) >> 28);
            d.w[b * 8 + c] = sc * v;
        }
    }
    return d;
}
static id<MTLBuffer> g32_buffer(void) {
    uint32_t* g = malloc(65536 * 4);
    for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream32((uint16_t) s);
    id<MTLBuffer> b = buf(g, 65536 * 4);
    free(g);
    return b;
}

// ---- matvec: every format, dense T = 1..8 and gather ---------------------------------------------------------------------
static void test_mv(int K) {   // K = 256 and 512: the matvec's one- and two-blocks-per-lane seed paths
    const int R = 24, S = 3;   // stacked slices
    unsigned sd = 1;
    uint16_t* w = malloc(2 * (size_t) S * R * K);
    for (int i = 0; i < S * R * K; ++i) w[i] = f2bf((float) (frand(&sd) * 0.05));
    float x[MV_MAXT * K], xg[16 * K];
    for (int i = 0; i < MV_MAXT * K; ++i) x[i] = bfr(frand(&sd));
    for (int i = 0; i < 16 * K; ++i) xg[i] = bfr(frand(&sd));
    // encodings
    uint32_t* q8 = malloc((size_t) S * R * K), *q4 = malloc((size_t) S * R * K / 2);
    uint16_t* s8 = malloc(2 * (size_t) S * R * K / 64), *b8 = malloc(2 * (size_t) S * R * K / 64);
    uint16_t* s4 = malloc(2 * (size_t) S * R * K / 64), *b4 = malloc(2 * (size_t) S * R * K / 64);
    nslm_affine_quantize(w, S * R, K, 8, q8, s8, b8);
    nslm_affine_quantize(w, S * R, K, 4, q4, s4, b4);
    uint16_t* d8 = malloc(2 * (size_t) S * R * K), *d4 = malloc(2 * (size_t) S * R * K);
    nslm_affine_dequantize(q8, s8, b8, S * R, K, 8, d8);
    nslm_affine_dequantize(q4, s4, b4, S * R, K, 4, d4);
    // seeds: random seeds / nibbles; reference = the exact unrounded weights
    const int nb = S * R * K / 8;
    uint16_t* seeds = malloc(2 * (size_t) nb), *nibs = malloc(2 * (size_t) nb);
    int32_t ebias[3] = {-18, -16, -20};
    for (int i = 0; i < nb; ++i) { seeds[i] = (uint16_t) (1 + (sd = sd * 1103515245u + 12345u) % 65535); nibs[i] = (uint16_t) ((sd >> 3) & 0xFFFF); }
    double* wseed = malloc(sizeof(double) * (size_t) S * R * K);
    for (int s = 0; s < S; ++s)
        for (int b = 0; b < R * K / 8; ++b) {
            const int bi = s * R * K / 8 + b;
            uint16_t st[24];
            lfsr_states(seeds[bi], 24, st);
            const uint16_t nbw = nibs[bi];
            const double sc = (double) NSLM_R32 * pow(2.0, ebias[s] + nslm_ecode(nbw));
            for (int c = 0; c < 8; ++c)
                wseed[(size_t) bi * 8 + c] = sc * (((double) st[3 * c] - 32768) * nslm_q(nbw, 0) + ((double) st[3 * c + 1] - 32768) * nslm_q(nbw, 1) +
                                                   ((double) st[3 * c + 2] - 32768) * nslm_q(nbw, 2));
        }
    uint32_t* G = malloc(65536 * 4);
    for (uint32_t s = 0; s < 65536; ++s) G[s] = lfsr_stream24((uint16_t) s);
    id<MTLBuffer> gb = buf(G, 65536 * 4), g32 = g32_buffer();
    const P4 p4 = p4_make(S, R, K, &sd);
    const char* names[5] = {"bf16", "seed4", "q8", "q4", "seed4p4"};
    for (int fmt = 0; fmt < 5; ++fmt) {
        id<MTLBuffer> W = fmt == MF_BF16 ? buf(w, 2 * (size_t) S * R * K) : fmt == MF_Q8 ? buf(q8, (size_t) S * R * K)
                         : fmt == MF_Q4 ? buf(q4, (size_t) S * R * K / 2) : fmt == MF_SEED4P4 ? buf(p4.seeds, 2 * p4.nb) : buf(seeds, 2 * (size_t) nb);
        id<MTLBuffer> Sb = fmt == MF_Q8 ? buf(s8, 2 * (size_t) S * R * K / 64) : fmt == MF_Q4 ? buf(s4, 2 * (size_t) S * R * K / 64)
                          : fmt == MF_SEED4 ? buf(nibs, 2 * (size_t) nb) : fmt == MF_SEED4P4 ? buf(p4.coefs, 2 * p4.nb) : buf(NULL, 16);
        id<MTLBuffer> Bb = fmt == MF_Q8 ? buf(b8, 2 * (size_t) S * R * K / 64) : fmt == MF_Q4 ? buf(b4, 2 * (size_t) S * R * K / 64)
                          : fmt == MF_SEED4 ? buf(ebias, 12) : fmt == MF_SEED4P4 ? buf(p4.bias, 12) : buf(NULL, 16);
        id<MTLBuffer> Nb = fmt == MF_SEED4P4 ? buf(p4.nib, p4.nb / 2 + 1) : buf(NULL, 16);   // P = 4 exponent nibbles
        id<MTLBuffer> Gt = fmt == MF_SEED4P4 ? g32 : gb;
        // the reference weight of element (slice, row, col).  Q4 decode is MLX's qmv form: s * q + b unrounded, i.e.
        // y = sum over groups of (s * sum q x + b * sum x); the other formats dequantize to BF16 exactly as mx.dequantize.
        double (^wref)(int, int, int) = ^double(int s, int r, int c) {
            const size_t i = ((size_t) s * R + r) * K + c;
            if (fmt == MF_Q4) return (double) bf2f(s4[i / 64]) * (double) ((q4[i / 8] >> (4 * (i % 8))) & 15u) + bf2f(b4[i / 64]);
            if (fmt == MF_SEED4P4) return p4.w[i];
            return fmt == MF_BF16 ? bf2f(w[i]) : fmt == MF_Q8 ? bf2f(d8[i]) : wseed[i];
        };
        double worst = 0;
        for (int T = 1; T <= MV_MAXT; ++T) {   // dense on slice 0; also the residual add for T = 3
            const int add = T == 3;
            float y0[MV_MAXT * R];
            for (int i = 0; i < MV_MAXT * R; ++i) y0[i] = bfr(frand(&sd));
            id<MTLBuffer> xb = buf(x, sizeof x), yb = buf(y0, sizeof y0), sel = buf((int[]){0}, 4);
            MvArgs a = {K, R, T, 1, K, R, 0, add};
            id<MTLComputePipelineState> p = pipe_("k_mv", fmt, T);
            run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:p];
                [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:Sb offset:0 atIndex:2]; [e setBuffer:Bb offset:0 atIndex:3];
                [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:sel offset:0 atIndex:6];
                [e setBuffer:Gt offset:0 atIndex:7]; [e setBuffer:Nb offset:0 atIndex:8];
                [e dispatchThreadgroups:MTLSizeMake((R + MV_RPT(fmt) - 1) / MV_RPT(fmt), 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * MV_ROWS, 1, 1)];
            });
            const float* y = (const float*) yb.contents;
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < R; ++r) {
                    double acc = 0, mag = 0;
                    for (int c = 0; c < K; ++c) { acc += wref(0, r, c) * x[t * K + c]; mag += fabs(wref(0, r, c) * x[t * K + c]); }
                    const double ref = add ? bfr(y0[t * R + r] + bfr(acc)) : bfr(acc);
                    const double bound = (fmt == MF_SEED4 || fmt == MF_SEED4P4 ? 64 : 48) * 1.2e-7 * mag * (add ? 2 : 1) + 1e-30;
                    if (!close_bf(y[t * R + r], ref, bound)) {
                        ++fails;
                        if (fails < 10) printf("FAIL mv %s dense T=%d t=%d r=%d: %g vs %g\n", names[fmt], T, t, r, y[t * R + r], ref);
                    }
                    worst = fmax(worst, fabs(y[t * R + r] - ref) / (fabs(ref) / 128 + bound));
                }
        }
        {   // gather: 6 pairs, slices sel[p], inputs p / xdiv (xdiv 2)
            const int P = 6, xdiv = 2;
            int selv[6] = {2, 0, 1, 2, 1, 0};
            id<MTLBuffer> xb = buf(xg, sizeof xg), yb = buf(NULL, 4 * (size_t) P * R), sel = buf(selv, sizeof selv);
            MvArgs a = {K, R, P, xdiv, K, R, 0, 0};
            id<MTLComputePipelineState> p = pipe_("k_mv", fmt, 0);
            run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:p];
                [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:Sb offset:0 atIndex:2]; [e setBuffer:Bb offset:0 atIndex:3];
                [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:sel offset:0 atIndex:6];
                [e setBuffer:Gt offset:0 atIndex:7]; [e setBuffer:Nb offset:0 atIndex:8];
                [e dispatchThreadgroups:MTLSizeMake((R + MV_RPT(fmt) - 1) / MV_RPT(fmt), P, 1) threadsPerThreadgroup:MTLSizeMake(32 * MV_ROWS, 1, 1)];
            });
            const float* y = (const float*) yb.contents;
            for (int pp = 0; pp < P; ++pp)
                for (int r = 0; r < R; ++r) {
                    double acc = 0, mag = 0;
                    for (int c = 0; c < K; ++c) { const double v = wref(selv[pp], r, c) * xg[(pp / xdiv) * K + c]; acc += v; mag += fabs(v); }
                    const double ref = bfr(acc), bound = (fmt == MF_SEED4 || fmt == MF_SEED4P4 ? 64 : 48) * 1.2e-7 * mag;
                    if (!close_bf(y[pp * R + r], ref, bound)) {
                        ++fails;
                        if (fails < 10) printf("FAIL mv %s gather p=%d r=%d: %g vs %g\n", names[fmt], pp, r, y[pp * R + r], ref);
                    }
                    worst = fmax(worst, fabs(y[pp * R + r] - ref) / (fabs(ref) / 128 + bound));
                }
        }
        {   // k_mv_gu: gate and up over the selected experts + SwiGLU in one dispatch.  Gate = slice sel[p], up = the same
            // buffers one slice on (sel in {0, 1}); a = bf16(silu(bf16 g) * bf16 u), within one BF16 flip of g or u
            const int P = 6, xdiv = 2;
            int selv[6] = {1, 0, 1, 1, 0, 0};
            id<MTLBuffer> xb = buf(xg, sizeof xg), yb = buf(NULL, 4 * (size_t) P * R), sel = buf(selv, sizeof selv);
            MvArgs a = {K, R, P, xdiv, K, R, 0, 0};
            const size_t wsl = fmt == MF_BF16 ? 2 * (size_t) R * K : fmt == MF_Q8 ? (size_t) R * K : fmt == MF_Q4 ? (size_t) R * K / 2 : 2 * (size_t) R * K / 8;
            const size_t ssl = fmt == MF_SEED4 || fmt == MF_SEED4P4 ? 2 * (size_t) R * K / 8 : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (size_t) R * K / 64 : 0;
            const size_t bsl = fmt == MF_SEED4 || fmt == MF_SEED4P4 ? 4 : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (size_t) R * K / 64 : 0;
            const size_t nsl = fmt == MF_SEED4P4 ? (size_t) R * K / 16 : 0;   // one slice of exponent nibbles
            id<MTLComputePipelineState> p = pipe_("k_mv_gu", fmt, 0);
            run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:p];
                [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:Sb offset:0 atIndex:2]; [e setBuffer:Bb offset:0 atIndex:3];
                [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:sel offset:0 atIndex:6];
                [e setBuffer:Gt offset:0 atIndex:7];
                [e setBuffer:W offset:wsl atIndex:8]; [e setBuffer:Sb offset:ssl atIndex:9]; [e setBuffer:Bb offset:bsl atIndex:10];
                [e setBuffer:Nb offset:0 atIndex:11]; [e setBuffer:Nb offset:nsl atIndex:12];
                [e dispatchThreadgroups:MTLSizeMake((R + MV_RPT(fmt) - 1) / MV_RPT(fmt), P, 1) threadsPerThreadgroup:MTLSizeMake(32 * MV_ROWS, 1, 1)];
            });
            const float* y = (const float*) yb.contents;
            int badgu = 0;
            for (int pp = 0; pp < P; ++pp)
                for (int r = 0; r < R; ++r) {
                    double ag = 0, au = 0;
                    for (int c = 0; c < K; ++c) {
                        ag += wref(selv[pp], r, c) * xg[(pp / xdiv) * K + c];
                        au += wref(selv[pp] + 1, r, c) * xg[(pp / xdiv) * K + c];
                    }
                    const double gg = bfr(ag), uu = bfr(au);
                    const double ref = bfr(bfr(gg * bfr(1 / (1 + exp(-gg)))) * uu);
                    const double slack = (fabs(uu) * (fabs(gg) + 1) + fabs(gg) * fabs(uu)) / 128 + 1e-7;   // one BF16 flip of g or u
                    if (!close_bf(y[pp * R + r], ref, slack)) {
                        ++badgu;
                        if (badgu < 4) printf("FAIL mv_gu %s p=%d r=%d: %g vs %g\n", names[fmt], pp, r, y[pp * R + r], ref);
                    }
                }
            CHECK(!badgu, "k_mv_gu %s: %d mismatches", names[fmt], badgu);
        }
        printf("k_mv %-5s K %d dense T=1..8 + gather: worst error / (1 bf16 ulp + f32 bound) %.3f\n", names[fmt], K, worst);
    }
    free(w); free(q8); free(q4); free(s8); free(b8); free(s4); free(b4); free(d8); free(d4); free(seeds); free(nibs); free(wseed); free(G);
}

// ---- grouped norm, router, swiglu, combines, rope + KV, attention, embed, argmax ----------------------------------------------
static void test_misc(void) {
    unsigned sd = 7;
    const int d = 256, T = 3;
    float x[T * d];
    uint16_t wn[d];
    for (int i = 0; i < T * d; ++i) x[i] = bfr(frand(&sd) * 3);
    for (int i = 0; i < d; ++i) wn[i] = f2bf((float) (1 + 0.3 * frand(&sd)));
    {   // k_gnorm
        id<MTLBuffer> xb = buf(x, sizeof x), wb = buf(wn, sizeof wn), yb = buf(NULL, sizeof x);
        id<MTLComputePipelineState> p = pipe_("k_gnorm", 0, 0);
        const int32_t dd = d;
        const float eps = 1e-6f;
        run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p]; [e setBytes:&dd length:4 atIndex:0]; [e setBytes:&eps length:4 atIndex:1];
            [e setBuffer:xb offset:0 atIndex:2]; [e setBuffer:wb offset:0 atIndex:3]; [e setBuffer:yb offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        const float* y = (const float*) yb.contents;
        int bad = 0;
        for (int t = 0; t < T; ++t)
            for (int g = 0; g < 2; ++g) {
                double ss = 0;
                for (int c = 0; c < d / 2; ++c) ss += (double) x[t * d + g * d / 2 + c] * x[t * d + g * d / 2 + c];
                const double r = 1.0 / sqrt(ss / (d / 2) + 1e-6);
                for (int c = 0; c < d / 2; ++c) {
                    const int i = g * d / 2 + c;
                    bad += !close_bf(y[t * d + i], bfr(bf2f(wn[i]) * (x[t * d + i] * r)), 1e-6);
                }
            }
        CHECK(!bad, "k_gnorm: %d mismatches", bad);
    }
    {   // k_router: n 20, top 4, partitions of d / 2, BF16 weights + bias
        const int n = 20, k = 4;
        uint16_t W[n * d], bias[n];
        for (int i = 0; i < n * d; ++i) W[i] = f2bf((float) (frand(&sd) * 0.05));
        for (int i = 0; i < n; ++i) bias[i] = f2bf((float) (frand(&sd) * 0.01));
        id<MTLBuffer> wb = buf(W, sizeof W), bb = buf(bias, sizeof bias), xb = buf(x, sizeof x), ib = buf(NULL, 4 * T * k),
                      tb = buf(NULL, 4 * T * k), sb = buf(NULL, 4 * T * n), scb = buf(NULL, 4 * T * n);
        RouterArgs a = {d, n, k, d / 2, 2.5f};
        const int32_t TT = T;
        id<MTLComputePipelineState> pl = pipe_("k_router_logits", 0, 0), pt = pipe_("k_router_topk", 0, 0);
        run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pl]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:wb offset:0 atIndex:1]; [e setBuffer:bb offset:0 atIndex:2]; [e setBuffer:xb offset:0 atIndex:3];
            [e setBuffer:scb offset:0 atIndex:4]; [e setBuffer:sb offset:0 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((n + 7) / 8, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e setComputePipelineState:pt]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:scb offset:0 atIndex:1]; [e setBuffer:sb offset:0 atIndex:2]; [e setBuffer:ib offset:0 atIndex:3];
            [e setBuffer:tb offset:0 atIndex:4]; [e setBytes:&TT length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        });
        const int* ind = (const int*) ib.contents;
        const float* wt = (const float*) tb.contents;
        int bad = 0;
        for (int t = 0; t < T; ++t) {
            double score[64], sel[64];
            for (int e = 0; e < n; ++e) {
                double s0 = 0, s1 = 0;
                for (int c = 0; c < d / 2; ++c) s0 += bf2f(W[e * d + c]) * x[t * d + c];
                for (int c = d / 2; c < d; ++c) s1 += bf2f(W[e * d + c]) * x[t * d + c];
                score[e] = 1 / (1 + exp(-((double) bfr(s0) + bfr(s1))));
                sel[e] = score[e] + bf2f(bias[e]);
            }
            int ch[8];
            double sum = 0;
            for (int j = 0; j < k; ++j) {
                int best = -1;
                for (int e = 0; e < n; ++e) {
                    int tk = 0;
                    for (int q = 0; q < j; ++q) tk |= ch[q] == e;
                    if (!tk && (best < 0 || sel[e] > sel[best])) best = e;
                }
                ch[j] = best;
                sum += score[best];
            }
            for (int j = 0; j < k; ++j) {
                bad += ind[t * k + j] != ch[j];
                bad += fabs(wt[t * k + j] - score[ch[j]] / sum * 2.5) > 1e-5;
            }
        }
        CHECK(!bad, "k_router: %d mismatches", bad);
    }
    for (int shape = 0; shape < 2; ++shape) {   // k_router_topk alone at MoVA's shapes, with exact ties (lowest id wins)
        const int n = shape ? 64 : 100, k = shape ? 4 : 8, TK = 6;
        float sc[TK * 100], se[TK * 100];
        for (int i = 0; i < TK * n; ++i) {
            sc[i] = (float) frand(&sd) * 0.5f + 0.5f;
            se[i] = (float) ((int) (frand(&sd) * 12)) / 16.0f;   // few distinct values: many ties
        }
        id<MTLBuffer> scb = buf(sc, sizeof sc), seb = buf(se, sizeof se), ib = buf(NULL, 4 * TK * k), tb = buf(NULL, 4 * TK * k);
        RouterArgs a = {d, n, k, d / 2, 2.5f};
        const int32_t TT = TK;
        id<MTLComputePipelineState> pt = pipe_("k_router_topk", 0, 0);
        run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pt]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:scb offset:0 atIndex:1]; [e setBuffer:seb offset:0 atIndex:2]; [e setBuffer:ib offset:0 atIndex:3];
            [e setBuffer:tb offset:0 atIndex:4]; [e setBytes:&TT length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(TK, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        });
        const int* ind = (const int*) ib.contents;
        const float* wt = (const float*) tb.contents;
        int bad = 0;
        for (int t = 0; t < TK; ++t) {
            int ch[8];
            double sum = 0;
            for (int j = 0; j < k; ++j) {
                int best = -1;
                for (int e = 0; e < n; ++e) {
                    int tk = 0;
                    for (int q = 0; q < j; ++q) tk |= ch[q] == e;
                    if (!tk && (best < 0 || se[t * n + e] > se[t * n + best])) best = e;
                }
                ch[j] = best;
                sum += sc[t * n + best];
            }
            for (int j = 0; j < k; ++j) {
                bad += ind[t * k + j] != ch[j];
                bad += fabs(wt[t * k + j] - sc[t * n + ch[j]] / sum * 2.5) > 1e-5;
            }
        }
        CHECK(!bad, "k_router_topk n %d k %d (ties): %d mismatches", n, k, bad);
    }
    {   // k_swiglu, k_moe_combine, k_vcombine
        const int n = 64;
        float g[n], u[n];
        for (int i = 0; i < n; ++i) { g[i] = bfr(frand(&sd) * 4); u[i] = bfr(frand(&sd)); }
        id<MTLBuffer> gb = buf(g, sizeof g), ub = buf(u, sizeof u), ab = buf(NULL, sizeof g);
        id<MTLComputePipelineState> p = pipe_("k_swiglu", 0, 0);
        const int32_t nn = n;
        run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p]; [e setBuffer:gb offset:0 atIndex:0]; [e setBuffer:ub offset:0 atIndex:1];
            [e setBuffer:ab offset:0 atIndex:2]; [e setBytes:&nn length:4 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        });
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const double s = bfr(1 / (1 + exp(-(double) g[i]))), si = bfr(g[i] * s), ref = bfr(si * u[i]);
            bad += !close_bf(((float*) ab.contents)[i], ref, 1e-7);
        }
        CHECK(!bad, "k_swiglu: %d mismatches", bad);
        const int dd = 32, k = 3, TT = 2;
        float D[TT * k * dd], w[TT * k], sh[TT * dd], xx[TT * dd];
        for (int i = 0; i < TT * k * dd; ++i) D[i] = bfr(frand(&sd));
        for (int i = 0; i < TT * k; ++i) w[i] = (float) (0.3 + 0.5 * fabs(frand(&sd)));
        for (int i = 0; i < TT * dd; ++i) { sh[i] = bfr(frand(&sd)); xx[i] = bfr(frand(&sd)); }
        id<MTLBuffer> Db = buf(D, sizeof D), wb = buf(w, sizeof w), shb = buf(sh, sizeof sh), xb = buf(xx, sizeof xx), vb = buf(NULL, sizeof sh);
        const int32_t dk_[2] = {dd, k};
        const int32_t* dk = dk_;
        id<MTLComputePipelineState> pc = pipe_("k_moe_combine", 0, 0), pv = pipe_("k_vcombine", 0, 0);
        run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pc]; [e setBuffer:Db offset:0 atIndex:0]; [e setBuffer:wb offset:0 atIndex:1];
            [e setBuffer:shb offset:0 atIndex:2]; [e setBuffer:xb offset:0 atIndex:3]; [e setBytes:dk length:8 atIndex:4];
            [e dispatchThreads:MTLSizeMake(dd, TT, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e setComputePipelineState:pv]; [e setBuffer:Db offset:0 atIndex:0]; [e setBuffer:wb offset:0 atIndex:1];
            [e setBuffer:vb offset:0 atIndex:2]; [e setBytes:dk length:8 atIndex:3];
            [e dispatchThreads:MTLSizeMake(dd, TT, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        });
        int bc = 0, bv = 0;
        for (int t = 0; t < TT; ++t)
            for (int c = 0; c < dd; ++c) {
                double s = 0, v = 0;
                for (int j = 0; j < k; ++j) {
                    const double dv = D[(t * k + j) * dd + c], wj = bfr(w[t * k + j]);
                    s += bfr(dv * wj);
                    const double sg = bfr(1 / (1 + exp(-dv))), si = bfr(dv * sg);
                    v += bfr(si * wj);
                }
                bc += !close_bf(((float*) xb.contents)[t * dd + c], bfr(xx[t * dd + c] + bfr(bfr(s) + sh[t * dd + c])), 1e-6);
                bv += !close_bf(((float*) vb.contents)[t * dd + c], bfr(v), 1e-6);
            }
        CHECK(!bc && !bv, "k_moe_combine %d, k_vcombine %d mismatches", bc, bv);
    }
    {   // k_rope_kv + k_attn + k_attn_reduce: 2 query heads per KV head, 1 KV head, 5 cached positions, rows at 3 and 4
        const int nh = 4, nkv = 1, P = 5, T2 = 2;
        float q[T2 * nh * 128], k[T2 * nkv * 128], v[T2 * nkv * 128], g[T2 * nh * 128];
        for (int i = 0; i < T2 * nh * 128; ++i) { q[i] = bfr(frand(&sd)); g[i] = bfr(frand(&sd) * 3); }
        for (int i = 0; i < T2 * nkv * 128; ++i) { k[i] = bfr(frand(&sd)); v[i] = bfr(frand(&sd)); }
        uint16_t Kc[P * 128], Vc[P * 128];
        for (int i = 0; i < 3 * 128; ++i) { Kc[i] = f2bf((float) frand(&sd)); Vc[i] = f2bf((float) frand(&sd)); }
        float inv[64];
        for (int pp = 0; pp < 64; ++pp) inv[pp] = (float) pow(1e7, -2.0 * pp / 128);
        RowInfo ri[2] = {{3, {0}}, {4, {0}}};
        float qcopy[T2 * nh * 128];
        memcpy(qcopy, q, sizeof q);
        id<MTLBuffer> qb = buf(q, sizeof q), kb = buf(k, sizeof k), vb = buf(v, sizeof v), Kb = buf(Kc, sizeof Kc), Vb = buf(Vc, sizeof Vc),
                      rb = buf(ri, sizeof ri), ib = buf(inv, sizeof inv), gb = buf(g, sizeof g), ob = buf(NULL, sizeof q);
        const int ns = 2;
        AttnArgs a = {nh, nkv, 3, ns, (float) (1 / sqrt(128.0))};
        id<MTLBuffer> pb = buf(NULL, 4 * (size_t) T2 * nh * ns * 130);
        const int32_t hk_[2] = {nh, nkv};
        const int32_t* hk = hk_;
        id<MTLComputePipelineState> pr = pipe_("k_rope_kv", 0, 0), pa = pipe_("k_attn", 0, 0), pd = pipe_("k_attn_reduce", 0, 0);
        run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pr];
            [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kb offset:0 atIndex:1]; [e setBuffer:vb offset:0 atIndex:2];
            [e setBuffer:Kb offset:0 atIndex:3]; [e setBuffer:Vb offset:0 atIndex:4]; [e setBuffer:rb offset:0 atIndex:5];
            [e setBuffer:ib offset:0 atIndex:6]; [e setBytes:hk length:8 atIndex:7];
            [e dispatchThreads:MTLSizeMake(nh * 64, T2, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e setComputePipelineState:pa]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:Kb offset:0 atIndex:2]; [e setBuffer:Vb offset:0 atIndex:3];
            [e setBuffer:rb offset:0 atIndex:4]; [e setBuffer:pb offset:0 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(ns, nkv, T2) threadsPerThreadgroup:MTLSizeMake(32 * nh, 1, 1)];
            [e setComputePipelineState:pd]; [e setBytes:&a length:sizeof a atIndex:0]; [e setBuffer:pb offset:0 atIndex:1];
            [e setBuffer:gb offset:0 atIndex:2]; [e setBuffer:ob offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(nh, T2, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
        // reference: rope, cache, causal attention, gate
        double kc[P][128], vc[P][128];
        for (int pp = 0; pp < 3; ++pp) for (int dd = 0; dd < 128; ++dd) { kc[pp][dd] = bf2f(Kc[pp * 128 + dd]); vc[pp][dd] = bf2f(Vc[pp * 128 + dd]); }
        double qr[T2][nh][128];
        for (int t = 0; t < T2; ++t) {
            const int pos = ri[t].pos;
            for (int i = 0; i < 64; ++i) {
                const double th = (double) (float) ((float) pos * inv[i]), cs = cos(th), sn = sin(th);
                for (int h = 0; h < nh; ++h) {
                    const double x0 = qcopy[t * nh * 128 + h * 128 + i], x1 = qcopy[t * nh * 128 + h * 128 + i + 64];
                    qr[t][h][i] = bfr(x0 * cs - x1 * sn);
                    qr[t][h][i + 64] = bfr(x1 * cs + x0 * sn);
                }
                const double k0 = k[t * 128 + i], k1 = k[t * 128 + i + 64];
                kc[pos][i] = bfr(k0 * cs - k1 * sn);
                kc[pos][i + 64] = bfr(k1 * cs + k0 * sn);
            }
            for (int dd = 0; dd < 128; ++dd) vc[pos][dd] = v[t * 128 + dd];
        }
        int bad = 0;
        const float* o = (const float*) ob.contents;
        for (int t = 0; t < T2; ++t)
            for (int h = 0; h < nh; ++h) {
                double s[P], m = -1e300, l = 0;
                for (int pp = 0; pp <= ri[t].pos; ++pp) {
                    double acc = 0;
                    for (int dd = 0; dd < 128; ++dd) acc += qr[t][h][dd] * a.scale * kc[pp][dd];
                    s[pp] = acc;
                    m = fmax(m, acc);
                }
                for (int pp = 0; pp <= ri[t].pos; ++pp) l += exp(s[pp] - m);
                for (int dd = 0; dd < 128; ++dd) {
                    double acc = 0;
                    for (int pp = 0; pp <= ri[t].pos; ++pp) acc += exp(s[pp] - m) / l * vc[pp][dd];
                    const double gx = g[t * nh * 128 + h * 128 + dd] * 0.69314718055994531;
                    const double sp = bfr((fmax(gx, 0) + log1p(exp(-fabs(gx)))) / 0.69314718055994531);
                    const double ref = bfr(bfr(acc) * sp);
                    if (!close_bf(o[t * nh * 128 + h * 128 + dd], ref, 2e-6)) ++bad;
                }
            }
        CHECK(!bad, "rope + attention + gate: %d mismatches of %d", bad, T2 * nh * 128);
    }
    {   // k_embed (BF16 and Q8) and k_argmax
        const int V = 40, dd = 64;
        uint16_t E[V * dd];
        for (int i = 0; i < V * dd; ++i) E[i] = f2bf((float) frand(&sd));
        uint32_t q8[V * dd / 4];
        uint16_t s8[V], b8[V], deq[V * dd];
        nslm_affine_quantize(E, V, dd, 8, q8, s8, b8);
        nslm_affine_dequantize(q8, s8, b8, V, dd, 8, deq);
        int ids[3] = {7, 0, 39};
        int bad = 0;
        for (int fmt = 0; fmt < 3; fmt += 2) {
            id<MTLBuffer> Eb = buf(E, sizeof E), Qb = buf(q8, sizeof q8), Sb = buf(s8, sizeof s8), Bb = buf(b8, sizeof b8),
                          ib = buf(ids, sizeof ids), xb = buf(NULL, 4 * 3 * dd);
            const int32_t d2 = dd;
            id<MTLComputePipelineState> p = pipe_("k_embed", fmt, 0);
            run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:p]; [e setBytes:&d2 length:4 atIndex:0]; [e setBuffer:Eb offset:0 atIndex:1];
                [e setBuffer:Qb offset:0 atIndex:2]; [e setBuffer:Sb offset:0 atIndex:3]; [e setBuffer:Bb offset:0 atIndex:4];
                [e setBuffer:ib offset:0 atIndex:5]; [e setBuffer:xb offset:0 atIndex:6];
                [e dispatchThreads:MTLSizeMake(dd, 3, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            });
            for (int t = 0; t < 3; ++t)
                for (int c = 0; c < dd; ++c) bad += ((float*) xb.contents)[t * dd + c] != bf2f(fmt == 2 ? deq[ids[t] * dd + c] : E[ids[t] * dd + c]);
        }
        CHECK(!bad, "k_embed: %d mismatches", bad);
        const int VV = 250624;
        float* lg = malloc(4 * (size_t) VV * 2);
        for (int i = 0; i < 2 * VV; ++i) lg[i] = bfr(frand(&sd) * 10);
        lg[123456] = 50; lg[200000] = 50;                 // a tie: the lowest id wins
        lg[VV + 99] = 60;
        id<MTLBuffer> lb = buf(lg, 4 * (size_t) VV * 2), ob = buf(NULL, 8);
        const int32_t vv = VV;
        id<MTLComputePipelineState> p = pipe_("k_argmax", 0, 0);
        run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p]; [e setBuffer:lb offset:0 atIndex:0]; [e setBuffer:ob offset:0 atIndex:1];
            [e setBytes:&vv length:4 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        });
        CHECK(((int*) ob.contents)[0] == 123456 && ((int*) ob.contents)[1] == 99, "k_argmax: %d %d", ((int*) ob.contents)[0], ((int*) ob.contents)[1]);
        free(lg);
    }
}


// ---- decode attention (k_attn + k_attn_reduce) at a GQA 4 layout: several splits, rows at different positions -----------
static void test_attn_decode(void) {
    unsigned sd = 21;
    const int nh = 8, nkv = 2, NP = 301, T = 3, ns = 5;
    const int pos[3] = {300, 299, 37};
    float* q = malloc(4 * (size_t) T * nh * 128), *g = malloc(4 * (size_t) T * nh * 128);
    uint16_t* Kc = malloc(2 * (size_t) NP * nkv * 128), *Vc = malloc(2 * (size_t) NP * nkv * 128);
    for (int i = 0; i < T * nh * 128; ++i) { q[i] = bfr(frand(&sd) * 2); g[i] = bfr(frand(&sd) * 3); }
    for (int i = 0; i < NP * nkv * 128; ++i) { Kc[i] = f2bf((float) (frand(&sd) * 2)); Vc[i] = f2bf((float) frand(&sd)); }
    RowInfo ri[3];
    for (int t = 0; t < T; ++t) ri[t].pos = pos[t];
    AttnArgs a = {nh, nkv, (NP + ns - 1) / ns, ns, (float) (1 / sqrt(128.0))};
    id<MTLBuffer> qb = buf(q, 4 * (size_t) T * nh * 128), Kb = buf(Kc, 2 * (size_t) NP * nkv * 128), Vb = buf(Vc, 2 * (size_t) NP * nkv * 128),
                  rb = buf(ri, sizeof ri), gb = buf(g, 4 * (size_t) T * nh * 128), ob = buf(NULL, 4 * (size_t) T * nh * 128),
                  pb = buf(NULL, 4 * (size_t) T * nh * ns * 130);
    id<MTLComputePipelineState> pa = pipe_("k_attn", 0, 0), pd = pipe_("k_attn_reduce", 0, 0);
    run(^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pa]; [e setBytes:&a length:sizeof a atIndex:0];
        [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:Kb offset:0 atIndex:2]; [e setBuffer:Vb offset:0 atIndex:3];
        [e setBuffer:rb offset:0 atIndex:4]; [e setBuffer:pb offset:0 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(ns, nkv, T) threadsPerThreadgroup:MTLSizeMake(32 * ATT_SG, 1, 1)];
        [e setComputePipelineState:pd]; [e setBytes:&a length:sizeof a atIndex:0]; [e setBuffer:pb offset:0 atIndex:1];
        [e setBuffer:gb offset:0 atIndex:2]; [e setBuffer:ob offset:0 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(nh, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    });
    const float* o = (const float*) ob.contents;
    int bad = 0;
    double* s = malloc(sizeof(double) * NP);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < nh; ++h) {
            const int kvh = h / (nh / nkv);
            double m = -1e300, l = 0;
            for (int pp = 0; pp <= pos[t]; ++pp) {
                double acc = 0;
                for (int dd = 0; dd < 128; ++dd) acc += (double) q[((size_t) t * nh + h) * 128 + dd] * a.scale * bf2f(Kc[((size_t) pp * nkv + kvh) * 128 + dd]);
                s[pp] = acc;
                m = fmax(m, acc);
            }
            for (int pp = 0; pp <= pos[t]; ++pp) l += exp(s[pp] - m);
            for (int dd = 0; dd < 128; ++dd) {
                double acc = 0;
                for (int pp = 0; pp <= pos[t]; ++pp) acc += exp(s[pp] - m) / l * bf2f(Vc[((size_t) pp * nkv + kvh) * 128 + dd]);
                const size_t i = ((size_t) t * nh + h) * 128 + dd;
                const double gx = g[i] * 0.69314718055994531;
                const double sp = bfr((fmax(gx, 0) + log1p(exp(-fabs(gx)))) / 0.69314718055994531);
                if (!close_bf(o[i], bfr(bfr(acc) * sp), 2e-6)) {
                    if (bad < 3) printf("  k_attn t %d h %d d %d: %.8g vs %.8g\n", t, h, dd, o[i], bfr(bfr(acc) * sp));
                    ++bad;
                }
            }
        }
    CHECK(!bad, "k_attn decode (%d rows, %d splits, GQA %d/%d): %d mismatches of %d", T, ns, nh, nkv, bad, T * nh * 128);
    printf("k_attn decode: %d rows x %d heads, %d splits, %d mismatches\n", T, nh, ns, bad);
    free(q); free(g); free(Kc); free(Vc); free(s);
}

// ---- prefill attention (k_attn_prefill): key tiles through simdgroup matrices, causal, the softplus gate -----------------
// MLX's prefill attention rounding points: q * scale and the probabilities to BF16 for the matrix products (f32 sums),
// the row sum from the unrounded probabilities.  Reference in double with those two roundings; outputs within 1 BF16 ulp,
// except at most 2 of the (row, head) pairs (a probability within f32 noise of a BF16 rounding boundary can flip).
static void test_attn_prefill(void) {
    unsigned sd = 11;
    const int nh = 8, nkv = 2, P0 = 37, T = 45, NP = P0 + T;   // rows at positions P0 .. P0 + T - 1, keys 0 .. their own
    float* q = malloc(4 * (size_t) T * nh * 128), *g = malloc(4 * (size_t) T * nh * 128);
    uint16_t* Kc = malloc(2 * (size_t) NP * nkv * 128), *Vc = malloc(2 * (size_t) NP * nkv * 128);
    for (int i = 0; i < T * nh * 128; ++i) { q[i] = bfr(frand(&sd) * 2); g[i] = bfr(frand(&sd) * 3); }
    for (int i = 0; i < NP * nkv * 128; ++i) { Kc[i] = f2bf((float) (frand(&sd) * 2)); Vc[i] = f2bf((float) frand(&sd)); }
    RowInfo* ri = calloc((size_t) T, sizeof(RowInfo));
    for (int t = 0; t < T; ++t) ri[t].pos = P0 + t;
    AttnArgs a = {nh, nkv, 0, 1, (float) (1 / sqrt(128.0))};
    const int32_t TT = T;
    id<MTLBuffer> qb = buf(q, 4 * (size_t) T * nh * 128), Kb = buf(Kc, 2 * (size_t) NP * nkv * 128), Vb = buf(Vc, 2 * (size_t) NP * nkv * 128),
                  rb = buf(ri, sizeof(RowInfo) * (size_t) T), gb = buf(g, 4 * (size_t) T * nh * 128), ob = buf(NULL, 4 * (size_t) T * nh * 128);
    id<MTLComputePipelineState> pf = pipe_("k_attn_prefill", 0, 0);
    CHECK(pf != nil, "k_attn_prefill: no pipeline");
    if (!pf) return;
    run(^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pf]; [e setBytes:&a length:sizeof a atIndex:0];
        [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:Kb offset:0 atIndex:2]; [e setBuffer:Vb offset:0 atIndex:3];
        [e setBuffer:rb offset:0 atIndex:4]; [e setBuffer:gb offset:0 atIndex:5]; [e setBuffer:ob offset:0 atIndex:6];
        [e setBytes:&TT length:4 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake((T + 8 * ATTF_RS - 1) / (8 * ATTF_RS), nkv, 1) threadsPerThreadgroup:MTLSizeMake(32 * ATTF_G * ATTF_RS, 1, 1)];
    });
    const float* o = (const float*) ob.contents;
    int bad = 0, badpairs = 0;
    double* s = malloc(sizeof(double) * NP);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < nh; ++h) {
            const int bad0 = bad;
            const int kvh = h / (nh / nkv), pos = ri[t].pos;
            double m = -1e300, l = 0;
            for (int pp = 0; pp <= pos; ++pp) {
                double acc = 0;
                for (int dd = 0; dd < 128; ++dd) acc += (double) bfr(q[((size_t) t * nh + h) * 128 + dd] * a.scale) * bf2f(Kc[((size_t) pp * nkv + kvh) * 128 + dd]);
                s[pp] = acc;
                m = fmax(m, acc);
            }
            // online over key tiles of ATTF_BK from key 0: P = bf16(exp(s - running max)), O and l rescaled per tile
            double O[128] = {0}, mr = -1e300;
            for (int k0 = 0; k0 <= pos; k0 += ATTF_BK) {
                double tm = -1e300;
                for (int pp = k0; pp < k0 + ATTF_BK && pp <= pos; ++pp) tm = fmax(tm, s[pp]);
                const double mn = fmax(mr, tm), cor = mr == -1e300 ? 1.0 : exp(mr - mn);
                l *= cor;
                for (int dd = 0; dd < 128; ++dd) O[dd] *= cor;
                for (int pp = k0; pp < k0 + ATTF_BK && pp <= pos; ++pp) {
                    const double pv = exp(s[pp] - mn);
                    l += pv;
                    for (int dd = 0; dd < 128; ++dd) O[dd] += (double) bfr(pv) * bf2f(Vc[((size_t) pp * nkv + kvh) * 128 + dd]);
                }
                mr = mn;
            }
            (void) m;
            for (int dd = 0; dd < 128; ++dd) {
                const double acc = O[dd] / l;
                const size_t i = ((size_t) t * nh + h) * 128 + dd;
                const double gx = g[i] * 0.69314718055994531;
                const double sp = bfr((fmax(gx, 0) + log1p(exp(-fabs(gx)))) / 0.69314718055994531);
                const double ref = bfr(bfr(acc) * sp);
                if (!close_bf(o[i], ref, 2e-6)) {
                    if (bad < 3) printf("  k_attn_prefill t %d h %d d %d: %.8g vs %.8g\n", t, h, dd, o[i], ref);
                    ++bad;
                }
            }
            badpairs += bad > bad0;
        }
    CHECK(badpairs <= 2, "k_attn_prefill (T %d after %d cached, GQA %d/%d, gate): %d (row, head) pairs off (%d outputs of %d)", T, P0,
          nh, nkv, badpairs, bad, T * nh * 128);
    printf("k_attn_prefill: %d rows x %d heads, %d (row, head) pairs off, %d outputs\n", T, nh, badpairs, bad);
    free(q); free(g); free(Kc); free(Vc); free(ri); free(s);
}


// ---- GEMM: dense (T tokens, partial tiles) and grouped (a tile table over a permuted pair list), every format ------------
static void test_mm(void) {
    const int R = 40, S = 3;   // R not a multiple of 32; K = 128: four K steps, two Q8/Q4 groups
    unsigned sd = 3;
    uint16_t* w;
    uint32_t* q8, *q4;
    uint16_t* s8, *b8, *s4, *b4;
    const int K2 = 128;
    w = malloc(2 * (size_t) S * R * K2);
    for (int i = 0; i < S * R * K2; ++i) w[i] = f2bf((float) (frand(&sd) * 0.05));
    q8 = malloc((size_t) S * R * K2); q4 = malloc((size_t) S * R * K2 / 2);
    s8 = malloc(2 * (size_t) S * R * K2 / 64); b8 = malloc(2 * (size_t) S * R * K2 / 64);
    s4 = malloc(2 * (size_t) S * R * K2 / 64); b4 = malloc(2 * (size_t) S * R * K2 / 64);
    nslm_affine_quantize(w, S * R, K2, 8, q8, s8, b8);
    nslm_affine_quantize(w, S * R, K2, 4, q4, s4, b4);
    uint16_t* d8 = malloc(2 * (size_t) S * R * K2), *d4 = malloc(2 * (size_t) S * R * K2);
    nslm_affine_dequantize(q8, s8, b8, S * R, K2, 8, d8);
    nslm_affine_dequantize(q4, s4, b4, S * R, K2, 4, d4);
    const int nb = S * R * K2 / 8;
    uint16_t* seeds = malloc(2 * (size_t) nb), *nibs = malloc(2 * (size_t) nb);
    int32_t ebias[3] = {-18, -16, -20};
    for (int i = 0; i < nb; ++i) { seeds[i] = (uint16_t) (1 + (sd = sd * 1103515245u + 12345u) % 65535); nibs[i] = (uint16_t) ((sd >> 3) & 0xFFFF); }
    double* wseed = malloc(sizeof(double) * (size_t) S * R * K2);
    for (int s = 0; s < S; ++s)
        for (int b = 0; b < R * K2 / 8; ++b) {
            const int bi = s * R * K2 / 8 + b;
            uint16_t st[24];
            lfsr_states(seeds[bi], 24, st);
            const double sc = (double) NSLM_R32 * pow(2.0, ebias[s] + nslm_ecode(nibs[bi]));
            for (int c = 0; c < 8; ++c)
                wseed[(size_t) bi * 8 + c] = sc * (((double) st[3 * c] - 32768) * nslm_q(nibs[bi], 0) + ((double) st[3 * c + 1] - 32768) * nslm_q(nibs[bi], 1) +
                                                   ((double) st[3 * c + 2] - 32768) * nslm_q(nibs[bi], 2));
        }
    uint32_t* G = malloc(65536 * 4);
    for (uint32_t s = 0; s < 65536; ++s) G[s] = lfsr_stream24((uint16_t) s);
    id<MTLBuffer> gb = buf(G, 65536 * 4), g32 = g32_buffer();
    const P4 p4 = p4_make(S, R, K2, &sd);
    const int T = 45, P = 50, xdiv = 2;
    float* x = malloc(4 * (size_t) T * K2);
    for (int i = 0; i < T * K2; ++i) x[i] = bfr(frand(&sd));
    const char* names[5] = {"bf16", "seed4", "q8", "q4", "seed4p4"};
    for (int fmt = 0; fmt < 5; ++fmt) {
        id<MTLBuffer> W = fmt == MF_BF16 ? buf(w, 2 * (size_t) S * R * K2) : fmt == MF_Q8 ? buf(q8, (size_t) S * R * K2)
                         : fmt == MF_Q4 ? buf(q4, (size_t) S * R * K2 / 2) : fmt == MF_SEED4P4 ? buf(p4.seeds, 2 * p4.nb) : buf(seeds, 2 * (size_t) nb);
        id<MTLBuffer> Sb = fmt == MF_Q8 ? buf(s8, 2 * (size_t) S * R * K2 / 64) : fmt == MF_Q4 ? buf(s4, 2 * (size_t) S * R * K2 / 64)
                          : fmt == MF_SEED4 ? buf(nibs, 2 * (size_t) nb) : fmt == MF_SEED4P4 ? buf(p4.coefs, 2 * p4.nb) : buf(NULL, 16);
        id<MTLBuffer> Bb = fmt == MF_Q8 ? buf(b8, 2 * (size_t) S * R * K2 / 64) : fmt == MF_Q4 ? buf(b4, 2 * (size_t) S * R * K2 / 64)
                          : fmt == MF_SEED4 ? buf(ebias, 12) : fmt == MF_SEED4P4 ? buf(p4.bias, 12) : buf(NULL, 16);
        id<MTLBuffer> Nb = fmt == MF_SEED4P4 ? buf(p4.nib, p4.nb / 2 + 1) : buf(NULL, 16), Gt = fmt == MF_SEED4P4 ? g32 : gb;
        double (^wref)(int, int, int) = ^double(int s, int r, int c) {
            const size_t i = ((size_t) s * R + r) * K2 + c;
            return fmt == MF_BF16 ? bf2f(w[i]) : fmt == MF_Q8 ? bf2f(d8[i]) : fmt == MF_Q4 ? bf2f(d4[i]) : fmt == MF_SEED4P4 ? p4.w[i] : wseed[i];
        };
        int bad = 0;
        {   // dense, slice 0, residual add
            float* y0 = malloc(4 * (size_t) T * R);
            for (int i = 0; i < T * R; ++i) y0[i] = bfr(frand(&sd));
            id<MTLBuffer> xb = buf(x, 4 * (size_t) T * K2), yb = buf(y0, 4 * (size_t) T * R), pb = buf(NULL, 16), tb = buf(NULL, 16);
            MmArgs a = {K2, R, T, K2, R, 1, 1, 0};
            id<MTLComputePipelineState> p = pipe_("k_mm", fmt, 1);
            run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:p]; [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:Sb offset:0 atIndex:2]; [e setBuffer:Bb offset:0 atIndex:3];
                [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:pb offset:0 atIndex:6];
                [e setBuffer:Gt offset:0 atIndex:7]; [e setBuffer:tb offset:0 atIndex:8]; [e setBuffer:Nb offset:0 atIndex:9];
                [e dispatchThreadgroups:MTLSizeMake((R + MM_BM - 1) / MM_BM, (T + MM_BN - 1) / MM_BN, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            });
            const float* y = (const float*) yb.contents;
            for (int n = 0; n < T; ++n)
                for (int r = 0; r < R; ++r) {
                    double acc = 0, mag = 0;
                    for (int c = 0; c < K2; ++c) { const double v = wref(0, r, c) * x[n * K2 + c]; acc += v; mag += fabs(v); }
                    if (!close_bf(y[n * R + r], bfr(y0[n * R + r] + bfr(acc)), 2 * 64 * 1.2e-7 * mag)) {
                        if (bad++ < 3) printf("FAIL mm %s dense n=%d r=%d: %g vs %g\n", names[fmt], n, r, y[n * R + r], bfr(y0[n * R + r] + bfr(acc)));
                    }
                }
            free(y0);
        }
        {   // grouped: P pairs (input pair / xdiv), slices assigned per pair, sorted into a tile table (<= 32 per tile)
            int slc[P], perm[P], np = 0;
            for (int pp = 0; pp < P; ++pp) slc[pp] = (pp * 7) % S;
            MmTile tl[16];
            int nt = 0;
            for (int s = 0; s < S; ++s) {
                const int st = np;
                for (int pp = 0; pp < P; ++pp) if (slc[pp] == s) perm[np++] = pp;
                for (int o = st; o < np; o += MM_BN) tl[nt++] = (MmTile){s, o, np - o < MM_BN ? np - o : MM_BN, 0};
            }
            id<MTLBuffer> xb = buf(x, 4 * (size_t) T * K2), yb = buf(NULL, 4 * (size_t) P * R), pb = buf(perm, sizeof perm), tb = buf(tl, sizeof tl);
            MmArgs a = {K2, R, 0, K2, R, xdiv, 0, 0};
            id<MTLComputePipelineState> p = pipe_("k_mm", fmt, 0);
            run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:p]; [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:Sb offset:0 atIndex:2]; [e setBuffer:Bb offset:0 atIndex:3];
                [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:pb offset:0 atIndex:6];
                [e setBuffer:Gt offset:0 atIndex:7]; [e setBuffer:tb offset:0 atIndex:8]; [e setBuffer:Nb offset:0 atIndex:9];
                [e dispatchThreadgroups:MTLSizeMake((R + MM_BM - 1) / MM_BM, nt, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            });
            const float* y = (const float*) yb.contents;
            for (int pp = 0; pp < P; ++pp)
                for (int r = 0; r < R; ++r) {
                    double acc = 0, mag = 0;
                    for (int c = 0; c < K2; ++c) { const double v = wref(slc[pp], r, c) * x[(pp / xdiv) * K2 + c]; acc += v; mag += fabs(v); }
                    if (!close_bf(y[pp * R + r], bfr(acc), 64 * 1.2e-7 * mag)) {
                        if (bad++ < 6) printf("FAIL mm %s grouped p=%d r=%d: %g vs %g\n", names[fmt], pp, r, y[pp * R + r], bfr(acc));
                    }
                }
        }
        CHECK(!bad, "k_mm %s: %d mismatches", names[fmt], bad);
        printf("k_mm %-5s dense (T=45, residual) + grouped (50 pairs, 3 slices): %s\n", names[fmt], bad ? "FAIL" : "ok");
    }
    free(w); free(q8); free(q4); free(s8); free(b8); free(s4); free(b4); free(d8); free(d4); free(seeds); free(nibs); free(wseed); free(G); free(x);
}

int main(void) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        NSError* err = nil;
        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"out/res/kernels_moe.metallib"] error:&err];
        if (!lib) { printf("FAIL: no out/res/kernels_moe.metallib\n"); return 1; }
        test_mv(256);
        test_mv(512);
        test_misc();
        test_mm();
        test_attn_prefill();
        test_attn_decode();
        printf("%s\n", fails ? "FAIL" : "PASS");
        return fails != 0;
    }
}
