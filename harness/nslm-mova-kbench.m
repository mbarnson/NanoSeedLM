// harness/nslm-mova-kbench.m - micro-benchmark of the MoVA kernels at MoVA's shapes (decode matvec per format, fused
// gate/up, attention): GPU time per call and effective bandwidth.  Random weights; timing only.
//
//   nslm-mova-kbench [--iters 200] [--kernel k_mv] [--lib PATH] [--nr-seed N] [--nr-aff N] [--gu 0|1]
//                    [--dattn POS [--splits 64] [--per 128]] [--attn P0]
//
// Time per call = (GPUEndTime - GPUStartTime) / iters over one serial command buffer, best of reps after a warm-up.
// Gather cases rotate through the slices so successive calls don't hit the same expert in cache.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "kernels_moe.metal"
#include "lfsr.h"

static id<MTLDevice> dev;
static id<MTLLibrary> lib;

static id<MTLComputePipelineState> pipe_(const char* name, int fmt, int T) {
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    short f = (short) fmt, t = (short) T;
    [cv setConstantValue:&f type:MTLDataTypeShort atIndex:0];
    [cv setConstantValue:&t type:MTLDataTypeShort atIndex:1];
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@(name) constantValues:cv error:&err];
    if (!fn) { fprintf(stderr, "no %s\n", name); exit(2); }
    return [dev newComputePipelineStateWithFunction:fn error:&err];
}
static id<MTLBuffer> rnd(size_t n, unsigned seed) {
    id<MTLBuffer> b = [dev newBufferWithLength:n > 16 ? n : 16 options:MTLResourceStorageModeShared];
    uint32_t* p = (uint32_t*) b.contents;
    for (size_t i = 0; i < b.length / 4; ++i) { seed = seed * 1103515245u + 12345u; p[i] = seed; }
    return b;
}

typedef struct { const char* name; int R, K, slices, P, xdiv; } Shape;   // P = 0: dense T = 1

int main(int argc, char** argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        const int iters = atoi(opt(argc, argv, "--iters", "200"));
        const char* kname = opt(argc, argv, "--kernel", "k_mv");
        // rows per simdgroup the metallib was built with (-DMV_NR_SEED / -DMV_NR_AFF); defaults = this build's
        int nr_seed = atoi(opt(argc, argv, "--nr-seed", "0")), nr_aff = atoi(opt(argc, argv, "--nr-aff", "0"));
        if (nr_seed <= 0) nr_seed = MV_NR_SEED;
        if (nr_aff <= 0) nr_aff = MV_NR_AFF;
        dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(opt(argc, argv, "--lib", "out/res/kernels_moe.metallib"))] error:&err];
        if (!lib) { fprintf(stderr, "no metallib\n"); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        print_machine_state("start_", machine_state());
        printf("device=%s\nkernel=%s\niters=%d\nnr_seed=%d\nnr_aff=%d\n", dev.name.UTF8String, kname, iters, nr_seed, nr_aff);
        id<MTLCommandQueue> qq = [dev newCommandQueue];
        if (atoi(opt(argc, argv, "--dattn", "0"))) {   // decode attention: k_attn + k_attn_reduce, 1 row at position P, MoVA heads
            const int P = atoi(opt(argc, argv, "--dattn", "0")), nh = 32, nkv = 8, NP = P + 1;
            const int maxs = atoi(opt(argc, argv, "--splits", "64")), per = atoi(opt(argc, argv, "--per", "128"));
            int ns = (NP + per - 1) / per;
            if (ns > maxs) ns = maxs;
            AttnArgs a = {nh, nkv, (NP + ns - 1) / ns, ns, 0.08838834764831845f};
            id<MTLBuffer> q = rnd(4 * (size_t) nh * 128, 5), g = rnd(4 * (size_t) nh * 128, 6), o = rnd(4 * (size_t) nh * 128, 7);
            id<MTLBuffer> K = rnd(2 * (size_t) NP * nkv * 128, 8), V = rnd(2 * (size_t) NP * nkv * 128, 9);
            id<MTLBuffer> part = [dev newBufferWithLength:4 * (size_t) nh * ns * 130 options:MTLResourceStorageModePrivate];
            float* qp = (float*) q.contents, *gp = (float*) g.contents;
            for (size_t i = 0; i < (size_t) nh * 128; ++i) { qp[i] = (float) ((i * 37) % 101) / 101.0f - 0.5f; gp[i] = qp[i]; }
            uint16_t* kp = (uint16_t*) K.contents, *vp = (uint16_t*) V.contents;
            for (size_t i = 0; i < (size_t) NP * nkv * 128; ++i) { kp[i] = (uint16_t) (0x3C00 | (kp[i] & 0xFF)); vp[i] = (uint16_t) (0x3C00 | (vp[i] & 0xFF)); }
            id<MTLBuffer> ri = [dev newBufferWithLength:sizeof(RowInfo) options:MTLResourceStorageModeShared];
            ((RowInfo*) ri.contents)[0].pos = P;
            id<MTLComputePipelineState> pa = pipe_("k_attn", 0, 0), pr = pipe_("k_attn_reduce", 0, 0);
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep) {
                id<MTLCommandBuffer> c2 = [qq commandBuffer];
                id<MTLComputeCommandEncoder> e = [c2 computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                for (int k = 0; k < 48; ++k) {   // one per layer
                    [e setComputePipelineState:pa];
                    [e setBytes:&a length:sizeof a atIndex:0];
                    [e setBuffer:q offset:0 atIndex:1]; [e setBuffer:K offset:0 atIndex:2]; [e setBuffer:V offset:0 atIndex:3];
                    [e setBuffer:ri offset:0 atIndex:4]; [e setBuffer:part offset:0 atIndex:5];
                    [e dispatchThreadgroups:MTLSizeMake(ns, nkv, 1) threadsPerThreadgroup:MTLSizeMake(32 * ATT_SG, 1, 1)];
                    [e setComputePipelineState:pr];
                    [e setBytes:&a length:sizeof a atIndex:0];
                    [e setBuffer:part offset:0 atIndex:1]; [e setBuffer:g offset:0 atIndex:2]; [e setBuffer:o offset:0 atIndex:3];
                    [e dispatchThreadgroups:MTLSizeMake(nh, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [e endEncoding];
                [c2 commit];
                [c2 waitUntilCompleted];
                const double tt = (c2.GPUEndTime - c2.GPUStartTime) / 48;
                if (rep > 0 && tt < best) best = tt;
            }
            const double mb = (double) NP * nkv * 128 * 2 * 2 / 1e6;
            printf("attn_decode ctx %d (%d splits): %.1f us/layer, %.2f ms/token (48 layers), KV %.1f MB/layer, %.0f GB/s\n", NP, ns,
                   best * 1e6, best * 48e3, mb, mb / 1e3 / best);
            return 0;
        }
        if (atoi(opt(argc, argv, "--attn", "0"))) {   // k_attn_prefill: T rows at positions P0 .. P0 + T - 1, MoVA heads
            const int T = 512, P0 = atoi(opt(argc, argv, "--attn", "0")), nh = 32, nkv = 8, NP = P0 + T;
            id<MTLBuffer> q = rnd(4 * (size_t) T * nh * 128, 5), g = rnd(4 * (size_t) T * nh * 128, 6), o = rnd(4 * (size_t) T * nh * 128, 7);
            id<MTLBuffer> K = rnd(2 * (size_t) NP * nkv * 128, 8), V = rnd(2 * (size_t) NP * nkv * 128, 9);
            float* qp = (float*) q.contents, *gp = (float*) g.contents;
            for (size_t i = 0; i < (size_t) T * nh * 128; ++i) { qp[i] = (float) ((i * 37) % 101) / 101.0f - 0.5f; gp[i] = qp[i]; }
            uint16_t* kp = (uint16_t*) K.contents, *vp = (uint16_t*) V.contents;
            for (size_t i = 0; i < (size_t) NP * nkv * 128; ++i) { kp[i] = (uint16_t) (0x3C00 | (kp[i] & 0xFF)); vp[i] = (uint16_t) (0x3C00 | (vp[i] & 0xFF)); }
            id<MTLBuffer> ri = [dev newBufferWithLength:sizeof(RowInfo) * T options:MTLResourceStorageModeShared];
            for (int i = 0; i < T; ++i) ((RowInfo*) ri.contents)[i].pos = P0 + i;
            AttnArgs a = {nh, nkv, 0, 1, 0.08838834764831845f};
            const int32_t TT = T;
            id<MTLComputePipelineState> pl = pipe_("k_attn_prefill", 0, 0);
            double best = 1e30;
            const int it = iters < 20 ? iters : 20;
            for (int rep = 0; rep < 3; ++rep) {
                id<MTLCommandBuffer> c2 = [qq commandBuffer];
                id<MTLComputeCommandEncoder> e = [c2 computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                [e setComputePipelineState:pl];
                [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:q offset:0 atIndex:1]; [e setBuffer:K offset:0 atIndex:2]; [e setBuffer:V offset:0 atIndex:3];
                [e setBuffer:ri offset:0 atIndex:4]; [e setBuffer:g offset:0 atIndex:5]; [e setBuffer:o offset:0 atIndex:6];
                [e setBytes:&TT length:4 atIndex:7];
                for (int k = 0; k < it; ++k)
                    [e dispatchThreadgroups:MTLSizeMake((T + 8 * ATTF_RS - 1) / (8 * ATTF_RS), nkv, 1)
                      threadsPerThreadgroup:MTLSizeMake(32 * ATTF_G * ATTF_RS, 1, 1)];
                [e endEncoding];
                [c2 commit];
                [c2 waitUntilCompleted];
                const double tt = (c2.GPUEndTime - c2.GPUStartTime) / it;
                if (rep > 0 && tt < best) best = tt;
            }
            const double flops = 4.0 * 128 * nh * ((double) T * P0 + (double) T * (T + 1) / 2);
            printf("attn_prefill T %d after %d: %.3f ms/call, %.2f TFLOP/s (RS %d, BK %d)\n", T, P0, best * 1e3, flops / best / 1e12, ATTF_RS, ATTF_BK);
            return 0;
        }
        const Shape shapes[] = {
            {"attn_q (4096x2560)", 4096, 2560, 1, 0, 1},       {"attn_kv (1024x2560)", 1024, 2560, 1, 0, 1},
            {"attn_o (2560x4096)", 2560, 4096, 1, 0, 1},       {"shared_gu (768x2560)", 768, 2560, 1, 0, 1},
            {"shared_d (2560x768)", 2560, 768, 1, 0, 1},       {"head (250624x2560)", 250624, 2560, 1, 0, 1},
            {"expert_gu x8 (768x2560)", 768, 2560, 100, 8, 1}, {"expert_d x8 (2560x768)", 2560, 768, 100, 8, 1},
            {"value x4 (1024x2560)", 1024, 2560, 64, 4, 1},
        };
        const char* fnames[6] = {"bf16", "seed4", "q8", "q4", "seedp4", "seedp8"};
        uint32_t* G = malloc(65536 * 4);
        for (uint32_t s = 0; s < 65536; ++s) G[s] = lfsr_stream24((uint16_t) s);
        id<MTLBuffer> gb = [dev newBufferWithBytes:G length:65536 * 4 options:MTLResourceStorageModeShared];
        for (uint32_t s = 0; s < 65536; ++s) G[s] = lfsr_stream32((uint16_t) s);
        id<MTLBuffer> g32 = [dev newBufferWithBytes:G length:65536 * 4 options:MTLResourceStorageModeShared];
        printf("%-26s %-6s %10s %10s %9s\n", "shape", "format", "us/call", "MB/call", "GB/s");
        for (size_t si = 0; si < sizeof shapes / sizeof shapes[0]; ++si) {
            Shape s = shapes[si];
            int copies = 1;   // dense: rotate through enough copies (>= 512 MB of BF16) that each call reads from DRAM
            if (!s.P) { while ((size_t) copies * s.R * s.K * 2 < (512u << 20) && copies < 64) ++copies; s.slices = copies; }
            for (int fmt = 0; fmt < 6; ++fmt) {
                if (fmt == MF_SEED6P8 && s.P && s.slices != 64) continue;   // P = 8: dense tensors (and the value shape)
                const size_t n = (size_t) s.slices * s.R * s.K, nb = n / 8;
                const bool p4 = fmt == MF_SEED4P4 || fmt == MF_SEED6P8, p8 = fmt == MF_SEED6P8;
                const size_t wbytes = fmt == MF_BF16 ? 2 * n : fmt == MF_Q8 ? n : fmt == MF_Q4 ? n / 2 : 2 * nb;
                const size_t sbytes = p8 ? 4 * nb : fmt == MF_SEED4 || p4 ? 2 * nb : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (n / 64) : 16;
                const size_t bbytes = fmt == MF_SEED4 ? 4 * (size_t) s.slices : p4 ? 4 * (size_t) s.slices + nb / 2
                                    : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (n / 64) : 16;
                id<MTLBuffer> W = rnd(wbytes, 1), S = rnd(sbytes, 2), B = rnd(bbytes, 3);
                if (fmt == MF_Q8 || fmt == MF_Q4) {   // finite BF16 scales / biases
                    uint16_t* sp = (uint16_t*) S.contents, *bp = (uint16_t*) B.contents;
                    for (size_t i = 0; i < n / 64; ++i) { sp[i] = 0x3C00; bp[i] = 0xBC00; }
                }
                if (fmt == MF_SEED4 || p4) { int32_t* eb = (int32_t*) B.contents; for (int i = 0; i < s.slices; ++i) eb[i] = -18; }
                if (fmt == MF_BF16) { uint16_t* wp = (uint16_t*) W.contents; for (size_t i = 0; i < n; ++i) wp[i] = (uint16_t) (0x3C00 | (wp[i] & 0x00FF)); }
                const int P = s.P ? s.P : 1;
                id<MTLBuffer> X = rnd(4 * (size_t) P * s.K, 4), Y = [dev newBufferWithLength:4 * (size_t) P * s.R options:MTLResourceStorageModeShared];
                float* xp = (float*) X.contents;
                for (size_t i = 0; i < (size_t) P * s.K; ++i) xp[i] = (float) ((i * 37) % 101) / 101.0f - 0.5f;
                // per-call selections: P distinct slices, rotating
                const int nsel = 64;
                id<MTLBuffer> sel = [dev newBufferWithLength:4 * (size_t) nsel * P options:MTLResourceStorageModeShared];
                int32_t* sp = (int32_t*) sel.contents;
                for (int c = 0; c < nsel; ++c)
                    for (int p = 0; p < P; ++p) sp[c * P + p] = s.P ? (c * 13 + p * 7) % s.slices : 0;
                MvArgs a = {s.K, s.R, P, s.xdiv, s.K, s.R, 0, 0};
                id<MTLComputePipelineState> pl = pipe_(kname, fmt, s.P ? 0 : 1);
                const int rpt = MV_ROWS * (fmt == MF_SEED4 ? nr_seed : nr_aff);   // (P = 4: one row per simdgroup)
                double best = 1e30;
                for (int rep = 0; rep < 3; ++rep) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                    [e setComputePipelineState:pl];
                    [e setBytes:&a length:sizeof a atIndex:0];
                    [e setBuffer:W offset:0 atIndex:1];
                    [e setBuffer:S offset:0 atIndex:2];
                    [e setBuffer:B offset:0 atIndex:3];
                    [e setBuffer:X offset:0 atIndex:4];
                    [e setBuffer:Y offset:0 atIndex:5];
                    [e setBuffer:p4 ? g32 : gb offset:0 atIndex:7];
                    [e setBuffer:B offset:p4 ? 4 * (size_t) s.slices : 0 atIndex:8];   // P = 4 exponent nibbles
                    for (int it = 0; it < iters; ++it) {
                        [e setBuffer:sel offset:4 * (size_t) (it % nsel) * P atIndex:6];
                        if (!s.P) {   // dense copy it % copies
                            const size_t c = (size_t) (it % copies);
                            [e setBuffer:W offset:c * (wbytes / copies) atIndex:1];
                            if (fmt != MF_BF16) [e setBuffer:S offset:c * (sbytes / copies) atIndex:2];
                            if (fmt == MF_Q8 || fmt == MF_Q4) [e setBuffer:B offset:c * (bbytes / copies) atIndex:3];
                            if (p4) { [e setBuffer:B offset:4 * c atIndex:3]; [e setBuffer:B offset:4 * (size_t) s.slices + c * (nb / copies) / 2 atIndex:8]; }
                        }
                        [e dispatchThreadgroups:MTLSizeMake((NSUInteger) (s.R + rpt - 1) / rpt, (NSUInteger) (s.P ? s.P : 1), 1)
                          threadsPerThreadgroup:MTLSizeMake(32 * MV_ROWS, 1, 1)];
                    }
                    [e endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    const double t = (cb.GPUEndTime - cb.GPUStartTime) / iters;
                    if (rep > 0 && t < best) best = t;   // rep 0 = warm-up
                }
                // bytes one call reads: the selected slices' weights + scales/biases (or nibbles)
                const double frac = (double) P / s.slices;
                const double mb = ((double) wbytes + (fmt == MF_BF16 ? 0 : (double) sbytes + (fmt == MF_SEED4 ? 0 : (double) bbytes))) * frac / 1e6;
                printf("%-26s %-6s %10.1f %10.3f %9.1f\n", s.name, fnames[fmt], best * 1e6, mb, mb / 1e3 / best);
            }
        }
        if (atoi(opt(argc, argv, "--gu", "1"))) {   // fused gate + up + SwiGLU (k_mv_gu) vs k_mv twice + k_swiglu, expert shapes
            const int R = 768, K = 2560, SL = 100, P = 8;
            printf("%-26s %-6s %10s %10s\n", "experts gate+up+swiglu x8", "format", "us fused", "us 3-pass");
            for (int fmt = 0; fmt < 5; ++fmt) {
                const size_t n = (size_t) SL * R * K, nb = n / 8;
                const bool p4 = fmt == MF_SEED4P4;
                const size_t wbytes = fmt == MF_BF16 ? 2 * n : fmt == MF_Q8 ? n : fmt == MF_Q4 ? n / 2 : 2 * nb;
                const size_t sbytes = fmt == MF_SEED4 || p4 ? 2 * nb : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (n / 64) : 16;
                const size_t bbytes = fmt == MF_SEED4 ? 4 * (size_t) SL : p4 ? 4 * (size_t) SL + nb / 2 : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (n / 64) : 16;
                id<MTLBuffer> Wb = rnd(wbytes, 1), Sb = rnd(sbytes, 2), Bb = rnd(bbytes, 3);
                if (fmt == MF_Q8 || fmt == MF_Q4) { uint16_t* sp = (uint16_t*) Sb.contents, *bp = (uint16_t*) Bb.contents; for (size_t i = 0; i < n / 64; ++i) { sp[i] = 0x3C00; bp[i] = 0xBC00; } }
                if (fmt == MF_SEED4 || p4) { int32_t* eb = (int32_t*) Bb.contents; for (int i = 0; i < SL; ++i) eb[i] = -18; }
                if (fmt == MF_BF16) { uint16_t* wp = (uint16_t*) Wb.contents; for (size_t i = 0; i < n; ++i) wp[i] = (uint16_t) (0x3C00 | (wp[i] & 0x00FF)); }
                const size_t half_w = wbytes / 2, half_s = fmt == MF_BF16 ? 0 : sbytes / 2,
                             half_b = fmt == MF_SEED4 || p4 ? 4 * (SL / 2) : fmt == MF_BF16 ? 0 : bbytes / 2;
                const size_t nib0 = 4 * (size_t) SL, nib_half = nib0 + nb / 4;   // P = 4: the nibbles of slices 0.. and SL / 2..
                id<MTLBuffer> X = rnd(4 * (size_t) K, 4), G1 = [dev newBufferWithLength:4 * (size_t) P * R options:MTLResourceStorageModePrivate],
                              U1 = [dev newBufferWithLength:4 * (size_t) P * R options:MTLResourceStorageModePrivate],
                              A1 = [dev newBufferWithLength:4 * (size_t) P * R options:MTLResourceStorageModePrivate];
                float* xp = (float*) X.contents;
                for (int i = 0; i < K; ++i) xp[i] = (float) ((i * 37) % 101) / 101.0f - 0.5f;
                const int nsel = 32;
                id<MTLBuffer> sel = [dev newBufferWithLength:4 * (size_t) nsel * P options:MTLResourceStorageModeShared];
                for (int c = 0; c < nsel; ++c) for (int p = 0; p < P; ++p) ((int32_t*) sel.contents)[c * P + p] = (c * 13 + p * 7) % (SL / 2);
                MvArgs a = {K, R, P, P, K, R, 0, 0};
                const int32_t nn = P * R;
                id<MTLComputePipelineState> pf = pipe_("k_mv_gu", fmt, 0), pm = pipe_("k_mv", fmt, 0), ps = pipe_("k_swiglu", 0, 0);
                double best[2] = {1e30, 1e30};
                for (int mode = 0; mode < 2; ++mode)
                    for (int rep = 0; rep < 3; ++rep) {
                        id<MTLCommandBuffer> cb = [qq commandBuffer];
                        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                        for (int it = 0; it < iters; ++it) {
                            const size_t so = 4 * (size_t) (it % nsel) * P;
                            const MTLSize grid = MTLSizeMake((R + MV_ROWS - 1) / MV_ROWS, P, 1), tpg = MTLSizeMake(32 * MV_ROWS, 1, 1);
                            if (mode == 0) {
                                [e setComputePipelineState:pf];
                                [e setBytes:&a length:sizeof a atIndex:0];
                                [e setBuffer:Wb offset:0 atIndex:1]; [e setBuffer:Sb offset:0 atIndex:2]; [e setBuffer:Bb offset:0 atIndex:3];
                                [e setBuffer:X offset:0 atIndex:4]; [e setBuffer:A1 offset:0 atIndex:5]; [e setBuffer:sel offset:so atIndex:6];
                                [e setBuffer:p4 ? g32 : gb offset:0 atIndex:7];
                                [e setBuffer:Wb offset:half_w atIndex:8]; [e setBuffer:Sb offset:half_s atIndex:9]; [e setBuffer:Bb offset:half_b atIndex:10];
                                [e setBuffer:Bb offset:nib0 atIndex:11]; [e setBuffer:Bb offset:nib_half atIndex:12];
                                [e dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
                            } else {
                                for (int h = 0; h < 2; ++h) {
                                    [e setComputePipelineState:pm];
                                    [e setBytes:&a length:sizeof a atIndex:0];
                                    [e setBuffer:Wb offset:h ? half_w : 0 atIndex:1]; [e setBuffer:Sb offset:h ? half_s : 0 atIndex:2];
                                    [e setBuffer:Bb offset:h ? half_b : 0 atIndex:3];
                                    [e setBuffer:X offset:0 atIndex:4]; [e setBuffer:h ? U1 : G1 offset:0 atIndex:5]; [e setBuffer:sel offset:so atIndex:6];
                                    [e setBuffer:p4 ? g32 : gb offset:0 atIndex:7];
                                    [e setBuffer:Bb offset:h ? nib_half : nib0 atIndex:8];
                                    [e dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
                                }
                                [e setComputePipelineState:ps];
                                [e setBuffer:G1 offset:0 atIndex:0]; [e setBuffer:U1 offset:0 atIndex:1]; [e setBuffer:A1 offset:0 atIndex:2];
                                [e setBytes:&nn length:4 atIndex:3];
                                [e dispatchThreads:MTLSizeMake((NSUInteger) nn, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            }
                        }
                        [e endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const double tt = (cb.GPUEndTime - cb.GPUStartTime) / iters;
                        if (rep > 0 && tt < best[mode]) best[mode] = tt;
                    }
                printf("%-26s %-6s %10.1f %10.1f\n", "", fnames[fmt], best[0] * 1e6, best[1] * 1e6);
            }
        }
        print_machine_state("end_", machine_state());
        return 0;
    }
}
