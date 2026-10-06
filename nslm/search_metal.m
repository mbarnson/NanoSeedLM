// nslm/search_metal.m - Metal glue for the GPU seed search (nslm/search_gpu.h; the kernel is nslm/search.metal).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"
#include "search_gpu.h"

struct NslmGpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLComputePipelineState> pipe, pipe_full;   // sqrt(h) / unweighted search; the full-transform mode
    id<MTLComputePipelineState> pre, pre_full;     // the fragile-seed pre-pass (k_seed_exact)
    id<MTLBuffer> fl1, ex1;                        // its buffers for nslm_gpu_search_block (one column group)
    id<MTLBuffer> G;   // per-seed stream table (lfsr_stream24)
    float exact_kappa; // fragile-seed threshold for the exact double table (SearchArgs.exact_kappa)
};

NslmGpu* nslm_gpu_open(const char* metallib, char* err, int errlen) {
    @autoreleasepool {
        NslmGpu* g = (NslmGpu*) calloc(1, sizeof(NslmGpu));
        g->dev = MTLCreateSystemDefaultDevice();
        if (!g->dev) { snprintf(err, errlen, "no Metal device"); free(g); return NULL; }
        NSError* e = nil;
        id<MTLLibrary> lib = [g->dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib)] error:&e];
        for (int full = 0; full <= 1; ++full) {
            MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
            bool fb = full != 0;
            [cv setConstantValue:&fb type:MTLDataTypeBool atIndex:0];
            id<MTLFunction> f = lib ? [lib newFunctionWithName:@"k_seed_search" constantValues:cv error:&e] : nil;
            id<MTLComputePipelineState> p = f ? [g->dev newComputePipelineStateWithFunction:f error:&e] : nil;
            id<MTLFunction> fx = lib ? [lib newFunctionWithName:@"k_seed_exact" constantValues:cv error:&e] : nil;
            id<MTLComputePipelineState> px = fx ? [g->dev newComputePipelineStateWithFunction:fx error:&e] : nil;
            if (full) { g->pipe_full = p; g->pre_full = px; } else { g->pipe = p; g->pre = px; }
        }
        if (!g->pipe || !g->pipe_full || !g->pre || !g->pre_full) {
            snprintf(err, errlen, "%s: %s", metallib, e ? e.localizedDescription.UTF8String : "no k_seed_search");
            g->dev = nil;
            free(g);
            return NULL;
        }
        g->q = [g->dev newCommandQueue];
        // default: every seed's table from the exact double emulation (bit-identical to the CPU's);
        // NSLM_EXACT_KAPPA=K limits it to seeds with condition estimate > K (faster, near-ties possible)
        const char* ek = getenv("NSLM_EXACT_KAPPA");
        g->exact_kappa = ek ? (float) atof(ek) : 0.0f;
        g->G = [g->dev newBufferWithLength:65536 * 4 options:MTLResourceStorageModeShared];
        uint32_t* t = (uint32_t*) g->G.contents;
        t[0] = 0;
        for (int s = 1; s < 65536; ++s) t[s] = lfsr_stream24((uint16_t) s);
        return g;
    }
}

int nslm_gpu_search(NslmGpu* g, const float* w, int rows, int cols, const float* sh, int bias, const SearchOpts* o,
                    int prune, uint16_t* seed, uint16_t* nib, float* err, char* msg, int msglen) {
    @autoreleasepool {
        if (cols % 8 || rows < 1 || o->n_exp < 1 || o->n_exp > 3) { snprintf(msg, msglen, "bad shape or options"); return -1; }
        const int ng = cols / 8;
        const size_t nb = (size_t) rows * ng;
        id<MTLBuffer> W = [g->dev newBufferWithBytes:w length:sizeof(float) * (size_t) rows * cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> SH = [g->dev newBufferWithLength:sizeof(float) * (size_t) cols options:MTLResourceStorageModeShared];
        float* shp = (float*) SH.contents;
        for (int c = 0; c < cols; ++c) shp[c] = sh ? sh[c] : 1.0f;   // sqrt(h) = 1: f32-exact no-op
        id<MTLBuffer> So = [g->dev newBufferWithLength:2 * nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> No = [g->dev newBufferWithLength:2 * nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> Eo = [g->dev newBufferWithLength:4 * nb options:MTLResourceStorageModeShared];
        // column groups per command buffer: ~1e10 block-seeds, so no command buffer runs long
        const double per_group = (double) rows * o->n_seeds;
        int gstep = (int) (1e10 / per_group);
        if (gstep < 1) gstep = 1;
        if (gstep > ng) gstep = ng;
        if (gstep > 32) gstep = 32;   // the pre-pass buffers: 65536 x 49 bytes per column group of a batch
        id<MTLBuffer> FL = [g->dev newBufferWithLength:(size_t) gstep * 65536 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> EX = [g->dev newBufferWithLength:(size_t) gstep * 65536 * 48 options:MTLResourceStorageModePrivate];
        NSMutableArray<id<MTLCommandBuffer>>* cbs = [NSMutableArray array];
        for (int g0 = 0; g0 < ng; g0 += gstep) {
            const int gn = ng - g0 < gstep ? ng - g0 : gstep;
            SearchArgs a = {rows, cols, g0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}, prune,
                            g->exact_kappa, 0};
            id<MTLCommandBuffer> cb = [g->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            [enc setComputePipelineState:g->pre];
            [enc setBytes:&a length:sizeof a atIndex:0];
            [enc setBuffer:SH offset:0 atIndex:2];
            [enc setBuffer:g->G offset:0 atIndex:3];
            [enc setBuffer:FL offset:0 atIndex:7];
            [enc setBuffer:EX offset:0 atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) gn, 256, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setComputePipelineState:g->pipe];
            [enc setBytes:&a length:sizeof a atIndex:0];
            [enc setBuffer:W offset:0 atIndex:1];
            [enc setBuffer:SH offset:0 atIndex:2];
            [enc setBuffer:g->G offset:0 atIndex:3];
            [enc setBuffer:So offset:0 atIndex:4];
            [enc setBuffer:No offset:0 atIndex:5];
            [enc setBuffer:Eo offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) gn, (NSUInteger) ((rows + SG_ROWS - 1) / SG_ROWS), 1)
                threadsPerThreadgroup:MTLSizeMake(SG_TPB, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cbs addObject:cb];
        }
        for (id<MTLCommandBuffer> cb in cbs) {
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) {
                snprintf(msg, msglen, "GPU error: %s", cb.error.localizedDescription.UTF8String);
                return -1;
            }
        }
        memcpy(seed, So.contents, 2 * nb);
        memcpy(nib, No.contents, 2 * nb);
        if (err) memcpy(err, Eo.contents, 4 * nb);
        return 0;
    }
}

int nslm_gpu_search_block(NslmGpu* g, const float* w8, int rows, const float a[64], int bias, const SearchOpts* o,
                          uint16_t* seed, uint16_t* nib, float* err, char* msg, int msglen) {
    @autoreleasepool {
        id<MTLBuffer> W = [g->dev newBufferWithBytes:w8 length:sizeof(float) * (size_t) rows * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Ab = [g->dev newBufferWithBytes:a length:sizeof(float) * 64 options:MTLResourceStorageModeShared];
        id<MTLBuffer> So = [g->dev newBufferWithLength:2 * (size_t) rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> No = [g->dev newBufferWithLength:2 * (size_t) rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> Eo = [g->dev newBufferWithLength:4 * (size_t) rows options:MTLResourceStorageModeShared];
        SearchArgs sa = {rows, 8, 0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}, 1,
                          g->exact_kappa, 1};
        if (!g->fl1) {
            g->fl1 = [g->dev newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
            g->ex1 = [g->dev newBufferWithLength:65536 * 48 options:MTLResourceStorageModePrivate];
        }
        id<MTLCommandBuffer> cb = [g->q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        [enc setComputePipelineState:g->pre_full];
        [enc setBytes:&sa length:sizeof sa atIndex:0];
        [enc setBuffer:Ab offset:0 atIndex:2];
        [enc setBuffer:g->G offset:0 atIndex:3];
        [enc setBuffer:g->fl1 offset:0 atIndex:7];
        [enc setBuffer:g->ex1 offset:0 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(1, 256, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:g->pipe_full];
        [enc setBytes:&sa length:sizeof sa atIndex:0];
        [enc setBuffer:W offset:0 atIndex:1];
        [enc setBuffer:Ab offset:0 atIndex:2];
        [enc setBuffer:g->G offset:0 atIndex:3];
        [enc setBuffer:So offset:0 atIndex:4];
        [enc setBuffer:No offset:0 atIndex:5];
        [enc setBuffer:Eo offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger) ((rows + SG_ROWS - 1) / SG_ROWS), 1) threadsPerThreadgroup:MTLSizeMake(SG_TPB, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { snprintf(msg, msglen, "GPU error: %s", cb.error.localizedDescription.UTF8String); return -1; }
        memcpy(seed, So.contents, 2 * (size_t) rows);
        memcpy(nib, No.contents, 2 * (size_t) rows);
        if (err) memcpy(err, Eo.contents, 4 * (size_t) rows);
        return 0;
    }
}

void nslm_gpu_close(NslmGpu* g) {
    if (!g) return;
    g->pipe = nil; g->pipe_full = nil; g->pre = nil; g->pre_full = nil; g->fl1 = nil; g->ex1 = nil; g->q = nil; g->G = nil; g->dev = nil;
    free(g);
}
