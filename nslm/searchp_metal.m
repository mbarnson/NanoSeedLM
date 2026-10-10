// nslm/searchp_metal.m - Metal glue for the P = 3 / 8 seed search (nslm/searchp_gpu.h; the kernel is nslm/searchp.metal).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "searchp_gpu.h"

struct NslmPGpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLComputePipelineState> pipe[2][2];   // [P = 3, 8][sqrt(h), a full 8 x 8 transform per column group]
};

NslmPGpu* nslmp_gpu_open(const char* metallib, char* err, int errlen) {
    @autoreleasepool {
        NslmPGpu* g = (NslmPGpu*) calloc(1, sizeof(NslmPGpu));
        g->dev = MTLCreateSystemDefaultDevice();
        NSError* e = nil;
        id<MTLLibrary> lib = g->dev ? [g->dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib)] error:&e] : nil;
        int ok = lib != nil;
        for (int pi = 0; pi < 2; ++pi)
            for (int full = 0; full <= 1; ++full) {
                MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
                bool fb = full != 0;
                int P = pi ? 8 : 3;
                [cv setConstantValue:&fb type:MTLDataTypeBool atIndex:0];
                [cv setConstantValue:&P type:MTLDataTypeInt atIndex:1];
                id<MTLFunction> f = lib ? [lib newFunctionWithName:@"k_seed_searchp" constantValues:cv error:&e] : nil;
                g->pipe[pi][full] = f ? [g->dev newComputePipelineStateWithFunction:f error:&e] : nil;
                ok &= g->pipe[pi][full] != nil;
            }
        if (!ok) {
            snprintf(err, errlen, "%s: %s", metallib, e ? e.localizedDescription.UTF8String : "no k_seed_searchp");
            g->dev = nil;
            free(g);
            return NULL;
        }
        g->q = [g->dev newCommandQueue];
        return g;
    }
}

int nslmp_gpu_search(NslmPGpu* g, int P, const float* w, int rows, int cols, const float* sh, const float* A, int bias,
                     const Search4Opts* o, uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err, char* msg, int msglen) {
    @autoreleasepool {
        if (cols % 8 || rows < 1 || o->n_exp < 1 || o->n_exp > 3 || (P != 3 && P != 8)) { snprintf(msg, msglen, "bad shape or options"); return -1; }
        const int ng = cols / 8;
        const size_t nb = (size_t) rows * ng;
        id<MTLBuffer> W = [g->dev newBufferWithBytes:w length:sizeof(float) * (size_t) rows * cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> SH;
        if (A) SH = [g->dev newBufferWithBytes:A length:sizeof(float) * 64 * (size_t) ng options:MTLResourceStorageModeShared];
        else {
            SH = [g->dev newBufferWithLength:sizeof(float) * (size_t) cols options:MTLResourceStorageModeShared];
            float* shp = (float*) SH.contents;
            for (int c = 0; c < cols; ++c) shp[c] = sh ? sh[c] : 1.0f;
        }
        id<MTLBuffer> So = [g->dev newBufferWithLength:2 * nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> Co = [g->dev newBufferWithLength:4 * nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> Eo = [g->dev newBufferWithLength:nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> Ro = [g->dev newBufferWithLength:4 * nb options:MTLResourceStorageModeShared];
        // column groups per command buffer: about 1.5e9 block-seeds at P = 8 (more at P = 3), so no command buffer runs long
        int gstep = (int) ((P == 8 ? 1.5e9 : 6e9) / ((double) rows * o->n_seeds));
        if (gstep < 1) gstep = 1;
        if (gstep > ng) gstep = ng;
        NSMutableArray<id<MTLCommandBuffer>>* cbs = [NSMutableArray array];
        for (int g0 = 0; g0 < ng; g0 += gstep) {
            const int gn = ng - g0 < gstep ? ng - g0 : gstep;
            SearchPArgs a = {rows, cols, g0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}};
            id<MTLCommandBuffer> cb = [g->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g->pipe[P == 8][A != NULL]];
            [enc setBytes:&a length:sizeof a atIndex:0];
            [enc setBuffer:W offset:0 atIndex:1];
            [enc setBuffer:SH offset:0 atIndex:2];
            [enc setBuffer:So offset:0 atIndex:3];
            [enc setBuffer:Co offset:0 atIndex:4];
            [enc setBuffer:Eo offset:0 atIndex:5];
            [enc setBuffer:Ro offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) gn, (NSUInteger) ((rows + SP_ROWS - 1) / SP_ROWS), 1)
                threadsPerThreadgroup:MTLSizeMake(SP_TPB, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cbs addObject:cb];
        }
        for (id<MTLCommandBuffer> cb in cbs) {
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) { snprintf(msg, msglen, "GPU error: %s", cb.error.localizedDescription.UTF8String); return -1; }
        }
        memcpy(seed, So.contents, 2 * nb);
        memcpy(coef, Co.contents, 4 * nb);
        memcpy(ecode, Eo.contents, nb);
        if (err) memcpy(err, Ro.contents, 4 * nb);
        return 0;
    }
}

void nslmp_gpu_close(NslmPGpu* g) {
    if (!g) return;
    for (int pi = 0; pi < 2; ++pi) g->pipe[pi][0] = g->pipe[pi][1] = nil;
    g->q = nil; g->dev = nil;
    free(g);
}
