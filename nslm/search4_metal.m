// nslm/search4_metal.m - Metal glue for the P = 4 seed search (nslm/search4_gpu.h; the kernel is nslm/search4.metal).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "search4_gpu.h"

struct Nslm4Gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLComputePipelineState> pipe;
};

Nslm4Gpu* nslm4_gpu_open(const char* metallib, char* err, int errlen) {
    @autoreleasepool {
        Nslm4Gpu* g = (Nslm4Gpu*) calloc(1, sizeof(Nslm4Gpu));
        g->dev = MTLCreateSystemDefaultDevice();
        if (!g->dev) { snprintf(err, errlen, "no Metal device"); free(g); return NULL; }
        NSError* e = nil;
        id<MTLLibrary> lib = g->dev ? [g->dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib)] error:&e] : nil;
        id<MTLFunction> f = lib ? [lib newFunctionWithName:@"k_seed_search4"] : nil;
        g->pipe = f ? [g->dev newComputePipelineStateWithFunction:f error:&e] : nil;
        if (!g->pipe) {
            snprintf(err, errlen, "%s: %s", metallib, e ? e.localizedDescription.UTF8String : "no k_seed_search4");
            g->dev = nil;
            free(g);
            return NULL;
        }
        g->q = [g->dev newCommandQueue];
        return g;
    }
}

int nslm4_gpu_search(Nslm4Gpu* g, const float* w, int rows, int cols, const float* sh, int bias, const Search4Opts* o,
                     uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err, char* msg, int msglen) {
    @autoreleasepool {
        if (cols % 8 || rows < 1 || o->n_exp < 1 || o->n_exp > 3) { snprintf(msg, msglen, "bad shape or options"); return -1; }
        const int ng = cols / 8;
        const size_t nb = (size_t) rows * ng;
        id<MTLBuffer> W = [g->dev newBufferWithBytes:w length:sizeof(float) * (size_t) rows * cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> SH = [g->dev newBufferWithLength:sizeof(float) * (size_t) cols options:MTLResourceStorageModeShared];
        float* shp = (float*) SH.contents;
        for (int c = 0; c < cols; ++c) shp[c] = sh ? sh[c] : 1.0f;
        id<MTLBuffer> So = [g->dev newBufferWithLength:2 * nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> Co = [g->dev newBufferWithLength:2 * nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> Eo = [g->dev newBufferWithLength:nb options:MTLResourceStorageModeShared];
        id<MTLBuffer> Ro = [g->dev newBufferWithLength:4 * nb options:MTLResourceStorageModeShared];
        // column groups per command buffer: about 5e9 block-seeds, so no command buffer runs long
        int gstep = (int) (5e9 / ((double) rows * o->n_seeds));
        if (gstep < 1) gstep = 1;
        if (gstep > ng) gstep = ng;
        NSMutableArray<id<MTLCommandBuffer>>* cbs = [NSMutableArray array];
        for (int g0 = 0; g0 < ng; g0 += gstep) {
            const int gn = ng - g0 < gstep ? ng - g0 : gstep;
            Search4Args a = {rows, cols, g0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}};
            id<MTLCommandBuffer> cb = [g->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g->pipe];
            [enc setBytes:&a length:sizeof a atIndex:0];
            [enc setBuffer:W offset:0 atIndex:1];
            [enc setBuffer:SH offset:0 atIndex:2];
            [enc setBuffer:So offset:0 atIndex:3];
            [enc setBuffer:Co offset:0 atIndex:4];
            [enc setBuffer:Eo offset:0 atIndex:5];
            [enc setBuffer:Ro offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger) gn, (NSUInteger) ((rows + S4_ROWS - 1) / S4_ROWS), 1)
                threadsPerThreadgroup:MTLSizeMake(S4_TPB, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cbs addObject:cb];
        }
        for (id<MTLCommandBuffer> cb in cbs) {
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) { snprintf(msg, msglen, "GPU error: %s", cb.error.localizedDescription.UTF8String); return -1; }
        }
        memcpy(seed, So.contents, 2 * nb);
        memcpy(coef, Co.contents, 2 * nb);
        memcpy(ecode, Eo.contents, nb);
        if (err) memcpy(err, Ro.contents, 4 * nb);
        return 0;
    }
}

void nslm4_gpu_close(Nslm4Gpu* g) {
    if (!g) return;
    g->pipe = nil; g->q = nil; g->dev = nil;
    free(g);
}
