// tests/test_search_gpu.m - GPU seed table (nslm/search.metal) vs CPU (nslm/lib_search.c): every seed's U, Gi, R bit
// for bit, unweighted and with extreme sqrt(h) (one channel ~2700x the others).  The float-float build is reported;
// the exact double emulation must match on every seed.
//
//   test_search_gpu [out/res/search.metallib]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "lfsr.h"
#include "search.h"

static SeedTab g_tab[65536];

int main(int argc, char** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argc > 1 ? argv[1] : "out/res/search.metallib")] error:&e];
        id<MTLComputePipelineState> p = lib ? [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_seedtab_test"] error:&e] : nil;
        if (!p) { fprintf(stderr, "no k_seedtab_test: %s\n", e.localizedDescription.UTF8String); return 2; }
        id<MTLBuffer> G = [dev newBufferWithLength:65536 * 4 options:MTLResourceStorageModeShared];
        for (int s = 1; s < 65536; ++s) ((uint32_t*) G.contents)[s] = lfsr_stream24((uint16_t) s);
        const float shs[3][8] = {{1, 1, 1, 1, 1, 1, 1, 1},
                                 {0.1012f, 0.07429f, 0.06921f, 0.06821f, 0.08077f, 0.06578f, 185.2f, 0.06626f},
                                 {0.9f, 3.7f, 0.02f, 1.1f, 12.5f, 0.6f, 0.33f, 2.2f}};
        const char* names[3] = {"unweighted", "extreme sqrt(h) (L3 down cg216)", "mixed sqrt(h)"};
        int fail = 0;
        for (int w = 0; w < 3; ++w) {
            nslm_seedtab_build_weighted(g_tab, shs[w], 1, 65536);
            for (int mode = 0; mode <= 1; ++mode) {
                id<MTLBuffer> SH = [dev newBufferWithBytes:shs[w] length:32 options:MTLResourceStorageModeShared];
                id<MTLBuffer> O = [dev newBufferWithLength:65536 * 36 * 4 options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:p];
                [enc setBuffer:SH offset:0 atIndex:0];
                [enc setBuffer:G offset:0 atIndex:1];
                [enc setBuffer:O offset:0 atIndex:2];
                [enc setBytes:&mode length:4 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(65536, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const float* o = (const float*) O.contents;
                int badU = 0, badGi = 0, badR = 0;
                for (int s = 1; s < 65536; ++s) {
                    const float* g = o + (size_t) s * 36;
                    badU += memcmp(g, g_tab[s].U, 96) != 0;
                    badGi += memcmp(g + 24, g_tab[s].Gi, 24) != 0;
                    badR += memcmp(g + 30, g_tab[s].R, 24) != 0;
                }
                printf("%-34s %-14s U %s, Gi differs on %5d seeds, R differs on %5d seeds\n", names[w],
                       mode ? "exact double" : "float-float", badU ? "DIFFERS" : "identical", badGi, badR);
                if (mode || badU) fail += badU + badGi + badR > 0;
            }
        }
        printf("test_search_gpu: %s\n", fail ? "FAIL" : "PASS");
        return fail ? 1 : 0;
    }
}
