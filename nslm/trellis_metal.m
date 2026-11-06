// nslm/trellis_metal.m - the trellis encoder of nslm/trellis.h on Metal (kernels: nslm/trellis.metal; API: trellis_gpu.h):
// the same words, scales and error as nslm_tq_encode_row, bit for bit (tests/test_trellis_gpu.c). The GPU runs each
// pass's Viterbi (k_tq_vit_SPT_K, one threadgroup a block) and its traceback (k_tq_trace). The host does what the C
// encoder does in double: the first scale, the weighted least-squares scale after each pass, the bits and the error.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"
#include "trellis.h"
#include "trellis_gpu.h"

// The host sums (the scales and the error) as nslm/lib_trellis.c, which the C build compiles with -ffp-contract=off; the
// Objective-C rule does not, so no fused multiply-adds here either
#pragma clang fp contract(off)

typedef struct { int L, K, T, nblk; int nt, spt; uint32_t back_stride; } TqVitArgs;   // = nslm/trellis.metal
typedef struct { uint32_t w0; int32_t h0; int32_t n; float a; } TqBlk;                 // = nslm/trellis.metal

#define TQ_BACK_BYTES (512u << 20)   // back pointers of one dispatch at most (L = 16: 1 MB a block of 256)

struct NslmTqGpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLLibrary> lib;
    id<MTLComputePipelineState> vit[9][9];   // [log2 states a thread][K], made on first use under mu
    id<MTLComputePipelineState> trace;
    pthread_mutex_t mu;                      // nslm-moe's workers encode on one NslmTqGpu at once
};

static void tq_free(NslmTqGpu* g) {
    if (!g) return;
    for (int i = 0; i < 9; ++i)
        for (int k = 0; k < 9; ++k) g->vit[i][k] = nil;
    g->trace = nil; g->lib = nil; g->q = nil; g->dev = nil;
    pthread_mutex_destroy(&g->mu);
    free(g);
}

NslmTqGpu* nslm_tq_gpu_open(char* err, int errlen) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { snprintf(err, (size_t) errlen, "no Metal device"); return NULL; }
        const char* res = getenv("NSLM_RES") ? getenv("NSLM_RES") : "out/res";
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithFormat:@"%s/trellis.metallib", res]] error:&e];
        id<MTLFunction> f = lib ? [lib newFunctionWithName:@"k_tq_trace"] : nil;
        id<MTLComputePipelineState> tr = f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
        if (!tr) {
            snprintf(err, (size_t) errlen, "%s/trellis.metallib (NSLM_RES): %s", res, e ? e.localizedDescription.UTF8String : "no k_tq_trace");
            return NULL;
        }
        NslmTqGpu* g = (NslmTqGpu*) calloc(1, sizeof(NslmTqGpu));
        g->dev = dev;
        g->lib = lib;
        g->trace = tr;
        g->q = [dev newCommandQueue];
        pthread_mutex_init(&g->mu, NULL);
        return g;
    }
}

void nslm_tq_gpu_close(NslmTqGpu* g) { tq_free(g); }

static id<MTLComputePipelineState> vit_pipe(NslmTqGpu* g, int spt, int K) {
    const int ls = __builtin_ctz((unsigned) spt);
    if (ls > 8 || K > 8) return nil;
    pthread_mutex_lock(&g->mu);
    if (!g->vit[ls][K]) {
        NSError* e = nil;
        id<MTLFunction> f = [g->lib newFunctionWithName:[NSString stringWithFormat:@"k_tq_vit_%d_%d", spt, K]];
        g->vit[ls][K] = f ? [g->dev newComputePipelineStateWithFunction:f error:&e] : nil;
    }
    id<MTLComputePipelineState> p = g->vit[ls][K];
    pthread_mutex_unlock(&g->mu);
    return p;
}

static void put_bits(uint32_t* row, uint64_t bit, uint32_t v, int nb) {   // as nslm/lib_trellis.c
    for (int i = 0; i < nb; ++i, ++bit)
        if (v >> i & 1u) row[bit >> 5] |= 1u << (bit & 31); else row[bit >> 5] &= ~(1u << (bit & 31));
}

int nslm_tq_gpu_encode(NslmTqGpu* g, const float* w, const float* h, int rows, int cols, int T, int L, int K, uint32_t* words,
                       uint16_t* scales, double* err, char* msg, int msglen) {
    @autoreleasepool {
        if (rows < 1 || cols < 1 || T < 1 || T > 65535 || L > 16 || K < 1 || K > 8 || 2 * K > L) {
            snprintf(msg, (size_t) msglen, "bad shape or L, K (rows %d cols %d T %d L %d K %d)", rows, cols, T, L, K);
            return -1;
        }
        const int N = 1 << L, H = 1 << K, spt = H > N / 1024 ? H : N / 1024, nt = N / spt;
        id<MTLComputePipelineState> vp = vit_pipe(g, spt, K);
        if (!vp || (int) vp.maxTotalThreadsPerThreadgroup < nt) {
            snprintf(msg, (size_t) msglen, "k_tq_vit_%d_%d: %s", spt, K, vp ? "too few threads a threadgroup" : "no kernel");
            return -1;
        }
        const int nbr = (cols + T - 1) / T;
        const size_t nb = (size_t) rows * nbr, nw = nslm_tq_row_words(cols, T, L, K);
        const uint64_t nbits = nslm_tq_block_bits(T, L, K);
        id<MTLBuffer> W = [g->dev newBufferWithBytes:w length:sizeof(float) * (size_t) rows * cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> Hb = h ? [g->dev newBufferWithBytes:h length:sizeof(float) * (size_t) cols options:MTLResourceStorageModeShared]
                             : [g->dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        const uint32_t bstride = (uint32_t) T * (uint32_t) nt;   // back-pointer words of a block
        size_t batch = TQ_BACK_BYTES / (4 * (size_t) bstride);
        if (batch < 1) batch = 1;
        if (batch > nb) batch = nb;
        id<MTLBuffer> Bk = [g->dev newBufferWithLength:4 * (size_t) bstride * batch options:MTLResourceStorageModePrivate];
        id<MTLBuffer> Blk = [g->dev newBufferWithLength:sizeof(TqBlk) * batch options:MTLResourceStorageModeShared];
        id<MTLBuffer> Fin = [g->dev newBufferWithLength:4 * batch options:MTLResourceStorageModeShared];
        id<MTLBuffer> Path = [g->dev newBufferWithLength:2 * (size_t) T * batch options:MTLResourceStorageModeShared];
        float* a = (float*) malloc(sizeof(float) * nb);
        uint16_t* st = (uint16_t*) calloc(nb * (size_t) T, 2);   // each block's path of its last pass
        size_t* act = (size_t*) malloc(sizeof(size_t) * nb);
        if (!W || !Hb || !Bk || !Blk || !Fin || !Path || !a || !st || !act) {
            free(a); free(st); free(act);
            snprintf(msg, (size_t) msglen, "out of memory");
            return -1;
        }
        for (size_t bi = 0; bi < nb; ++bi) {   // the first scale, as nslm_tq_encode_row
            const int r = (int) (bi / nbr), j0 = (int) (bi % nbr) * T, n = cols - j0 < T ? cols - j0 : T;
            const float* wb = w + (size_t) r * cols + j0, *hb = h ? h + j0 : NULL;
            double sw = 0, sh = 0;
            for (int t = 0; t < n; ++t) { const double ht = hb ? hb[t] : 1.0; sw += ht * wb[t] * wb[t]; sh += ht; }
            a[bi] = sh > 0 ? (float) (sqrt(sw / sh) / 147.8) : 0.0f;
        }
        int rc = 0;
        for (int pass = 0; pass < 3 && !rc; ++pass) {
            size_t na = 0;
            for (size_t bi = 0; bi < nb; ++bi) if (a[bi] != 0.0f) act[na++] = bi;
            for (size_t c0 = 0; c0 < na && !rc; c0 += batch) {
                const size_t cn = na - c0 < batch ? na - c0 : batch;
                TqBlk* bl = (TqBlk*) Blk.contents;
                for (size_t x = 0; x < cn; ++x) {
                    const size_t bi = act[c0 + x];
                    const int r = (int) (bi / nbr), j0 = (int) (bi % nbr) * T;
                    bl[x] = (TqBlk) {(uint32_t) ((size_t) r * cols + j0), h ? j0 : -1, cols - j0 < T ? cols - j0 : T, a[bi]};
                }
                const TqVitArgs va = {L, K, T, (int) cn, nt, spt, bstride};
                id<MTLCommandBuffer> cb = [g->q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:vp];
                [enc setBytes:&va length:sizeof va atIndex:0];
                [enc setBuffer:W offset:0 atIndex:1];
                [enc setBuffer:Hb offset:0 atIndex:2];
                [enc setBuffer:Blk offset:0 atIndex:3];
                [enc setBuffer:Bk offset:0 atIndex:4];
                [enc setBuffer:Fin offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(cn, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger) nt, 1, 1)];
                [enc setComputePipelineState:g->trace];
                [enc setBuffer:Path offset:0 atIndex:6];
                [enc dispatchThreads:MTLSizeMake(cn, 1, 1) threadsPerThreadgroup:MTLSizeMake(cn < 64 ? cn : 64, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) { snprintf(msg, (size_t) msglen, "trellis encode: %s", cb.error.localizedDescription.UTF8String); rc = -1; break; }
                const uint16_t* pp = (const uint16_t*) Path.contents;
                for (size_t x = 0; x < cn; ++x) {   // the path, then the weighted least-squares scale for it
                    const size_t bi = act[c0 + x];
                    const int r = (int) (bi / nbr), j0 = (int) (bi % nbr) * T, n = cols - j0 < T ? cols - j0 : T;
                    const float* wb = w + (size_t) r * cols + j0, *hb = h ? h + j0 : NULL;
                    uint16_t* sp = st + bi * (size_t) T;
                    memcpy(sp, pp + x * (size_t) T, 2 * (size_t) n);
                    double num = 0, den = 0;
                    for (int t = 0; t < n; ++t) {
                        const double ht = hb ? hb[t] : 1.0, q = nslm_tq_code((uint32_t) sp[t]);
                        num += ht * wb[t] * q; den += ht * q * q;
                    }
                    if (den > 0) a[bi] = (float) (num / den);
                }
            }
        }
        if (!rc) {
            memset(words, 0, sizeof(uint32_t) * nw * (size_t) rows);
            for (int r = 0; r < rows; ++r) {
                double e = 0;
                uint32_t* row = words + nw * (size_t) r;
                for (int b = 0, j0 = 0; j0 < cols; ++b, j0 += T) {
                    const size_t bi = (size_t) r * nbr + b;
                    const int n = cols - j0 < T ? cols - j0 : T;
                    const float* wb = w + (size_t) r * cols + j0, *hb = h ? h + j0 : NULL;
                    uint16_t* sp = st + bi * (size_t) T;
                    if (a[bi] == 0.0f) memset(sp, 0, 2 * (size_t) n);
                    scales[(size_t) r * nbr + b] = nslm_f2bf(a[bi]);
                    const float ar = nslm_bf2f(scales[(size_t) r * nbr + b]);
                    const uint64_t base = (uint64_t) b * nbits;
                    put_bits(row, base, sp[0], L);
                    for (int t = 1; t < n; ++t) put_bits(row, base + (uint64_t) K * t + (uint64_t) (L - K), (uint32_t) sp[t] >> (L - K), K);
                    for (int t = 0; t < n; ++t) {
                        const float v = ar * (float) nslm_tq_code(sp[t]), d = wb[t] - v;
                        e += (hb ? hb[t] : 1.0) * d * d;
                    }
                }
                if (err) err[r] = e;
            }
        }
        free(a); free(st); free(act);
        return rc;
    }
}
