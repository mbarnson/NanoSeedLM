// tests/kernel_backend_metal.m - tests/kernel_backend.h on the Metal kernels (engine/kernels_moe.metal, from
// out/res/kernels_moe.metallib), dispatched as the Metal engine dispatches them.  Shared-storage buffers: outputs are
// read straight from them.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

#include "kernel_backend.h"
#include "lfsr.h"

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;
static id<MTLBuffer> g24, g32;   // the seed tables: lfsr_stream24 (SEED4) and lfsr_stream32 (SEED4P4)

static id<MTLBuffer> buf(const void* p, size_t n) {
    id<MTLBuffer> b = [dev newBufferWithLength:n > 0 ? n : 16 options:MTLResourceStorageModeShared];
    if (p) memcpy(b.contents, p, n);
    else memset(b.contents, 0, n > 0 ? n : 16);
    return b;
}
static id<MTLComputePipelineState> pipe_(const char* name, int fmt, int T) {
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    short f = (short) fmt, t = (short) T;
    [cv setConstantValue:&f type:MTLDataTypeShort atIndex:0];
    [cv setConstantValue:&t type:MTLDataTypeShort atIndex:1];
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&err];
    id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!p) printf("FAIL: pipeline %s: %s\n", name, err ? err.localizedDescription.UTF8String : "not found");
    return p;
}
typedef void (^Enc)(id<MTLComputeCommandEncoder>);
static int run(Enc enc) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc(e);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { printf("FAIL: command buffer: %s\n", cb.error.localizedDescription.UTF8String); return -1; }
    return 0;
}
static id<MTLBuffer> table(uint32_t (*f)(uint16_t)) {
    uint32_t* g = malloc(65536 * 4);
    for (uint32_t s = 0; s < 65536; ++s) g[s] = f((uint16_t) s);
    id<MTLBuffer> b = buf(g, 65536 * 4);
    free(g);
    return b;
}
static uint32_t stream24(uint16_t s) { return lfsr_stream24(s); }
static uint32_t stream32(uint16_t s) { return lfsr_stream32(s); }

int kt_open(char* err, int errlen) {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) { snprintf(err, (size_t) errlen, "no Metal device"); return -1; }
    queue = [dev newCommandQueue];
    NSError* e = nil;
    lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"out/res/kernels_moe.metallib"] error:&e];
    if (!lib) { snprintf(err, (size_t) errlen, "no out/res/kernels_moe.metallib"); return -1; }
    g24 = table(stream24);
    g32 = table(stream32);
    return 0;
}
void kt_close(void) { lib = nil; queue = nil; dev = nil; g24 = g32 = nil; }
const char* kt_name(void) { return "Metal"; }
int kt_mm_tile(void) { return MM_BN; }
int kt_attn_key_tile(void) { return ATTF_BK; }
int kt_seed_gemm_exact(int on) { (void) on; return 0; }   // the Metal GEMM always multiplies the f32 seed weights

// The weight's buffers (the kernels address slices themselves) and its seed table.
typedef struct { id<MTLBuffer> W, S, B, N, G; } MW;
static MW mw(const KtWeight* w) {
    MW m;
    m.W = buf(w->w, w->wn);
    m.S = buf(w->s, w->sn);
    m.B = buf(w->b, w->bn);
    m.N = buf(w->e, w->en);
    m.G = w->fmt == MF_SEED4P4 ? g32 : g24;
    return m;
}
static int mv_dispatch(const KtWeight* w, MvArgs a, int T, int groups, id<MTLBuffer> xb, id<MTLBuffer> yb, id<MTLBuffer> sel) {
    const MW m = mw(w);
    id<MTLComputePipelineState> p = pipe_("k_mv", w->fmt, T);
    if (!p) return -1;
    return run(^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:p];
        [e setBytes:&a length:sizeof a atIndex:0];
        [e setBuffer:m.W offset:0 atIndex:1]; [e setBuffer:m.S offset:0 atIndex:2]; [e setBuffer:m.B offset:0 atIndex:3];
        [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:sel offset:0 atIndex:6];
        [e setBuffer:m.G offset:0 atIndex:7]; [e setBuffer:m.N offset:0 atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake((w->R + MV_RPT(w->fmt) - 1) / MV_RPT(w->fmt), groups, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * MV_ROWS, 1, 1)];
    });
}
int kt_mv(const KtWeight* w, int T, int add, const float* x, float* y) {
    const size_t yn = 4 * (size_t) T * w->R;
    id<MTLBuffer> xb = buf(x, 4 * (size_t) T * w->K), yb = buf(y, yn), sel = buf((int32_t[]) {0}, 4);
    MvArgs a = {w->K, w->R, T, 1, w->K, w->R, 0, add};
    if (mv_dispatch(w, a, T, 1, xb, yb, sel)) return -1;
    memcpy(y, yb.contents, yn);
    return 0;
}
int kt_mv_sel(const KtWeight* w, int P, int xdiv, const int32_t* sel, const float* x, float* y) {
    const size_t yn = 4 * (size_t) P * w->R;
    id<MTLBuffer> xb = buf(x, 4 * (size_t) ((P + xdiv - 1) / xdiv) * w->K), yb = buf(NULL, yn), sb = buf(sel, 4 * (size_t) P);
    MvArgs a = {w->K, w->R, P, xdiv, w->K, w->R, 0, 0};
    if (mv_dispatch(w, a, 0, P, xb, yb, sb)) return -1;
    memcpy(y, yb.contents, yn);
    return 0;
}
int kt_mv_gu(const KtWeight* w, int P, int xdiv, const int32_t* sel, const float* x, float* out) {
    const int fmt = w->fmt, R = w->R, K = w->K;
    const size_t yn = 4 * (size_t) P * R;
    id<MTLBuffer> xb = buf(x, 4 * (size_t) ((P + xdiv - 1) / xdiv) * K), yb = buf(NULL, yn), sb = buf(sel, 4 * (size_t) P);
    MvArgs a = {K, R, P, xdiv, K, R, 0, 0};
    // up = the same buffers one slice on
    const size_t wsl = fmt == MF_BF16 ? 2 * (size_t) R * K : fmt == MF_Q8 ? (size_t) R * K : fmt == MF_Q4 ? (size_t) R * K / 2 : 2 * (size_t) R * K / 8;
    const size_t ssl = fmt == MF_SEED4 || fmt == MF_SEED4P4 ? 2 * (size_t) R * K / 8 : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (size_t) R * K / 64 : 0;
    const size_t bsl = fmt == MF_SEED4 || fmt == MF_SEED4P4 ? 4 : (fmt == MF_Q8 || fmt == MF_Q4) ? 2 * (size_t) R * K / 64 : 0;
    const size_t nsl = fmt == MF_SEED4P4 ? (size_t) R * K / 16 : 0;   // one slice of exponent nibbles
    const MW m = mw(w);
    id<MTLComputePipelineState> p = pipe_("k_mv_gu", fmt, 0);
    if (!p) return -1;
    if (run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p];
            [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:m.W offset:0 atIndex:1]; [e setBuffer:m.S offset:0 atIndex:2]; [e setBuffer:m.B offset:0 atIndex:3];
            [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:sb offset:0 atIndex:6];
            [e setBuffer:m.G offset:0 atIndex:7];
            [e setBuffer:m.W offset:wsl atIndex:8]; [e setBuffer:m.S offset:ssl atIndex:9]; [e setBuffer:m.B offset:bsl atIndex:10];
            [e setBuffer:m.N offset:0 atIndex:11]; [e setBuffer:m.N offset:nsl atIndex:12];
            [e dispatchThreadgroups:MTLSizeMake((R + MV_RPT(fmt) - 1) / MV_RPT(fmt), P, 1) threadsPerThreadgroup:MTLSizeMake(32 * MV_ROWS, 1, 1)];
        }))
        return -1;
    memcpy(out, yb.contents, yn);
    return 0;
}
static int mm_dispatch(const KtWeight* w, MmArgs a, int dense, int gy, id<MTLBuffer> xb, id<MTLBuffer> yb, id<MTLBuffer> pb,
                       id<MTLBuffer> tb) {
    const MW m = mw(w);
    id<MTLComputePipelineState> p = pipe_("k_mm", w->fmt, dense);
    if (!p) return -1;
    return run(^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:p]; [e setBytes:&a length:sizeof a atIndex:0];
        [e setBuffer:m.W offset:0 atIndex:1]; [e setBuffer:m.S offset:0 atIndex:2]; [e setBuffer:m.B offset:0 atIndex:3];
        [e setBuffer:xb offset:0 atIndex:4]; [e setBuffer:yb offset:0 atIndex:5]; [e setBuffer:pb offset:0 atIndex:6];
        [e setBuffer:m.G offset:0 atIndex:7]; [e setBuffer:tb offset:0 atIndex:8]; [e setBuffer:m.N offset:0 atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake((w->R + MM_BM - 1) / MM_BM, gy, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    });
}
int kt_mm(const KtWeight* w, int T, const float* x, float* y) {
    const size_t yn = 4 * (size_t) T * w->R;
    id<MTLBuffer> xb = buf(x, 4 * (size_t) T * w->K), yb = buf(y, yn);
    MmArgs a = {w->K, w->R, T, w->K, w->R, 1, 1, 0};
    if (mm_dispatch(w, a, 1, (T + MM_BN - 1) / MM_BN, xb, yb, buf(NULL, 16), buf(NULL, 16))) return -1;
    memcpy(y, yb.contents, yn);
    return 0;
}
int kt_mm_grouped(const KtWeight* w, int P, int xdiv, const int32_t* perm, const MmTile* tiles, int nt, const float* x,
                  float* y) {
    const size_t yn = 4 * (size_t) P * w->R;
    id<MTLBuffer> xb = buf(x, 4 * (size_t) ((P + xdiv - 1) / xdiv) * w->K), yb = buf(NULL, yn);
    MmArgs a = {w->K, w->R, 0, w->K, w->R, xdiv, 0, 0};
    if (mm_dispatch(w, a, 0, nt, xb, yb, buf(perm, 4 * (size_t) P), buf(tiles, sizeof(MmTile) * (size_t) nt))) return -1;
    memcpy(y, yb.contents, yn);
    return 0;
}

int kt_gnorm(int d, float eps, const float* x, const uint16_t* w, float* y, int T) {
    const size_t n = 4 * (size_t) T * d;
    id<MTLBuffer> xb = buf(x, n), wb = buf(w, 2 * (size_t) d), yb = buf(NULL, n);
    id<MTLComputePipelineState> p = pipe_("k_gnorm", 0, 0);
    const int32_t dd = d;
    if (!p || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p]; [e setBytes:&dd length:4 atIndex:0]; [e setBytes:&eps length:4 atIndex:1];
            [e setBuffer:xb offset:0 atIndex:2]; [e setBuffer:wb offset:0 atIndex:3]; [e setBuffer:yb offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }))
        return -1;
    memcpy(y, yb.contents, n);
    return 0;
}
static void topk_encode(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pt, const RouterArgs* a, id<MTLBuffer> scb,
                        id<MTLBuffer> sb, id<MTLBuffer> ib, id<MTLBuffer> tb, int T) {
    const int32_t TT = T;
    [e setComputePipelineState:pt]; [e setBytes:a length:sizeof *a atIndex:0];
    [e setBuffer:scb offset:0 atIndex:1]; [e setBuffer:sb offset:0 atIndex:2]; [e setBuffer:ib offset:0 atIndex:3];
    [e setBuffer:tb offset:0 atIndex:4]; [e setBytes:&TT length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
int kt_router(RouterArgs a, const uint16_t* W, const uint16_t* bias, const float* x, int T, int32_t* ind, float* wt) {
    id<MTLBuffer> wb = buf(W, 2 * (size_t) a.n * a.d), bb = buf(bias, 2 * (size_t) a.n), xb = buf(x, 4 * (size_t) T * a.d),
                  ib = buf(NULL, 4 * (size_t) T * a.top_k), tb = buf(NULL, 4 * (size_t) T * a.top_k),
                  sb = buf(NULL, 4 * (size_t) T * a.n), scb = buf(NULL, 4 * (size_t) T * a.n);
    id<MTLComputePipelineState> pl = pipe_("k_router_logits", 0, 0), pt = pipe_("k_router_topk", 0, 0);
    if (!pl || !pt || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pl]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:wb offset:0 atIndex:1]; [e setBuffer:bb offset:0 atIndex:2]; [e setBuffer:xb offset:0 atIndex:3];
            [e setBuffer:scb offset:0 atIndex:4]; [e setBuffer:sb offset:0 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((a.n + 7) / 8, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            topk_encode(e, pt, &a, scb, sb, ib, tb, T);
        }))
        return -1;
    memcpy(ind, ib.contents, 4 * (size_t) T * a.top_k);
    memcpy(wt, tb.contents, 4 * (size_t) T * a.top_k);
    return 0;
}
int kt_router_topk(RouterArgs a, const float* score, const float* sel, int T, int32_t* ind, float* wt) {
    id<MTLBuffer> scb = buf(score, 4 * (size_t) T * a.n), sb = buf(sel, 4 * (size_t) T * a.n),
                  ib = buf(NULL, 4 * (size_t) T * a.top_k), tb = buf(NULL, 4 * (size_t) T * a.top_k);
    id<MTLComputePipelineState> pt = pipe_("k_router_topk", 0, 0);
    if (!pt || run(^(id<MTLComputeCommandEncoder> e) { topk_encode(e, pt, &a, scb, sb, ib, tb, T); })) return -1;
    memcpy(ind, ib.contents, 4 * (size_t) T * a.top_k);
    memcpy(wt, tb.contents, 4 * (size_t) T * a.top_k);
    return 0;
}
int kt_swiglu(const float* g, const float* u, float* a, int n) {
    id<MTLBuffer> gb = buf(g, 4 * (size_t) n), ub = buf(u, 4 * (size_t) n), ab = buf(NULL, 4 * (size_t) n);
    id<MTLComputePipelineState> p = pipe_("k_swiglu", 0, 0);
    const int32_t nn = n;
    if (!p || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p]; [e setBuffer:gb offset:0 atIndex:0]; [e setBuffer:ub offset:0 atIndex:1];
            [e setBuffer:ab offset:0 atIndex:2]; [e setBytes:&nn length:4 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((n + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }))
        return -1;
    memcpy(a, ab.contents, 4 * (size_t) n);
    return 0;
}
int kt_combine(const float* D, const float* w, const float* shared, float* x, float* v, int d, int k, int T) {
    id<MTLBuffer> Db = buf(D, 4 * (size_t) T * k * d), wb = buf(w, 4 * (size_t) T * k), shb = buf(shared, 4 * (size_t) T * d),
                  xb = buf(x, 4 * (size_t) T * d), vb = buf(NULL, 4 * (size_t) T * d);
    const int32_t dk[2] = {d, k};
    id<MTLComputePipelineState> pc = pipe_("k_moe_combine", 0, 0), pv = pipe_("k_vcombine", 0, 0);
    if (!pc || !pv || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pc]; [e setBuffer:Db offset:0 atIndex:0]; [e setBuffer:wb offset:0 atIndex:1];
            [e setBuffer:shb offset:0 atIndex:2]; [e setBuffer:xb offset:0 atIndex:3]; [e setBytes:dk length:8 atIndex:4];
            [e dispatchThreads:MTLSizeMake(d, T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e setComputePipelineState:pv]; [e setBuffer:Db offset:0 atIndex:0]; [e setBuffer:wb offset:0 atIndex:1];
            [e setBuffer:vb offset:0 atIndex:2]; [e setBytes:dk length:8 atIndex:3];
            [e dispatchThreads:MTLSizeMake(d, T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }))
        return -1;
    memcpy(x, xb.contents, 4 * (size_t) T * d);
    memcpy(v, vb.contents, 4 * (size_t) T * d);
    return 0;
}
static void attn_encode(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pa, id<MTLComputePipelineState> pd,
                        const AttnArgs* a, id<MTLBuffer> qb, id<MTLBuffer> Kb, id<MTLBuffer> Vb, id<MTLBuffer> rb,
                        id<MTLBuffer> pb, id<MTLBuffer> gb, id<MTLBuffer> ob, int T) {
    [e setComputePipelineState:pa]; [e setBytes:a length:sizeof *a atIndex:0];
    [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:Kb offset:0 atIndex:2]; [e setBuffer:Vb offset:0 atIndex:3];
    [e setBuffer:rb offset:0 atIndex:4]; [e setBuffer:pb offset:0 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(a->n_splits, a->n_kv, T) threadsPerThreadgroup:MTLSizeMake(32 * ATT_SG, 1, 1)];
    [e setComputePipelineState:pd]; [e setBytes:a length:sizeof *a atIndex:0]; [e setBuffer:pb offset:0 atIndex:1];
    [e setBuffer:gb offset:0 atIndex:2]; [e setBuffer:ob offset:0 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(a->n_head, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}
int kt_rope_attn(AttnArgs a, float* q, const float* k, const float* v, uint16_t* Kc, uint16_t* Vc, int npos,
                 const RowInfo* ri, const float* inv, const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, kn = 4 * (size_t) T * a.n_kv * 128, cn = 2 * (size_t) npos * a.n_kv * 128;
    id<MTLBuffer> qb = buf(q, qn), kb = buf(k, kn), vb = buf(v, kn), Kb = buf(Kc, cn), Vb = buf(Vc, cn),
                  rb = buf(ri, sizeof(RowInfo) * (size_t) T), ib = buf(inv, 4 * 64), gb = buf(g, qn), ob = buf(NULL, qn),
                  pb = buf(NULL, 4 * (size_t) T * a.n_head * a.n_splits * 130);
    const int32_t hk[2] = {a.n_head, a.n_kv};
    id<MTLComputePipelineState> pr = pipe_("k_rope_kv", 0, 0), pa = pipe_("k_attn", 0, 0), pd = pipe_("k_attn_reduce", 0, 0);
    if (!pr || !pa || !pd || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pr];
            [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kb offset:0 atIndex:1]; [e setBuffer:vb offset:0 atIndex:2];
            [e setBuffer:Kb offset:0 atIndex:3]; [e setBuffer:Vb offset:0 atIndex:4]; [e setBuffer:rb offset:0 atIndex:5];
            [e setBuffer:ib offset:0 atIndex:6]; [e setBytes:hk length:8 atIndex:7];
            [e dispatchThreads:MTLSizeMake(a.n_head * 64, T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            attn_encode(e, pa, pd, &a, qb, Kb, Vb, rb, pb, gb, ob, T);
        }))
        return -1;
    memcpy(q, qb.contents, qn);
    memcpy(Kc, Kb.contents, cn);
    memcpy(Vc, Vb.contents, cn);
    memcpy(o, ob.contents, qn);
    return 0;
}
int kt_attn(AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
            const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, cn = 2 * (size_t) npos * a.n_kv * 128;
    id<MTLBuffer> qb = buf(q, qn), Kb = buf(Kc, cn), Vb = buf(Vc, cn), rb = buf(ri, sizeof(RowInfo) * (size_t) T),
                  gb = buf(g, qn), ob = buf(NULL, qn), pb = buf(NULL, 4 * (size_t) T * a.n_head * a.n_splits * 130);
    id<MTLComputePipelineState> pa = pipe_("k_attn", 0, 0), pd = pipe_("k_attn_reduce", 0, 0);
    if (!pa || !pd || run(^(id<MTLComputeCommandEncoder> e) { attn_encode(e, pa, pd, &a, qb, Kb, Vb, rb, pb, gb, ob, T); }))
        return -1;
    memcpy(o, ob.contents, qn);
    return 0;
}
int kt_attn_prefill(AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
                    const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, cn = 2 * (size_t) npos * a.n_kv * 128;
    id<MTLBuffer> qb = buf(q, qn), Kb = buf(Kc, cn), Vb = buf(Vc, cn), rb = buf(ri, sizeof(RowInfo) * (size_t) T),
                  gb = buf(g, qn), ob = buf(NULL, qn);
    const int32_t TT = T;
    id<MTLComputePipelineState> pf = pipe_("k_attn_prefill", 0, 0);
    if (!pf || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pf]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:Kb offset:0 atIndex:2]; [e setBuffer:Vb offset:0 atIndex:3];
            [e setBuffer:rb offset:0 atIndex:4]; [e setBuffer:gb offset:0 atIndex:5]; [e setBuffer:ob offset:0 atIndex:6];
            [e setBytes:&TT length:4 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake((T + 8 * ATTF_RS - 1) / (8 * ATTF_RS), a.n_kv, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * ATTF_G * ATTF_RS, 1, 1)];
        }))
        return -1;
    memcpy(o, ob.contents, qn);
    return 0;
}
int kt_embed(int fmt, const uint16_t* E, const uint32_t* q8, const uint16_t* s8, const uint16_t* b8, int V, int d,
             const int32_t* ids, int n, float* x) {
    id<MTLBuffer> Eb = buf(E, 2 * (size_t) V * d), Qb = buf(q8, (size_t) V * d), Sb = buf(s8, 2 * (size_t) V * d / 64),
                  Bb = buf(b8, 2 * (size_t) V * d / 64), ib = buf(ids, 4 * (size_t) n), xb = buf(NULL, 4 * (size_t) n * d);
    const int32_t d2 = d;
    id<MTLComputePipelineState> p = pipe_("k_embed", fmt, 0);
    if (!p || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p]; [e setBytes:&d2 length:4 atIndex:0]; [e setBuffer:Eb offset:0 atIndex:1];
            [e setBuffer:Qb offset:0 atIndex:2]; [e setBuffer:Sb offset:0 atIndex:3]; [e setBuffer:Bb offset:0 atIndex:4];
            [e setBuffer:ib offset:0 atIndex:5]; [e setBuffer:xb offset:0 atIndex:6];
            [e dispatchThreads:MTLSizeMake(d, n, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }))
        return -1;
    memcpy(x, xb.contents, 4 * (size_t) n * d);
    return 0;
}
int kt_argmax(const float* logits, int V, int n, int32_t* out) {
    id<MTLBuffer> lb = buf(logits, 4 * (size_t) V * n), ob = buf(NULL, 4 * (size_t) n);
    const int32_t vv = V;
    id<MTLComputePipelineState> p = pipe_("k_argmax", 0, 0);
    if (!p || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:p]; [e setBuffer:lb offset:0 atIndex:0]; [e setBuffer:ob offset:0 atIndex:1];
            [e setBytes:&vv length:4 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }))
        return -1;
    memcpy(out, ob.contents, 4 * (size_t) n);
    return 0;
}
