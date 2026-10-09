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
        [e setBuffer:yb offset:0 atIndex:10];
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
    const int32_t dk2[2] = {d, k}, *dk = dk2;   // blocks cannot capture arrays
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
    const int32_t hk2[2] = {a.n_head, a.n_kv}, *hk = hk2;   // blocks cannot capture arrays
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
// The 8-bit caches: values at buffers (K, V), scales at (K + 6, V + 6) of the rope kernel, (6, 7) of the decode and
// (8, 9) of the prefill kernel.
int kt_rope_kv_q8(AttnArgs a, float* q, const float* k, const float* v, KtKvQ8 kv, int npos, const RowInfo* ri,
                  const float* inv, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, kn = 4 * (size_t) T * a.n_kv * 128;
    const size_t cn = (size_t) npos * a.n_kv * 128, sn = 4 * (size_t) npos * a.n_kv;
    id<MTLBuffer> qb = buf(q, qn), kb = buf(k, kn), vb = buf(v, kn), Kb = buf(kv.k, cn), Vb = buf(kv.v, cn), Ksb = buf(kv.ks, sn),
                  Vsb = buf(kv.vs, sn), rb = buf(ri, sizeof(RowInfo) * (size_t) T), ib = buf(inv, 4 * 64);
    const int32_t hk2[2] = {a.n_head, a.n_kv}, *hk = hk2;   // blocks cannot capture arrays
    id<MTLComputePipelineState> pr = pipe_("k_rope_kv_q8", 0, 0);
    if (!pr || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pr];
            [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kb offset:0 atIndex:1]; [e setBuffer:vb offset:0 atIndex:2];
            [e setBuffer:Kb offset:0 atIndex:3]; [e setBuffer:Vb offset:0 atIndex:4]; [e setBuffer:rb offset:0 atIndex:5];
            [e setBuffer:ib offset:0 atIndex:6]; [e setBytes:hk length:8 atIndex:7];
            [e setBuffer:Ksb offset:0 atIndex:8]; [e setBuffer:Vsb offset:0 atIndex:9];
            [e dispatchThreads:MTLSizeMake(a.n_head * 64, T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }))
        return -1;
    memcpy(q, qb.contents, qn);
    memcpy(kv.k, Kb.contents, cn);
    memcpy(kv.v, Vb.contents, cn);
    memcpy(kv.ks, Ksb.contents, sn);
    memcpy(kv.vs, Vsb.contents, sn);
    return 0;
}
int kt_attn_q8(AttnArgs a, const float* q, KtKvQ8 kv, int npos, const RowInfo* ri, const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, cn = (size_t) npos * a.n_kv * 128, sn = 4 * (size_t) npos * a.n_kv;
    id<MTLBuffer> qb = buf(q, qn), Kb = buf(kv.k, cn), Vb = buf(kv.v, cn), Ksb = buf(kv.ks, sn), Vsb = buf(kv.vs, sn),
                  rb = buf(ri, sizeof(RowInfo) * (size_t) T), gb = buf(g, qn), ob = buf(NULL, qn),
                  pb = buf(NULL, 4 * (size_t) T * a.n_head * a.n_splits * 130);
    id<MTLComputePipelineState> pa = pipe_("k_attn_q8", 0, 0), pd = pipe_("k_attn_reduce", 0, 0);
    if (!pa || !pd || run(^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:Ksb offset:0 atIndex:6]; [e setBuffer:Vsb offset:0 atIndex:7];   // kept across the pipeline switch
            attn_encode(e, pa, pd, &a, qb, Kb, Vb, rb, pb, gb, ob, T);
        }))
        return -1;
    memcpy(o, ob.contents, qn);
    return 0;
}
int kt_attn_prefill_q8(AttnArgs a, const float* q, KtKvQ8 kv, int npos, const RowInfo* ri, const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, cn = (size_t) npos * a.n_kv * 128, sn = 4 * (size_t) npos * a.n_kv;
    id<MTLBuffer> qb = buf(q, qn), Kb = buf(kv.k, cn), Vb = buf(kv.v, cn), Ksb = buf(kv.ks, sn), Vsb = buf(kv.vs, sn),
                  rb = buf(ri, sizeof(RowInfo) * (size_t) T), gb = buf(g, qn), ob = buf(NULL, qn);
    const int32_t TT = T;
    id<MTLComputePipelineState> pf = pipe_("k_attn_prefill_q8", 0, 0);
    if (!pf || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pf]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:Kb offset:0 atIndex:2]; [e setBuffer:Vb offset:0 atIndex:3];
            [e setBuffer:rb offset:0 atIndex:4]; [e setBuffer:gb offset:0 atIndex:5]; [e setBuffer:ob offset:0 atIndex:6];
            [e setBytes:&TT length:4 atIndex:7]; [e setBuffer:Ksb offset:0 atIndex:8]; [e setBuffer:Vsb offset:0 atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake((T + 8 * ATTF_RS - 1) / (8 * ATTF_RS), a.n_kv, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * ATTF_G * ATTF_RS, 1, 1)];
        }))
        return -1;
    memcpy(o, ob.contents, qn);
    return 0;
}
// ---- MLA -------------------------------------------------------------------------------------------------------------
int kt_heads_mv(int H, int O, int I, const uint16_t* W, const float* x, int xs, int hs, const float* g, float* y, int T) {
    const size_t xn = 4 * ((size_t) (T - 1) * xs + (size_t) (H - 1) * hs + I), yn = 4 * (size_t) T * H * O;
    id<MTLBuffer> wb = buf(W, 2 * (size_t) H * O * I), xb = buf(x, xn), gb = buf(g, g ? yn : 16), yb = buf(NULL, yn);
    const HmvArgs a = {H, O, I, xs, hs, g != NULL, {0}};
    if (T > MV_MAXT) {   // prompt rows, as the engine runs them: a GEMM per head (k_mm), with the gate
        const MmArgs m = {I, O, T, xs, H * O, 1, g ? 2 : 0, 0};
        id<MTLComputePipelineState> pm = pipe_("k_mm", MF_BF16, 1);
        if (!pm || run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:pm]; [e setBytes:&m length:sizeof m atIndex:0];
                for (int i = 2; i <= 9; ++i) if (i != 4 && i != 5) [e setBuffer:wb offset:0 atIndex:(NSUInteger) i];
                for (int h = 0; h < H; ++h) {
                    [e setBuffer:wb offset:(NSUInteger) h * O * I * 2 atIndex:1];
                    [e setBuffer:xb offset:(NSUInteger) h * hs * 4 atIndex:4];
                    [e setBuffer:yb offset:(NSUInteger) h * O * 4 atIndex:5];
                    [e setBuffer:(g ? gb : yb) offset:(NSUInteger) h * O * 4 atIndex:10];
                    [e dispatchThreadgroups:MTLSizeMake((O + MM_BM - 1) / MM_BM, (T + MM_BN - 1) / MM_BN, 1)
                      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }))
            return -1;
        memcpy(y, yb.contents, yn);
        return 0;
    }
    id<MTLComputePipelineState> pp = pipe_("k_heads_mv", 0, 0);
    if (!pp || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pp]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:wb offset:0 atIndex:1]; [e setBuffer:xb offset:0 atIndex:2]; [e setBuffer:gb offset:0 atIndex:3];
            [e setBuffer:yb offset:0 atIndex:4];
            [e dispatchThreads:MTLSizeMake((size_t) O * 32, H, T) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }))
        return -1;
    memcpy(y, yb.contents, yn);
    return 0;
}
// MLA cache bytes a position for n values in fmt (ENG_KV_*: 0 BF16, 2 FP8, 3 FP4): codes, block scales
// (key: the RoPE key, all FP8 when quantized; else the latent)
static int kv_lead(int fmt, int n, int key) { return fmt == 3 && !key ? (n < MLA_FP4_LEAD ? n : MLA_FP4_LEAD) : n; }
static size_t kv_rowb(int fmt, int n, int key) {
    return fmt >= 2 ? (size_t) (kv_lead(fmt, n, key) + (n - kv_lead(fmt, n, key)) / 2) : 2 * (size_t) n;
}
static size_t kv_srowb(int fmt, int n, int key) {
    return fmt >= 2 ? (size_t) (kv_lead(fmt, n, key) / 32 + (n - kv_lead(fmt, n, key)) / 16) : 0;
}
static int mla_attn_x(int fmt, int n_head, float scale, const int* kb, int nb, const float* qn, const float* qr, const float* kn,
                      const float* vn, const void* Kc, const uint8_t* Ks, int npos, const RowInfo* ri, const float* g, float* o, int T) {
    const int ld = n_head * 128, Tp = (T + 8 * MLPF_R - 1) / (8 * MLPF_R) * (8 * MLPF_R), nk = kb[nb];
    const size_t qn_n = 4 * (size_t) T * ld, pad_n = 4 * (size_t) Tp * ld, kv_n = 4 * (size_t) nk * ld;
    id<MTLBuffer> qb = buf(NULL, pad_n), rb_ = buf(NULL, pad_n), kb_ = buf(NULL, kv_n + 4 * (size_t) MLPF_BK * ld),
                  vb = buf(NULL, kv_n + 4 * (size_t) MLPF_BK * ld), Kb = buf(Kc, kv_rowb(fmt, 128, 1) * (size_t) npos),
                  Sb = buf(Ks, kv_srowb(fmt, 128, 1) * (size_t) npos),
                  rib = buf(ri, sizeof(RowInfo) * (size_t) T), mb = buf(NULL, 4 * (size_t) Tp * n_head),
                  lb = buf(NULL, 4 * (size_t) Tp * n_head), ob = buf(NULL, pad_n), gb = buf(g, qn_n), out = buf(NULL, qn_n);
    memcpy(qb.contents, qn, qn_n);
    memcpy(rb_.contents, qr, qn_n);
    memcpy(kb_.contents, kn, kv_n);
    memcpy(vb.contents, vn, kv_n);
    id<MTLComputePipelineState> pp = pipe_("k_mla_prefill", fmt, 0);
    if (!pp || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pp];
            for (int i = 0; i < nb; ++i) {
                const MlpArgs a = {n_head, kb[i], kb[i + 1], i == 0, i == nb - 1, scale, T, 0};
                [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:rb_ offset:0 atIndex:2];
                [e setBuffer:kb_ offset:(NSUInteger) kb[i] * ld * 4 atIndex:3]; [e setBuffer:vb offset:(NSUInteger) kb[i] * ld * 4 atIndex:4];
                [e setBuffer:Kb offset:0 atIndex:5]; [e setBuffer:rib offset:0 atIndex:6]; [e setBuffer:mb offset:0 atIndex:7];
                [e setBuffer:lb offset:0 atIndex:8]; [e setBuffer:ob offset:0 atIndex:9]; [e setBuffer:gb offset:0 atIndex:10];
                [e setBuffer:out offset:0 atIndex:11]; [e setBuffer:Sb offset:0 atIndex:12];
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger) Tp / (8 * MLPF_R), (NSUInteger) n_head, 1)
                  threadsPerThreadgroup:MTLSizeMake(32 * MLPF_R, 1, 1)];
            }
        }))
        return -1;
    memcpy(o, out.contents, qn_n);
    return 0;
}
int kt_mla_attn_x(int n_head, float scale, const int* kb, int nb, const float* qn, const float* qr, const float* kn,
                  const float* vn, const uint16_t* Kc, int npos, const RowInfo* ri, const float* g, float* o, int T) {
    return mla_attn_x(0, n_head, scale, kb, nb, qn, qr, kn, vn, Kc, NULL, npos, ri, g, o, T);
}
int kt_mla_attn_xq(int fmt, int n_head, float scale, const int* kb, int nb, const float* qn, const float* qr, const float* kn,
                   const float* vn, const uint8_t* Kq, const uint8_t* Ks, int npos, const RowInfo* ri, const float* g, float* o, int T) {
    return mla_attn_x(fmt, n_head, scale, kb, nb, qn, qr, kn, vn, Kq, Ks, npos, ri, g, o, T);
}
static int mla_rope(MlaArgs a, int fmt, float* qr, const float* kr, const float* c, void* Kc, void* Vc, uint8_t* Ks, uint8_t* Vs,
                    int npos, const RowInfo* ri, const float* inv, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, kn = kv_rowb(fmt, 128, 1) * npos, vn = kv_rowb(fmt, a.r, 0) * npos;
    const size_t ksn = kv_srowb(fmt, 128, 1) * npos, vsn = kv_srowb(fmt, a.r, 0) * npos;
    id<MTLBuffer> qb = buf(qr, qn), kb = buf(kr, 4 * (size_t) T * 128), cb = buf(c, 4 * (size_t) T * a.r),
                  Kb = buf(Kc, kn), Vb = buf(Vc, vn), Ksb = buf(Ks, ksn), Vsb = buf(Vs, vsn),
                  rb = buf(ri, sizeof(RowInfo) * (size_t) T), ib = buf(inv, 4 * 64);
    id<MTLComputePipelineState> pp = pipe_("k_mla_rope", fmt, 0);
    const int nx = (a.n_head + 1) * 64 > a.r ? (a.n_head + 1) * 64 : a.r;
    if (!pp || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pp]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:qb offset:0 atIndex:1]; [e setBuffer:kb offset:0 atIndex:2]; [e setBuffer:cb offset:0 atIndex:3];
            [e setBuffer:Kb offset:0 atIndex:4]; [e setBuffer:Vb offset:0 atIndex:5]; [e setBuffer:rb offset:0 atIndex:6];
            [e setBuffer:ib offset:0 atIndex:7]; [e setBuffer:Ksb offset:0 atIndex:8]; [e setBuffer:Vsb offset:0 atIndex:9];
            [e dispatchThreads:MTLSizeMake(nx, T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }))
        return -1;
    memcpy(qr, qb.contents, qn);
    memcpy(Kc, Kb.contents, kn);
    memcpy(Vc, Vb.contents, vn);
    if (Ks) { memcpy(Ks, Ksb.contents, ksn); memcpy(Vs, Vsb.contents, vsn); }
    return 0;
}
int kt_mla_rope(MlaArgs a, float* qr, const float* kr, const float* c, uint16_t* Kc, uint16_t* Vc, int npos,
                const RowInfo* ri, const float* inv, int T) {
    return mla_rope(a, 0, qr, kr, c, Kc, Vc, NULL, NULL, npos, ri, inv, T);
}
int kt_mla_rope_q(MlaArgs a, int fmt, float* qr, const float* kr, const float* c, KtKvMla kv, int npos, const RowInfo* ri,
                  const float* inv, int T) {
    return mla_rope(a, fmt, qr, kr, c, kv.k, kv.v, kv.ks, kv.vs, npos, ri, inv, T);
}
static int mla_attn(MlaArgs a, int fmt, const float* ql, const float* qr, const void* Kc, const void* Vc, const uint8_t* Ks,
                    const uint8_t* Vs, int npos, const RowInfo* ri, float* olat, int T) {
    const size_t on = 4 * (size_t) T * a.n_head * a.r;
    id<MTLBuffer> lb = buf(ql, on), qb = buf(qr, 4 * (size_t) T * a.n_head * 128), Kb = buf(Kc, kv_rowb(fmt, 128, 1) * npos),
                  Vb = buf(Vc, kv_rowb(fmt, a.r, 0) * npos), Ksb = buf(Ks, kv_srowb(fmt, 128, 1) * npos),
                  Vsb = buf(Vs, kv_srowb(fmt, a.r, 0) * npos), rb = buf(ri, sizeof(RowInfo) * (size_t) T), ob = buf(NULL, on),
                  pb = buf(NULL, 4 * (size_t) T * a.n_head * a.n_splits * (a.r + 2));
    id<MTLComputePipelineState> pa = pipe_("k_mla_attn", fmt, (a.r + MLAF_DC - 1) / MLAF_DC), pd = pipe_("k_mla_reduce", 0, 0);
    if (!pa || !pd) return -1;
    const int ng = (a.n_head + MLAF_Q - 1) / MLAF_Q;
    if (run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pa]; [e setBytes:&a length:sizeof a atIndex:0];
            [e setBuffer:lb offset:0 atIndex:1]; [e setBuffer:qb offset:0 atIndex:2]; [e setBuffer:Kb offset:0 atIndex:3];
            [e setBuffer:Vb offset:0 atIndex:4]; [e setBuffer:rb offset:0 atIndex:5];
            [e setBuffer:(a.n_splits > 1 ? pb : ob) offset:0 atIndex:6];
            [e setBuffer:Ksb offset:0 atIndex:7]; [e setBuffer:Vsb offset:0 atIndex:8];
            [e dispatchThreadgroups:MTLSizeMake(a.n_splits, ng, T) threadsPerThreadgroup:MTLSizeMake(32 * MLAF_SG, 1, 1)];
            if (a.n_splits > 1) {
                [e setComputePipelineState:pd]; [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:pb offset:0 atIndex:1]; [e setBuffer:ob offset:0 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(a.n_head, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }))
        return -1;
    memcpy(olat, ob.contents, on);
    return 0;
}
int kt_mla_attn(MlaArgs a, const float* ql, const float* qr, const uint16_t* Kc, const uint16_t* Vc, int npos,
                const RowInfo* ri, float* olat, int T) {
    return mla_attn(a, 0, ql, qr, Kc, Vc, NULL, NULL, npos, ri, olat, T);
}
int kt_mla_attn_q(MlaArgs a, int fmt, const float* ql, const float* qr, KtKvMla kv, int npos, const RowInfo* ri, float* olat, int T) {
    return mla_attn(a, fmt, ql, qr, kv.k, kv.v, kv.ks, kv.vs, npos, ri, olat, T);
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
int kt_heads_q(int fmt, int tr, int H, int O, int I, const void* codes, const uint16_t* scales, const uint16_t* biases,
               const float* x, int xs, int hs, const float* g, float* y, int T) {
    const size_t n = (size_t) O * I, cb = fmt == MF_BF16 ? 2 * n : fmt == MF_Q8 ? n : n / 2, sb = n / 64 * 2;
    const size_t xn = 4 * ((size_t) (T - 1) * xs + (size_t) (H - 1) * hs + I), yn = 4 * (size_t) T * H * O;
    id<MTLBuffer> wb = buf(codes, cb * H), sbuf = buf(scales, fmt == MF_BF16 ? 0 : sb * H), bbuf = buf(biases, fmt == MF_BF16 ? 0 : sb * H),
                  xb = buf(x, xn), gb = buf(g, g ? yn : 16), yb = buf(NULL, yn);
    if (T <= MV_MAXT && !tr) {   // the decode matvec
        const HmvArgs a = {H, O, I, xs, hs, g != NULL, {0}};
        id<MTLComputePipelineState> pp = pipe_("k_heads_mv", fmt, 0);
        if (!pp || run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:pp]; [e setBytes:&a length:sizeof a atIndex:0];
                [e setBuffer:wb offset:0 atIndex:1]; [e setBuffer:xb offset:0 atIndex:2]; [e setBuffer:gb offset:0 atIndex:3];
                [e setBuffer:yb offset:0 atIndex:4]; [e setBuffer:sbuf offset:0 atIndex:5]; [e setBuffer:bbuf offset:0 atIndex:6];
                [e dispatchThreads:MTLSizeMake((size_t) O * 32, H, T) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }))
            return -1;
    } else {   // prompt rows: a GEMM per head (tr: W_h read transposed)
        const MmArgs m = {I, O, T, xs, H * O, 1, g ? 2 : 0, 0};
        id<MTLComputePipelineState> pm = pipe_("k_mm", fmt, tr ? 2 : 1);
        if (!pm || run(^(id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:pm]; [e setBytes:&m length:sizeof m atIndex:0];
                for (int i = 6; i <= 9; ++i) [e setBuffer:wb offset:0 atIndex:(NSUInteger) i];
                for (int h = 0; h < H; ++h) {
                    [e setBuffer:wb offset:(NSUInteger) (h * cb) atIndex:1];
                    [e setBuffer:(fmt == MF_BF16 ? wb : sbuf) offset:(NSUInteger) (fmt == MF_BF16 ? 0 : h * sb) atIndex:2];
                    [e setBuffer:(fmt == MF_BF16 ? wb : bbuf) offset:(NSUInteger) (fmt == MF_BF16 ? 0 : h * sb) atIndex:3];
                    [e setBuffer:xb offset:(NSUInteger) h * hs * 4 atIndex:4];
                    [e setBuffer:yb offset:(NSUInteger) h * O * 4 atIndex:5];
                    [e setBuffer:(g ? gb : yb) offset:(NSUInteger) h * O * 4 atIndex:10];
                    [e dispatchThreadgroups:MTLSizeMake((O + MM_BM - 1) / MM_BM, (T + MM_BN - 1) / MM_BN, 1)
                      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }))
            return -1;
    }
    memcpy(y, yb.contents, yn);
    return 0;
}
int kt_heads_mm_t(int H, int O, int I, const uint16_t* W, const float* x, int xs, int hs, float* y, int T) {
    const size_t xn = 4 * ((size_t) (T - 1) * xs + (size_t) (H - 1) * hs + I), yn = 4 * (size_t) T * H * O;
    id<MTLBuffer> wb = buf(W, 2 * (size_t) H * O * I), xb = buf(x, xn), yb = buf(NULL, yn);
    const MmArgs m = {I, O, T, xs, H * O, 1, 0, 0};
    id<MTLComputePipelineState> pm = pipe_("k_mm", MF_BF16, 2);
    if (!pm || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pm]; [e setBytes:&m length:sizeof m atIndex:0];
            for (int i = 2; i <= 10; ++i) if (i != 4 && i != 5) [e setBuffer:wb offset:0 atIndex:(NSUInteger) i];
            for (int h = 0; h < H; ++h) {
                [e setBuffer:wb offset:(NSUInteger) h * O * I * 2 atIndex:1];
                [e setBuffer:xb offset:(NSUInteger) h * hs * 4 atIndex:4];
                [e setBuffer:yb offset:(NSUInteger) h * O * 4 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((O + MM_BM - 1) / MM_BM, (T + MM_BN - 1) / MM_BN, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }))
        return -1;
    memcpy(y, yb.contents, yn);
    return 0;
}
int kt_kv_f32(int fmt, const uint8_t* codes, const uint8_t* scales, int len, int rows, float* y) {
    const uint32_t n = (uint32_t) len * rows, l = (uint32_t) len;
    id<MTLBuffer> cb = buf(codes, kv_rowb(fmt, len, 0) * rows), sb = buf(scales, kv_srowb(fmt, len, 0) * rows), yb = buf(NULL, 4 * (size_t) n);
    id<MTLComputePipelineState> pp = pipe_("k_kv_f32", fmt, 0);
    if (!pp || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pp]; [e setBuffer:cb offset:0 atIndex:0]; [e setBuffer:yb offset:0 atIndex:1];
            [e setBytes:&n length:4 atIndex:2]; [e setBuffer:sb offset:0 atIndex:3]; [e setBytes:&l length:4 atIndex:4];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }))
        return -1;
    memcpy(y, yb.contents, 4 * (size_t) n);
    return 0;
}
int kt_bf16_f32(const uint16_t* x, float* y, int n) {
    id<MTLBuffer> xb = buf(x, 2 * (size_t) n), yb = buf(NULL, 4 * (size_t) n);
    const uint32_t nn = (uint32_t) n;
    id<MTLComputePipelineState> pp = pipe_("k_bf16_f32", 0, 0);
    if (!pp || run(^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pp]; [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:yb offset:0 atIndex:1];
            [e setBytes:&nn length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake((NSUInteger) n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }))
        return -1;
    memcpy(y, yb.contents, 4 * (size_t) n);
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
int kt_mla_prefill(MlaArgs a, const float* q, const float* qr, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
                   const uint16_t* Wql, const uint16_t* Wvu, const float* g, float* o, int T, int dec_keys, uint16_t* Kn, uint16_t* Vd) {
    (void) a; (void) q; (void) qr; (void) Kc; (void) Vc; (void) npos; (void) ri; (void) Wql; (void) Wvu; (void) g; (void) o; (void) T;
    (void) dec_keys; (void) Kn; (void) Vd;
    return 1;   // the Metal engine computes prompt rows absorbed (k_mla_attn)
}
int kt_mla_decomp_q(MlaArgs a, int qfmt, const void* qc, const uint16_t* qs, const uint16_t* qb, int vfmt, const void* vc,
                    const uint16_t* vs, const uint16_t* vb, const uint16_t* c, int n, uint16_t* Kn, uint16_t* Vd) {
    (void) a; (void) qfmt; (void) qc; (void) qs; (void) qb; (void) vfmt; (void) vc; (void) vs; (void) vb; (void) c; (void) n;
    (void) Kn; (void) Vd;
    return 1;   // (as kt_mla_prefill)
}
