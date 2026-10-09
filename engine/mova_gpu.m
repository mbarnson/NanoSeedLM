// engine/mova_gpu.m - Metal host code of the GPU engine for K2-Horizon MoVA (implements engine_api.h, mova_ext.h).
// Shapes come from the config block (nslm/mova_cfg.h).  Tensors come from the model folder's mmapped shards
// (nslm/model_st.h), zero-copy where page-aligned; tensors the folder lacks are copied from the HF checkpoint as BF16
// (experts stacked [E][rows][cols]), so the original snapshot runs as the BF16 engine.  The forward mirrors the MLX
// reference implementation's BF16 rounding points.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <IOKit/IOKitLib.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "engine_api.h"
#include "kvq.h"
#include "model_st.h"
#include "kernels_moe.metal"   // argument structs and constants only
#include "lfsr.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"
#include "mova_ext.h"

#define MAX_ROWS 512             // rows per forward (prompt chunks); decode and T <= 8 use the matvecs
#define MAXP (MAX_ROWS * 8)      // (row, expert) pairs per forward
#ifndef MAX_SPLITS
#define MAX_SPLITS 32   // decode attention: at most this many key splits
#endif
#define MAX_LOGIT_ROWS 64        // LM head rows per pass (prompt scoring runs it in 64-row passes)
#define KVP_PAGE 65536           // paged KV: sparse page bytes (MTLSparsePageSize64)
#define KVP_HEAP 1024            // pages per heap (64 MB)
#define KVP_GROW 512             // positions a slot's backing grows by
#define MLP_KB 512               // MLA prompt rows: cached keys expanded per pass
#define MAX_TILES (MAXP / MM_BN + 128)

typedef struct {
    id<MTLBuffer> b[4];   // streams as in nslm/model_st.h
    uint64_t o[4];
    int fmt, slices, rows, cols;
} MW;

typedef struct {
    MW ln1, ln2, q, k, o, g, v;          // v: dense layers
    MW vr, vb, vx;                        // value router, bias, experts (sparse layers)
    MW mg, mu, md;                        // dense MLP
    MW r, rb, eg, eu, ed, sg, su, sd;     // MoE router, bias, experts, shared expert
    MW ka_x, ka_v, kr, qm, ql, vu;        // MLA: kv_a_x, kv_a_v (MoVA layers), k_rope_proj, q_rope_mix, q_lat, v_up
    int sparse, mla_r;                    // mla_r: MLA latent rank (0 for GQA)
} Layer;

typedef struct {
    int32_t* hist;
    int len, cap;
    int done;   // positions whose KV is computed (hist[0 .. done-1])
    int backed; // paged KV: positions whose pages are mapped
    int xend;   // MLA: prompt rows at positions below xend attend with the latent expanded per head
} Seq;

struct Eng {
    MovaCfg c;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipes;
    id<MTLResidencySet> residency;
    NSMutableArray<id<MTLBuffer>>* buffers;
    NsModel nm;
    char model_dir[1024];   // the original checkpoint opens lazily: only for tensors the folder does not hold
    MovaCkpt* ck;
    Layer* L;
    MW embed, norm, head;
    id<MTLBuffer> stab;                   // per-seed stream table (SEED4)
    id<MTLBuffer> stab32;                 // the 32-bit stream table (SEED4P4)
    id<MTLBuffer> __strong* Kc;
    id<MTLBuffer> __strong* Vc;
    id<MTLBuffer> __strong* Ks;           // 8-bit KV cache (kv_q8): one f32 scale per (position, KV head); Kc / Vc are int8
    id<MTLBuffer> __strong* Vs;
    int kv_q8;
    int kv_fmt;                           // ENG_KV_*: MLA caches in FP8 / FP4 (nslm/kvq.h) have codes in Kc / Vc, scales in Ks / Vs
    int64_t kv_cap;
    // scratch (MAX_ROWS rows)
    id<MTLBuffer> x, xn, q, k, v, gq, ao, ga, ua, aa, G, U, A, D, V, sh, logits, ids, ri, inv, part, inds, wts, vinds, vwts, am;
    id<MTLBuffer> perm, tiles, vperm, vtiles;   // grouped GEMM: pairs sorted by expert, tile tables (prompt chunks)
    id<MTLBuffer> lat, qrp, qlat, olat;   // MLA: latent c [T][r], query RoPE parts, absorbed queries, latent outputs
    id<MTLBuffer> latf, kn, vn, mst, lst, ost;   // MLA prompt rows: latent rows (f32), expanded keys / values, softmax state
    int prompt;                           // prompt rows: the prefill kernels for any row count (chunking-invariant caches)
    int expand, expand_min;               // MLA prompt rows: the latent expanded per head (Seq.xend)
    int gpu_cores;                        // the GPU's cores (IORegistry gpu-core-count; 0: unknown)
    int batch_gemm;                       // NSLM_BATCH_GEMM=N: decode steps of at least N (> MV_MAXT) slots run as forwards
                                          // of up to 64 rows through the GEMMs (faster at large N; not bit-equal to alone)
    int decode_rows;                      // this forward's rows are other slots' pending tokens: per-row attention
    EngMem mem;
    Seq* seqs;                            // slot s: KV cache rows s * slot_stride .. (+ slot_cap)
    int nseqs;
    int64_t slot_cap;
    int64_t slot_stride;                  // positions between slot bases (paged KV: each slot starts on a page)
    NSMutableArray* kv_bufs;              // the KV buffers in kv_copy's order (per layer K, V, then their scales)
    uint64_t* kv_rb;                      // their bytes per position
    int kvp;                              // paged KV: placement sparse buffers, 64 KB pages mapped from heaps as needed
    id kvp_q;                             // id<MTL4CommandQueue> (mapping updates)
    id<MTLSharedEvent> kvp_ev;
    uint64_t kvp_evv;
    NSMutableArray* kvp_heaps;            // id<MTLHeap>, or NSNull once released
    int* kvp_used;                        // pages in use per heap
    uint64_t (*kvp_free)[KVP_HEAP / 64];  // free-page bitmaps per heap
    int32_t** kvp_map;                    // per KV buffer: page -> heap * KVP_HEAP + heap page, or -1
    char desc[256];
    // route capture
    int route_on, route_max;
    int32_t* route_mlp, *route_val;
    float* route_mlp_sel, *route_val_sel;
    int route_rows;
    id<MTLBuffer> rsel, vsel;             // device copies of the selection scores of the current forward
    id<MTLBuffer> rscore;                 // router sigmoid scores (scratch)
    // timing
    id<MTLCounterSampleBuffer> csb;
    int timing_on;
    double tg_sec[MOVA_TG_N];
};

// ---- Ctx / Cmd ------------------------------------------------------------------------------------------------------

static id<MTLBuffer> alloc_k(Eng* e, uint64_t bytes, const char* label, int64_t* kind) {
    bytes = (bytes + 255) & ~255ull;
    id<MTLBuffer> b = [e->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) return nil;
    b.label = [NSString stringWithUTF8String:label];
    [e->buffers addObject:b];
    *kind += (int64_t) b.length;
    return b;
}
static id<MTLBuffer> scratch(Eng* e, uint64_t bytes, const char* label) { return alloc_k(e, bytes, label, &e->mem.scratch); }

static id<MTLComputePipelineState> pipe_(Eng* e, const char* name, int fmt, int T) {
    NSString* key = [NSString stringWithFormat:@"%s|%d|%d", name, fmt, T];
    id<MTLComputePipelineState> p = e->pipes[key];
    if (p) return p;
    NSError* err = nil;
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    short f = (short) fmt, t = (short) T;
    [cv setConstantValue:&f type:MTLDataTypeShort atIndex:0];
    [cv setConstantValue:&t type:MTLDataTypeShort atIndex:1];
    id<MTLFunction> fn = [e->lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&err];
    if (!fn) { fprintf(stderr, "mova engine: kernel %s: %s\n", name, err.localizedDescription.UTF8String); abort(); }
    p = [e->dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "mova engine: pipeline %s: %s\n", name, err.localizedDescription.UTF8String); abort(); }
    e->pipes[key] = p;
    return p;
}

#define MAX_ENC 2048
typedef struct {
    Eng* e;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    int group;                            // timing group of the commands being encoded
    int n_enc;                            // timing mode: encoders in this command buffer (one per group run)
    short groups[MAX_ENC];
} Cmd;

// Timing mode: one compute encoder per run of a kernel group, GPU timestamps at its start and end (stage-boundary
// counter sampling), summed per group when the command buffer completes.
static void open_encoder(Cmd* c, int group) {
    Eng* e = c->e;
    if (!e->timing_on || !e->csb || c->n_enc >= MAX_ENC) {
        if (!c->enc) c->enc = [c->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        c->group = group;
        return;
    }
    MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.dispatchType = MTLDispatchTypeSerial;
    MTLComputePassSampleBufferAttachmentDescriptor* att = pd.sampleBufferAttachments[0];
    att.sampleBuffer = e->csb;
    att.startOfEncoderSampleIndex = (NSUInteger) (2 * c->n_enc);
    att.endOfEncoderSampleIndex = (NSUInteger) (2 * c->n_enc + 1);
    c->enc = [c->cb computeCommandEncoderWithDescriptor:pd];
    c->groups[c->n_enc++] = (short) group;
    c->group = group;
}
static Cmd cmd_begin(Eng* e) {
    Cmd c;
    c.e = e;
    c.cb = [e->queue commandBufferWithUnretainedReferences];
    c.enc = nil;
    c.n_enc = 0;
    c.group = -1;
    open_encoder(&c, -1);
    return c;
}
static int cmd_wait(Cmd* c) {
    [c->enc endEncoding];
    [c->cb commit];
    [c->cb waitUntilCompleted];
    if (c->cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "mova engine: GPU error: %s\n", c->cb.error.localizedDescription.UTF8String);
        return -1;
    }
    Eng* e = c->e;
    if (e->timing_on && c->n_enc > 0 && e->csb) {
        NSData* d = [e->csb resolveCounterRange:NSMakeRange(0, (NSUInteger) (2 * c->n_enc))];
        const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*) d.bytes;
        // Encoders without dispatches (or samples the GPU dropped) read 0 / MTLCounterErrorValue: skipped.
        uint64_t lo = UINT64_MAX, hi = 0;
        double ticks = 0;
        for (int i = 0; i < c->n_enc; ++i) {
            const uint64_t s = ts[2 * i].timestamp, f = ts[2 * i + 1].timestamp;
            if (!s || !f || s == MTLCounterErrorValue || f == MTLCounterErrorValue || f < s) continue;
            if (s < lo) lo = s;
            if (f > hi) hi = f;
            ticks += (double) (f - s);
        }
        // GPU timestamp ticks -> seconds through the command buffer's own GPU time over the sampled span.  Consecutive
        // encoders overlap on this GPU, so each group is charged the time from the previous encoder's end to its own
        // end: the groups partition the command buffer's GPU time exactly.
        const double span = hi > lo ? (double) (hi - lo) : 0;
        const double scale = span > 0 ? (c->cb.GPUEndTime - c->cb.GPUStartTime) / span : 0;
        uint64_t prev = lo;
        for (int i = 0; i < c->n_enc; ++i) {
            const uint64_t s = ts[2 * i].timestamp, f = ts[2 * i + 1].timestamp;
            if (!s || !f || s == MTLCounterErrorValue || f == MTLCounterErrorValue || f < s) continue;
            const int g = c->groups[i] < 0 ? MOVA_TG_EMBED_NORM : c->groups[i];
            if (f > prev) { e->tg_sec[g] += scale * (double) (f - prev); prev = f; }
        }
        if (getenv("MOVA_TS_DEBUG"))
            fprintf(stderr, "ts n_enc %d  span %.0f  ticks %.0f  cb %.6f s\n", c->n_enc, span, ticks, c->cb.GPUEndTime - c->cb.GPUStartTime);
    }
    return 0;
}
static void cmd_group(Cmd* c, int group) {
    if (!c->e->timing_on || group == c->group) return;
    [c->enc endEncoding];
    c->enc = nil;
    open_encoder(c, group);
}
static void cpipe(Cmd* c, id<MTLComputePipelineState> p) { [c->enc setComputePipelineState:p]; }
static void cbuf(Cmd* c, int i, id<MTLBuffer> b, uint64_t off) { [c->enc setBuffer:b offset:off atIndex:i]; }
static void cbytes(Cmd* c, int i, const void* p, size_t n) { [c->enc setBytes:p length:n atIndex:i]; }
static void crun(Cmd* c, uint64_t x, uint64_t y, uint64_t z, uint64_t tx) {
    [c->enc dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
}

// ---- weights --------------------------------------------------------------------------------------------------------

// A tensor of the model folder: page-aligned streams map without a copy, the others are copied.
static int from_model(Eng* e, const char* name, MW* w, char* err, int errlen) {
    const NsTensor* t = ns_find(&e->nm, name);
    if (!t) return 1;
    *w = (MW){0};
    w->fmt = t->enc; w->slices = t->slices; w->rows = t->rows; w->cols = t->cols;
    for (int s = 0; s < 4; ++s) {
        const NsStream* st = &t->s[s];
        if (!st->len) continue;
        id<MTLBuffer> b;
        if (ns_mappable(&e->nm, st)) {
            b = [e->dev newBufferWithBytesNoCopy:(void*) st->p length:(st->len + NS_PAGE - 1) & ~(uint64_t) (NS_PAGE - 1)
                                         options:MTLResourceStorageModeShared deallocator:nil];
            if (b) { [e->buffers addObject:b]; e->mem.weights += (int64_t) st->len; }
        } else if ((b = alloc_k(e, st->len, name, &e->mem.weights))) memcpy(b.contents, st->p, st->len);
        if (!b) { snprintf(err, (size_t) errlen, "%s: cannot map stream %d", name, s); return -1; }
        w->b[s] = b;
    }
    return 0;
}

// BF16 from the checkpoint (slices > 1: the stacked experts, slice names from mova_slice_name; MLA's per-head maps are
// one 3-D tensor).
static int from_ckpt(Eng* e, const MovaTensor* t, MW* w, char* err, int errlen) {
    if (!e->ck && !(e->ck = mova_ckpt_open(e->model_dir, err, errlen))) return -1;
    *w = (MW){0};
    w->fmt = MF_BF16; w->slices = t->slices; w->rows = t->rows; w->cols = t->cols;
    const uint64_t one = (uint64_t) t->rows * t->cols * 2;
    id<MTLBuffer> b = alloc_k(e, one * (uint64_t) t->slices, t->name, &e->mem.weights);
    if (!b) { snprintf(err, (size_t) errlen, "%s: out of memory", t->name); return -1; }
    if (t->kind == MOVA_K_HEADS) {
        const uint16_t* src = mova_ckpt_bf16_3d(e->ck, t->name, t->slices, t->rows, t->cols, err, errlen);
        if (!src) return -1;
        memcpy(b.contents, src, one * (uint64_t) t->slices);
        w->b[0] = b;
        return 0;
    }
    char nm[128];
    for (int s = 0; s < t->slices; ++s) {
        const uint16_t* src = mova_ckpt_bf16(e->ck, mova_slice_name(t, s, nm, sizeof nm), t->rows, t->cols, err, errlen);
        if (!src) return -1;
        memcpy((uint8_t*) b.contents + one * (uint64_t) s, src, one);
    }
    w->b[0] = b;
    return 0;
}

static int load_tensor(Eng* e, const MovaTensor* all, int n, const char* name, MW* w, char* err, int errlen) {
    const MovaTensor* t = mova_find(all, n, name);
    if (!t) { snprintf(err, (size_t) errlen, "no logical tensor %s", name); return -1; }
    const int r = from_model(e, name, w, err, errlen);
    if (r < 0) return -1;
    if (r == 0) {
        if (w->slices != t->slices || w->rows != t->rows || w->cols != t->cols) {
            snprintf(err, (size_t) errlen, "%s: shape %d x %d x %d, expected %d x %d x %d", name, w->slices, w->rows, w->cols,
                     t->slices, t->rows, t->cols);
            return -1;
        }
        const int ok = t->kind == MOVA_K_EXPERTS ? 1
                       : (t->kind == MOVA_K_ROUTER || t->kind == MOVA_K_NORM || t->kind == MOVA_K_ROUTER_BIAS ||
                          t->kind == MOVA_K_HEADS) ? w->fmt == MF_BF16
                       : t->kind == MOVA_K_EMBED ? (w->fmt == MF_BF16 || w->fmt == MF_Q8)
                       : w->fmt != MF_SEED4 || t->kind == MOVA_K_VEXPERTS || t->kind == MOVA_K_LINEAR || t->kind == MOVA_K_HEAD;
        if (!ok) { snprintf(err, (size_t) errlen, "%s: encoding %d not supported for this tensor", name, w->fmt); return -1; }
        return 0;
    }
    return from_ckpt(e, t, w, err, errlen);
}

// ---- open / close ---------------------------------------------------------------------------------------------------

static void make_resident(Eng* e) {
    if (!e->residency) return;
    for (id<MTLBuffer> b in e->buffers) [e->residency addAllocation:b];
    [e->residency commit];
    [e->residency requestResidency];
}

// MLA cache bytes a position (nslm/kvq.h): the latent's n values (FP8; FP4: the first MLA_FP4_LEAD in FP8) and the
// RoPE key's (FP8 in both), codes then block scales
// The GPU's core count (IORegistry: AGXAccelerator gpu-core-count), 0 when not found
static int gpu_core_count(void) {
    int n = 0;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS) return 0;
    for (io_object_t s; !n && (s = IOIteratorNext(it));) {
        CFTypeRef v = IORegistryEntryCreateCFProperty(s, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (v && CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef) v, kCFNumberIntType, &n);
        if (v) CFRelease(v);
        IOObjectRelease(s);
    }
    IOObjectRelease(it);
    return n;
}

_Static_assert(MLA_FP4_LEAD == KVQ_FP4_LEAD, "the Metal FP4 cache layout is nslm/kvq.h's");
static int kv_lead(const Eng* e, int n) { return e->kv_fmt == ENG_KV_FP4 ? (n < MLA_FP4_LEAD ? n : MLA_FP4_LEAD) : n; }
static uint64_t kv_row(const Eng* e, int n) { return e->kv_fmt >= ENG_KV_FP8 ? (uint64_t) (kv_lead(e, n) + (n - kv_lead(e, n)) / 2) : 2 * (uint64_t) n; }
static uint64_t kv_srow(const Eng* e, int n) { return e->kv_fmt >= ENG_KV_FP8 ? (uint64_t) (kv_lead(e, n) / 32 + (n - kv_lead(e, n)) / 16) : 0; }
static uint64_t kv_rowk(const Eng* e, int n) { return e->kv_fmt >= ENG_KV_FP8 ? (uint64_t) n : 2 * (uint64_t) n; }
static uint64_t kv_srowk(const Eng* e, int n) { return e->kv_fmt >= ENG_KV_FP8 ? (uint64_t) n / 32 : 0; }

// Layer l's KV buffers' bytes per position: K (MLA: the RoPE key), V (MLA: the latent), their scales (0: none)
static void kv_rbs(const Eng* e, int l, uint64_t rb[4]) {
    const MovaCfg* c = &e->c;
    if (c->mla) {
        rb[0] = kv_rowk(e, c->mla_rope); rb[1] = kv_row(e, e->L[l].mla_r);
        rb[2] = kv_srowk(e, c->mla_rope); rb[3] = kv_srow(e, e->L[l].mla_r);
    } else {
        rb[0] = rb[1] = (uint64_t) c->n_kv * c->head_dim * (e->kv_q8 ? 1 : 2);
        rb[2] = rb[3] = e->kv_q8 ? (uint64_t) c->n_kv * 4 : 0;
    }
}
// One KV buffer of rb bytes per position for every slot.  Paged: a placement sparse buffer (pages mapped by kvp_back);
// otherwise memory for all of it.  e->mem.kv counts kv_cap positions either way.
static id<MTLBuffer> kv_alloc(Eng* e, uint64_t rb, const char* label) {
    id<MTLBuffer> b = nil;
    if (e->kvp) {
        if (@available(macOS 26.0, *)) {
            const uint64_t bytes = (uint64_t) e->slot_stride * (uint64_t) e->nseqs * rb;   // whole pages (slot_stride)
            b = [e->dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate placementSparsePageSize:MTLSparsePageSize64];
            if (b) {
                b.label = [NSString stringWithUTF8String:label];
                [e->buffers addObject:b];
                int32_t* m = (int32_t*) malloc(sizeof(int32_t) * (size_t) (bytes / KVP_PAGE));
                for (uint64_t i = 0; i < bytes / KVP_PAGE; ++i) m[i] = -1;
                e->kvp_map[e->kv_bufs.count] = m;
                e->mem.kv += (int64_t) ((uint64_t) e->kv_cap * rb);
            }
        }
    } else {
        b = alloc_k(e, (uint64_t) e->kv_cap * rb, label, &e->mem.kv);
    }
    if (b) { e->kv_rb[e->kv_bufs.count] = rb; [e->kv_bufs addObject:b]; }
    return b;
}
// A free heap page (heap * KVP_HEAP + page), from the lowest heap with one (so higher heaps empty and are released);
// a new heap when all are full.  -1: out of memory.
static int kvp_page(Eng* e) {
    const int nh = (int) e->kvp_heaps.count;
    for (int h = 0; h < nh; ++h) {
        if (e->kvp_heaps[h] == [NSNull null] || e->kvp_used[h] == KVP_HEAP) continue;
        for (int w = 0; w < KVP_HEAP / 64; ++w)
            if (e->kvp_free[h][w]) {
                const int b = __builtin_ctzll(e->kvp_free[h][w]);
                e->kvp_free[h][w] &= ~(1ull << b);
                ++e->kvp_used[h];
                return h * KVP_HEAP + w * 64 + b;
            }
    }
    id<MTLHeap> heap = nil;
    if (@available(macOS 26.0, *)) {
        MTLHeapDescriptor* d = [MTLHeapDescriptor new];
        d.type = MTLHeapTypePlacement;
        d.storageMode = MTLStorageModePrivate;
        d.size = (NSUInteger) KVP_HEAP * KVP_PAGE;
        d.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize64;
        heap = [e->dev newHeapWithDescriptor:d];
    }
    if (!heap) return -1;
    int h = 0;
    while (h < nh && e->kvp_heaps[h] != [NSNull null]) ++h;
    if (h == nh) {
        [e->kvp_heaps addObject:heap];
        e->kvp_used = (int*) realloc(e->kvp_used, sizeof(int) * (size_t) (nh + 1));
        e->kvp_free = (uint64_t (*)[KVP_HEAP / 64]) realloc(e->kvp_free, sizeof *e->kvp_free * (size_t) (nh + 1));
    } else e->kvp_heaps[h] = heap;
    for (int w = 0; w < KVP_HEAP / 64; ++w) e->kvp_free[h][w] = ~0ull;
    e->kvp_free[h][0] &= ~1ull;
    e->kvp_used[h] = 1;
    if (e->residency) { [e->residency addAllocation:heap]; [e->residency commit]; }
    return h * KVP_HEAP;
}
// Slot seq's KV pages for positions [0, n) mapped and the rest unmapped (n rounded up to KVP_GROW), in every KV buffer;
// unmapped pages go back to their heaps, and a heap with none in use is released.  0, or -1 (out of memory).
static int kvp_back(Eng* e, int seq, int n) {
    if (!e->kvp) return 0;
    Seq* s = &e->seqs[seq];
    const int want = n <= 0 ? 0 : (int) ((n + KVP_GROW - 1) / KVP_GROW * KVP_GROW < e->slot_stride ? (n + KVP_GROW - 1) / KVP_GROW * KVP_GROW : e->slot_stride);
    if (want == s->backed) return 0;
    int rc = 0;
    if (@available(macOS 26.0, *)) {
        id<MTL4CommandQueue> q = (id<MTL4CommandQueue>) e->kvp_q;
        MTL4UpdateSparseBufferMappingOperation ops[64];
        for (NSUInteger i = 0; i < e->kv_bufs.count && !rc; ++i) {
            id<MTLBuffer> b = e->kv_bufs[i];
            const uint64_t rb = e->kv_rb[i], p0 = (uint64_t) seq * (uint64_t) e->slot_stride * rb / KVP_PAGE;
            const uint64_t had = ((uint64_t) s->backed * rb + KVP_PAGE - 1) / KVP_PAGE, need = ((uint64_t) want * rb + KVP_PAGE - 1) / KVP_PAGE;
            int32_t* m = e->kvp_map[i];
            int no = 0, oh = -1;   // ops batched per heap (Metal: one heap per call)
            for (uint64_t t = p0 + had; t < p0 + need; ++t) {
                const int g = kvp_page(e);
                if (g < 0) { rc = -1; break; }
                m[t] = g;
                const int h = g / KVP_HEAP;
                if (no && (h != oh || no == 64)) {
                    [q updateBufferMappings:b heap:e->kvp_heaps[oh] operations:ops count:(NSUInteger) no];
                    no = 0;
                }
                if (no && ops[no - 1].bufferRange.location + ops[no - 1].bufferRange.length == t &&
                    ops[no - 1].heapOffset + ops[no - 1].bufferRange.length == (NSUInteger) (g % KVP_HEAP))
                    ++ops[no - 1].bufferRange.length;
                else ops[no++] = (MTL4UpdateSparseBufferMappingOperation) {MTLSparseTextureMappingModeMap, NSMakeRange((NSUInteger) t, 1), (NSUInteger) (g % KVP_HEAP)};
                oh = h;
            }
            if (no) [q updateBufferMappings:b heap:e->kvp_heaps[oh] operations:ops count:(NSUInteger) no];
            if (need < had) {
                const MTL4UpdateSparseBufferMappingOperation un = {MTLSparseTextureMappingModeUnmap, NSMakeRange((NSUInteger) (p0 + need), (NSUInteger) (had - need)), 0};
                [q updateBufferMappings:b heap:nil operations:&un count:1];
                for (uint64_t t = p0 + need; t < p0 + had; ++t) {
                    const int g = m[t], h = g / KVP_HEAP, k = g % KVP_HEAP;
                    m[t] = -1;
                    e->kvp_free[h][k / 64] |= 1ull << (k % 64);
                    --e->kvp_used[h];
                }
            }
        }
        for (NSUInteger i = 0; rc && i < e->kv_bufs.count; ++i) {   // out of memory: what this call mapped goes back
            const uint64_t rb = e->kv_rb[i], p0 = (uint64_t) seq * (uint64_t) e->slot_stride * rb / KVP_PAGE;
            const uint64_t had = ((uint64_t) s->backed * rb + KVP_PAGE - 1) / KVP_PAGE, need = ((uint64_t) want * rb + KVP_PAGE - 1) / KVP_PAGE;
            int32_t* m = e->kvp_map[i];
            for (uint64_t t = p0 + had; t < p0 + need; ++t) {
                if (m[t] < 0) continue;
                const MTL4UpdateSparseBufferMappingOperation un = {MTLSparseTextureMappingModeUnmap, NSMakeRange((NSUInteger) t, 1), 0};
                [q updateBufferMappings:e->kv_bufs[i] heap:nil operations:&un count:1];
                const int h = m[t] / KVP_HEAP, k = m[t] % KVP_HEAP;
                e->kvp_free[h][k / 64] |= 1ull << (k % 64);
                --e->kvp_used[h];
                m[t] = -1;
            }
        }
        [q signalEvent:e->kvp_ev value:++e->kvp_evv];
        [e->kvp_ev waitUntilSignaledValue:e->kvp_evv timeoutMS:60000];
        for (NSUInteger h = 0; h < e->kvp_heaps.count; ++h)   // heaps with no page in use go back to the OS
            if (e->kvp_heaps[h] != [NSNull null] && !e->kvp_used[h]) {
                if (e->residency) { [e->residency removeAllocation:e->kvp_heaps[h]]; [e->residency commit]; }
                e->kvp_heaps[h] = [NSNull null];
            }
    }
    if (!rc) s->backed = want;
    return rc;
}

Eng* eng_open(const EngOpts* o, char* err, int errlen) {
    @autoreleasepool {
        if (o->kv_format < ENG_KV_BF16 || o->kv_format > ENG_KV_FP4) {   // engine_api.h: never ignore a format
            snprintf(err, (size_t) errlen, "KV format %d not supported by the Metal engine", o->kv_format);
            return NULL;
        }
        Eng* e = (Eng*) calloc(1, sizeof *e);
        e->buffers = [NSMutableArray new];
        e->pipes = [NSMutableDictionary new];
        if (mova_cfg_load(&e->c, o->model_dir, err, errlen)) { free(e); return NULL; }
        const MovaCfg* c = &e->c;
        int kvq_ok = !c->mla || o->kv_format != ENG_KV_Q8;   // engine_api.h: never ignore a format
        for (int l = 0; l < c->n_layer && c->mla; ++l) kvq_ok &= o->kv_format < ENG_KV_FP8 || c->mla_rank[l] % 32 == 0;
        if (!kvq_ok || (!c->mla && o->kv_format >= ENG_KV_FP8)) {
            snprintf(err, (size_t) errlen, c->mla ? "MLA models: KV formats bf16, fp8 and fp4 (latent ranks a multiple of 32)"
                                                  : "GQA models: KV formats bf16 and q8");
            free(e);
            return NULL;
        }
        if (!c->mla && c->n_head != ATTF_G * c->n_kv) {   // k_attn_prefill shares each K/V tile among ATTF_G query heads
            snprintf(err, (size_t) errlen, "attention: %d query heads per KV head, the prefill kernel is built for %d", c->n_head / c->n_kv, ATTF_G);
            free(e);
            return NULL;
        }
        e->dev = MTLCreateSystemDefaultDevice();
        if (!e->dev) { snprintf(err, (size_t) errlen, "no Metal device"); free(e); return NULL; }
        e->queue = [e->dev newCommandQueue];
        NSError* nerr = nil;
        NSString* lib = [NSString stringWithFormat:@"%s/kernels_moe.metallib", o->resource_dir ? o->resource_dir : "out/res"];
        e->lib = [e->dev newLibraryWithURL:[NSURL fileURLWithPath:lib] error:&nerr];
        if (!e->lib) { snprintf(err, (size_t) errlen, "cannot load %s", lib.UTF8String); free(e); return NULL; }
        if (ns_open(&e->nm, o->model_dir, err, errlen)) { free(e); return NULL; }
        snprintf(e->model_dir, sizeof e->model_dir, "%s", o->model_dir);   // opened by from_ckpt if needed
        MovaTensor* all = NULL;
        const int na = mova_tensors(c, &all);
        e->L = (Layer*) calloc((size_t) c->n_layer, sizeof(Layer));
        char nm[160];
        int rc = 0;
#define LOAD(dst, ...) do { snprintf(nm, sizeof nm, __VA_ARGS__); if (!rc && load_tensor(e, all, na, nm, (dst), err, errlen)) rc = -1; } while (0)
        LOAD(&e->embed, "model.embed_tokens.weight");
        LOAD(&e->norm, "model.norm.weight");
        LOAD(&e->head, "lm_head.weight");
        int seeds = 0, seeds4 = 0;
        for (int l = 0; l < c->n_layer && !rc; ++l) {
            Layer* L = &e->L[l];
            L->sparse = l >= c->first_sparse;
            LOAD(&L->ln1, "model.layers.%d.input_layernorm.weight", l);
            LOAD(&L->ln2, "model.layers.%d.post_attention_layernorm.weight", l);
            LOAD(&L->q, "model.layers.%d.self_attn.q_proj.weight", l);
            if (!c->mla) LOAD(&L->k, "model.layers.%d.self_attn.k_proj.weight", l);
            LOAD(&L->o, "model.layers.%d.self_attn.o_proj.weight", l);
            LOAD(&L->g, "model.layers.%d.self_attn.gate_proj.weight", l);
            if (c->mla) {
                L->mla_r = c->mla_rank[l];
                LOAD(&L->ka_x, "model.layers.%d.self_attn.mla.kv_a_x", l);
                if (L->sparse) LOAD(&L->ka_v, "model.layers.%d.self_attn.mla.kv_a_v", l);
                LOAD(&L->kr, "model.layers.%d.self_attn.mla.k_rope_proj", l);
                LOAD(&L->qm, "model.layers.%d.self_attn.mla.q_rope_mix", l);
                LOAD(&L->ql, "model.layers.%d.self_attn.mla.q_lat", l);
                LOAD(&L->vu, "model.layers.%d.self_attn.mla.v_up", l);
            }
            if (!L->sparse) {
                if (!c->mla) LOAD(&L->v, "model.layers.%d.self_attn.v_proj.weight", l);
                LOAD(&L->mg, "model.layers.%d.mlp.gate_proj.weight", l);
                LOAD(&L->mu, "model.layers.%d.mlp.up_proj.weight", l);
                LOAD(&L->md, "model.layers.%d.mlp.down_proj.weight", l);
                continue;
            }
            LOAD(&L->vr, "model.layers.%d.self_attn.v_router.weight", l);
            LOAD(&L->vb, "model.layers.%d.self_attn.v_router.bias", l);
            LOAD(&L->vx, "model.layers.%d.self_attn.v_experts.weight", l);
            LOAD(&L->r, "model.layers.%d.mlp.gate.weight", l);
            LOAD(&L->rb, "model.layers.%d.mlp.gate.bias", l);
            LOAD(&L->eg, "model.layers.%d.mlp.experts.gate_proj.weight", l);
            LOAD(&L->eu, "model.layers.%d.mlp.experts.up_proj.weight", l);
            LOAD(&L->ed, "model.layers.%d.mlp.experts.down_proj.weight", l);
            LOAD(&L->sg, "model.layers.%d.mlp.shared_experts.gate_proj.weight", l);
            LOAD(&L->su, "model.layers.%d.mlp.shared_experts.up_proj.weight", l);
            LOAD(&L->sd, "model.layers.%d.mlp.shared_experts.down_proj.weight", l);
            seeds += L->eg.fmt == MF_SEED4 || L->ed.fmt == MF_SEED4;
            seeds4 += L->eg.fmt == MF_SEED4P4 || L->eu.fmt == MF_SEED4P4 || L->ed.fmt == MF_SEED4P4;
        }
#undef LOAD
        free(all);
        if (rc) { eng_close(e); return NULL; }
        if (seeds) {   // the per-seed stream table
            e->stab = alloc_k(e, 65536 * 4, "stab", &e->mem.lut);
            uint32_t* g = (uint32_t*) e->stab.contents;
            for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream24((uint16_t) s);
        } else e->stab = scratch(e, 256, "stab-dummy");
        if (seeds4) {   // the 32-bit stream table (SEED4P4)
            e->stab32 = alloc_k(e, 65536 * 4, "stab32", &e->mem.lut);
            uint32_t* g = (uint32_t*) e->stab32.contents;
            for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream32((uint16_t) s);
        } else e->stab32 = e->stab;
        // KV caches
        const int kvd = c->n_kv * c->head_dim;
        e->kv_cap = o->kv_tokens > 0 ? o->kv_tokens : 32768;
        e->nseqs = o->max_seqs > 0 ? o->max_seqs : 1;
        e->slot_cap = e->kv_cap / e->nseqs;
        if (e->slot_cap < 1 || e->kv_cap > INT32_MAX) {
            snprintf(err, (size_t) errlen, "KV cache: %lld tokens for %d sequences", (long long) e->kv_cap, e->nseqs);
            eng_close(e);
            return NULL;
        }
        e->kv_q8 = o->kv_format == ENG_KV_Q8;   // int8 values (half of BF16's bytes) and a scale per (position, head)
        e->kv_fmt = o->kv_format;
        e->Kc = (__strong id<MTLBuffer>*) calloc((size_t) c->n_layer, sizeof(id<MTLBuffer>));
        e->Vc = (__strong id<MTLBuffer>*) calloc((size_t) c->n_layer, sizeof(id<MTLBuffer>));
        e->Ks = (__strong id<MTLBuffer>*) calloc((size_t) c->n_layer, sizeof(id<MTLBuffer>));
        e->Vs = (__strong id<MTLBuffer>*) calloc((size_t) c->n_layer, sizeof(id<MTLBuffer>));
        int rmax = 0;
        for (int l = 0; l < c->n_layer; ++l) if (c->mla && e->L[l].mla_r > rmax) rmax = e->L[l].mla_r;
        // Paged KV (placement sparse buffers): slots start on a page of every KV buffer
        if (@available(macOS 26.4, *)) e->kvp = !getenv("NSLM_KV_DENSE") && e->dev.supportsPlacementSparse;
        e->slot_stride = e->slot_cap;
        if (e->kvp) {
            int64_t al = 1;
            for (int l = 0; l < c->n_layer; ++l) {
                uint64_t rb[4];
                kv_rbs(e, l, rb);
                for (int i = 0; i < 4; ++i) {
                    if (!rb[i]) continue;
                    uint64_t g = KVP_PAGE, x = rb[i];
                    while (x) { const uint64_t t = g % x; g = x; x = t; }
                    if ((int64_t) (KVP_PAGE / g) > al) al = (int64_t) (KVP_PAGE / g);
                }
            }
            e->slot_stride = (e->slot_cap + al - 1) / al * al;
            e->kvp_q = [e->dev newMTL4CommandQueue];
            e->kvp_ev = [e->dev newSharedEvent];
            e->kvp_heaps = [NSMutableArray new];
            if (!e->kvp_q || !e->kvp_ev) e->kvp = 0;
        }
        e->kv_bufs = [NSMutableArray new];
        e->kv_rb = (uint64_t*) calloc((size_t) c->n_layer * 4, sizeof(uint64_t));
        e->kvp_map = (int32_t**) calloc((size_t) c->n_layer * 4, sizeof(int32_t*));
        static const char* const kv_names[4] = {"K", "V", "K scales", "V scales"};
        for (int l = 0; l < c->n_layer; ++l) {   // MLA: the RoPE key and the latent (BF16, or FP8 / FP4 codes + block scales)
            uint64_t rb[4];
            kv_rbs(e, l, rb);
            id<MTLBuffer> __strong* dst[4] = {&e->Kc[l], &e->Vc[l], &e->Ks[l], &e->Vs[l]};
            for (int i = 0; i < 4; ++i) {
                if (!rb[i]) continue;
                id<MTLBuffer> b = kv_alloc(e, rb[i], kv_names[i]);
                if (!b) {
                    snprintf(err, (size_t) errlen, "KV cache: out of memory");
                    eng_close(e);
                    return NULL;
                }
                *dst[i] = b;
            }
        }
        // scratch
        const int T = MAX_ROWS, d = c->d, qd = c->n_head * c->head_dim, ffmax = c->ff_dense > c->ff_exp ? c->ff_dense : c->ff_exp;
        e->x = scratch(e, (uint64_t) T * d * 4, "x");
        e->xn = scratch(e, (uint64_t) T * d * 4, "xn");
        e->q = scratch(e, (uint64_t) T * qd * 4, "q");
        e->gq = scratch(e, (uint64_t) T * qd * 4, "gq");
        e->ao = scratch(e, (uint64_t) T * qd * 4, "ao");
        e->k = scratch(e, (uint64_t) T * kvd * 4, "k");
        e->v = scratch(e, (uint64_t) T * kvd * 4, "v");
        e->ga = scratch(e, (uint64_t) T * ffmax * 4, "ga");
        e->ua = scratch(e, (uint64_t) T * ffmax * 4, "ua");
        e->aa = scratch(e, (uint64_t) T * ffmax * 4, "aa");
        e->G = scratch(e, (uint64_t) MAXP * c->ff_exp * 4, "G");
        e->U = scratch(e, (uint64_t) MAXP * c->ff_exp * 4, "U");
        e->A = scratch(e, (uint64_t) MAXP * c->ff_exp * 4, "A");
        e->D = scratch(e, (uint64_t) MAXP * d * 4, "D");
        e->V = scratch(e, (uint64_t) MAXP * kvd * 4, "V");
        e->sh = scratch(e, (uint64_t) T * d * 4, "sh");
        e->logits = scratch(e, (uint64_t) MAX_LOGIT_ROWS * c->vocab * 4, "logits");
        e->perm = scratch(e, (uint64_t) MAXP * 4, "perm");
        e->tiles = scratch(e, (uint64_t) MAX_TILES * sizeof(MmTile), "tiles");
        e->vperm = scratch(e, (uint64_t) MAXP * 4, "vperm");
        e->vtiles = scratch(e, (uint64_t) MAX_TILES * sizeof(MmTile), "vtiles");
        e->ids = scratch(e, (uint64_t) T * 4, "ids");
        e->ri = scratch(e, (uint64_t) T * sizeof(RowInfo), "rowinfo");
        e->inv = scratch(e, 64 * 4, "invfreq");
        // rows x splits <= 1024 (encode_forward); MLA splits only forwards of <= MV_MAXT rows (prompt chunks: one pass)
        const uint64_t prow = c->mla && !getenv("NSLM_BATCH_GEMM") ? (uint64_t) MV_MAXT * MAX_SPLITS : 1024;
        e->batch_gemm = getenv("NSLM_BATCH_GEMM") ? atoi(getenv("NSLM_BATCH_GEMM")) : 0;
        e->expand = 1;
        e->gpu_cores = gpu_core_count();
        e->expand_min = o->mla_expand_min > 0 ? o->mla_expand_min : 256;
        if (e->batch_gemm && e->batch_gemm <= MV_MAXT) e->batch_gemm = MV_MAXT + 1;
        e->part = scratch(e, prow * c->n_head * ((rmax > ATT_HD ? rmax : ATT_HD) + 2) * 4, "attn partials");
        if (c->mla) {
            e->lat = scratch(e, (uint64_t) T * rmax * 4, "MLA latent");
            e->qrp = scratch(e, (uint64_t) T * qd * 4, "MLA query RoPE parts");
            e->qlat = scratch(e, (uint64_t) T * c->n_head * rmax * 4, "MLA absorbed queries");
            e->olat = scratch(e, (uint64_t) T * c->n_head * rmax * 4, "MLA latent outputs");
            e->latf = scratch(e, (uint64_t) MLP_KB * rmax * 4, "MLA latent rows (f32)");
            e->kn = scratch(e, (uint64_t) (MLP_KB + MLPF_BK) * qd * 4, "MLA expanded keys");   // + a tile: finite
            e->vn = scratch(e, (uint64_t) (MLP_KB + MLPF_BK) * qd * 4, "MLA expanded values");
            memset(e->kn.contents, 0, e->kn.length);
            memset(e->vn.contents, 0, e->vn.length);
            e->mst = scratch(e, (uint64_t) T * c->n_head * 4, "MLA prompt softmax max");
            e->lst = scratch(e, (uint64_t) T * c->n_head * 4, "MLA prompt softmax sums");
            e->ost = scratch(e, (uint64_t) T * qd * 4, "MLA prompt attention state");
            if (pipe_(e, "k_mla_attn", 0, 0).maxTotalThreadsPerThreadgroup < 32 * MLAF_SG) {
                snprintf(err, (size_t) errlen, "k_mla_attn: fewer than %d threads per threadgroup", 32 * MLAF_SG);
                eng_close(e);
                return NULL;
            }
        }
        e->inds = scratch(e, (uint64_t) c->n_layer * T * c->top_k * 4, "inds");       // per layer (route capture)
        e->wts = scratch(e, (uint64_t) c->n_layer * T * c->top_k * 4, "wts");
        e->vinds = scratch(e, (uint64_t) c->n_layer * T * c->top_kv * 4, "vinds");
        e->vwts = scratch(e, (uint64_t) c->n_layer * T * c->top_kv * 4, "vwts");
        e->rsel = scratch(e, (uint64_t) c->n_layer * T * c->n_exp * 4, "router selection scores");
        e->vsel = scratch(e, (uint64_t) c->n_layer * T * c->n_vexp * 4, "value router selection scores");
        e->rscore = scratch(e, (uint64_t) T * 128 * 4, "router scores");
        e->am = scratch(e, (uint64_t) T * 4, "argmax");
        float* inv = (float*) e->inv.contents;
        for (int p = 0; p < 64; ++p) inv[p] = (float) pow((double) c->rope_theta, -2.0 * p / (double) c->head_dim);
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
            e->residency = [e->dev newResidencySetWithDescriptor:rd error:nil];
            if (e->residency) [e->queue addResidencySet:e->residency];
        }
        make_resident(e);
        e->seqs = (Seq*) calloc((size_t) e->nseqs, sizeof(Seq));
        for (int s = 0; s < e->nseqs; ++s) {
            e->seqs[s].cap = 1024;
            e->seqs[s].hist = (int32_t*) malloc(sizeof(int32_t) * 1024);
        }
        const char* fm[5] = {"bf16", "seed4", "q8", "q4", "seed4p4"};
        char mla[48] = "";
        if (c->mla) snprintf(mla, sizeof mla, ", MLA latent %d (+%d RoPE)", rmax, c->mla_rope);
        snprintf(e->desc, sizeof e->desc, "mova engine: experts %s/%s/%s, attention %s, value experts %s, embed %s, head %s%s%s",
                 fm[e->L[c->first_sparse].eg.fmt], fm[e->L[c->first_sparse].eu.fmt], fm[e->L[c->first_sparse].ed.fmt],
                 fm[e->L[0].q.fmt], fm[e->L[c->first_sparse].vx.fmt], fm[e->embed.fmt], fm[e->head.fmt], e->kv_q8 ? ", KV q8" : e->kv_fmt == ENG_KV_FP8 ? ", KV fp8" : e->kv_fmt == ENG_KV_FP4 ? ", KV fp4" : "", mla);
        return e;
    }
}

void eng_close(Eng* e) {
    if (!e) return;
    @autoreleasepool {
        [e->buffers removeAllObjects];
        if (e->Kc) { for (int l = 0; l < e->c.n_layer; ++l) { e->Kc[l] = nil; e->Vc[l] = nil; } free(e->Kc); free(e->Vc); }
        if (e->Ks) { for (int l = 0; l < e->c.n_layer; ++l) { e->Ks[l] = nil; e->Vs[l] = nil; } free(e->Ks); free(e->Vs); }
        if (e->kvp_map) for (NSUInteger i = 0; i < e->kv_bufs.count; ++i) free(e->kvp_map[i]);
        free(e->kvp_map); free(e->kv_rb); free(e->kvp_used); free(e->kvp_free);
        e->kv_bufs = nil; e->kvp_heaps = nil; e->kvp_q = nil; e->kvp_ev = nil;
        ns_close(&e->nm);
        if (e->ck) mova_ckpt_close(e->ck);
        free(e->L);
        if (e->seqs) for (int s = 0; s < e->nseqs; ++s) free(e->seqs[s].hist);
        free(e->seqs);
        free(e->route_mlp); free(e->route_val); free(e->route_mlp_sel); free(e->route_val_sel);
        e->pipes = nil; e->buffers = nil; e->lib = nil; e->queue = nil; e->dev = nil; e->residency = nil;
        free(e);
    }
}

const char* eng_describe(Eng* e) { return e->desc; }
int eng_vocab(Eng* e) { return e->c.vocab; }
void eng_mem(Eng* e, EngMem* m) { *m = e->mem; m->gpu_allocated = (int64_t) e->dev.currentAllocatedSize; }

// ---- forward --------------------------------------------------------------------------------------------------------

// y (+)= W x.  Dense: rows of slice 0 for T tokens (T <= 8).  Gather: P pairs, slice sel[p], input row p / xdiv.
// The seed stream table for a tensor's format, and SEED4P4's exponent codes (stream 3).
static id<MTLBuffer> stab_for(Eng* e, const MW* W) { return W->fmt == MF_SEED4P4 ? e->stab32 : e->stab; }
static void bind_nibbles(Cmd* c, const MW* W, int index) {
    if (W->fmt == MF_SEED4P4) [c->enc setBuffer:W->b[3] offset:W->o[3] atIndex:(NSUInteger) index];
    else [c->enc setBuffer:c->e->stab offset:0 atIndex:(NSUInteger) index];   // unused
}
static void enc_mv_bind(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, int T, bool add, id<MTLBuffer> sel, int P,
                        int xdiv) {
    Eng* e = c->e;
    MvArgs a = {W->cols, W->rows, sel ? P : T, xdiv, xs, ys, 0, add ? 1 : 0};
    cpipe(c, pipe_(e, "k_mv", W->fmt, sel ? 0 : T));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 1, W->b[0], W->o[0]);
    cbuf(c, 2, W->b[1] ? W->b[1] : W->b[0], W->o[1]);
    cbuf(c, 3, W->b[2] ? W->b[2] : W->b[0], W->o[2]);
    cbuf(c, 4, X, 0);
    cbuf(c, 5, Y, 0);
    cbuf(c, 6, sel ? sel : e->ids, 0);
    cbuf(c, 7, stab_for(e, W), 0);
    bind_nibbles(c, W, 8);
}
static void enc_mv(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, int T, bool add, id<MTLBuffer> sel, int P,
                   int xdiv) {
    enc_mv_bind(c, W, X, xs, Y, ys, T, add, sel, P, xdiv);
    crun(c, (uint64_t) (W->rows + MV_RPT(W->fmt) - 1) / MV_RPT(W->fmt), sel ? (uint64_t) P : 1, 1, 32 * MV_ROWS);
}
static void enc_mv_sel(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, id<MTLBuffer> sel, uint64_t soff, int P,
                       int xdiv) {
    enc_mv_bind(c, W, X, xs, Y, ys, 0, false, sel, P, xdiv);
    [c->enc setBuffer:sel offset:soff atIndex:6];
    crun(c, (uint64_t) (W->rows + MV_RPT(W->fmt) - 1) / MV_RPT(W->fmt), (uint64_t) P, 1, 32 * MV_ROWS);
}
// Routed experts' gate + up + SwiGLU over the selected (token, expert) pairs in one dispatch (k_mv_gu): A[p] = the MLP
// activation of pair p.  Gate and up share the format (they come from the same packed configuration).
static void enc_mv_gu(Cmd* c, const MW* Wg, const MW* Wu, id<MTLBuffer> X, int xs, id<MTLBuffer> A, int ys, id<MTLBuffer> sel,
                      uint64_t soff, int P, int xdiv) {
    Eng* e = c->e;
    MvArgs a = {Wg->cols, Wg->rows, P, xdiv, xs, ys, 0, 0};
    cpipe(c, pipe_(e, "k_mv_gu", Wg->fmt, 0));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 1, Wg->b[0], Wg->o[0]);
    cbuf(c, 2, Wg->b[1] ? Wg->b[1] : Wg->b[0], Wg->o[1]);
    cbuf(c, 3, Wg->b[2] ? Wg->b[2] : Wg->b[0], Wg->o[2]);
    cbuf(c, 4, X, 0);
    cbuf(c, 5, A, 0);
    [c->enc setBuffer:sel offset:soff atIndex:6];
    cbuf(c, 7, stab_for(e, Wg), 0);
    cbuf(c, 8, Wu->b[0], Wu->o[0]);
    cbuf(c, 9, Wu->b[1] ? Wu->b[1] : Wu->b[0], Wu->o[1]);
    cbuf(c, 10, Wu->b[2] ? Wu->b[2] : Wu->b[0], Wu->o[2]);
    bind_nibbles(c, Wg, 11);
    bind_nibbles(c, Wu, 12);
    crun(c, (uint64_t) (Wg->rows + MV_ROWS - 1) / MV_ROWS, (uint64_t) P, 1, 32 * MV_ROWS);
}
static void enc_norm(Cmd* c, id<MTLBuffer> X, const MW* w, id<MTLBuffer> Y, int T) {
    Eng* e = c->e;
    const int32_t d = e->c.d;
    const float eps = e->c.eps;
    cpipe(c, pipe_(e, "k_gnorm", 0, 0));
    cbytes(c, 0, &d, 4);
    cbytes(c, 1, &eps, 4);
    cbuf(c, 2, X, 0);
    cbuf(c, 3, w->b[0], w->o[0]);
    cbuf(c, 4, Y, 0);
    crun(c, (uint64_t) T, 1, 1, 256);
}
static void enc_router(Cmd* c, const MW* W, const MW* bias, id<MTLBuffer> X, int n, int k, id<MTLBuffer> inds, id<MTLBuffer> wts,
                       id<MTLBuffer> sel, uint64_t off_ik, uint64_t off_sel, int T) {
    Eng* e = c->e;
    RouterArgs a = {e->c.d, n, k, e->c.d / e->c.router_parts, e->c.route_scale};
    cpipe(c, pipe_(e, "k_router_logits", 0, 0));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 1, W->b[0], W->o[0]);
    cbuf(c, 2, bias->b[0], bias->o[0]);
    cbuf(c, 3, X, 0);
    cbuf(c, 4, e->rscore, 0);
    cbuf(c, 5, sel, off_sel);
    crun(c, (uint64_t) (n + 7) / 8, (uint64_t) T, 1, 256);
    const int32_t TT = T;
    cpipe(c, pipe_(e, "k_router_topk", 0, 0));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 1, e->rscore, 0);
    cbuf(c, 2, sel, off_sel);
    cbuf(c, 3, inds, off_ik);
    cbuf(c, 4, wts, off_ik);
    cbytes(c, 5, &TT, 4);
    [c->enc dispatchThreadgroups:MTLSizeMake((NSUInteger) T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
static void enc_swiglu(Cmd* c, id<MTLBuffer> G, id<MTLBuffer> U, id<MTLBuffer> A, int n) {
    const int32_t nn = n;
    cpipe(c, pipe_(c->e, "k_swiglu", 0, 0));
    cbuf(c, 0, G, 0);
    cbuf(c, 1, U, 0);
    cbuf(c, 2, A, 0);
    cbytes(c, 3, &nn, 4);
    crun(c, (uint64_t) (n + 255) / 256, 1, 1, 256);
}

// Dense projection: matvec for T <= 8 rows, GEMM above.
static void enc_dense(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, int T, bool add) {
    Eng* e = c->e;
    if (T <= MV_MAXT && !e->prompt) { enc_mv(c, W, X, xs, Y, ys, T, add, nil, 0, 1); return; }
    MmArgs a = {W->cols, W->rows, T, xs, ys, 1, add ? 1 : 0, 0};
    cpipe(c, pipe_(e, "k_mm", W->fmt, 1));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 1, W->b[0], W->o[0]);
    cbuf(c, 2, W->b[1] ? W->b[1] : W->b[0], W->o[1]);
    cbuf(c, 3, W->b[2] ? W->b[2] : W->b[0], W->o[2]);
    cbuf(c, 4, X, 0);
    cbuf(c, 5, Y, 0);
    cbuf(c, 6, e->ids, 0);
    cbuf(c, 7, stab_for(e, W), 0);
    cbuf(c, 8, e->ids, 0);
    bind_nibbles(c, W, 9);
    cbuf(c, 10, Y, 0);   // the gate's (unused)
    crun(c, (uint64_t) (W->rows + MM_BM - 1) / MM_BM, (uint64_t) (T + MM_BN - 1) / MM_BN, 1, 128);
}
// Grouped expert projection over the tile table built by bucket(): pairs perm[], tiles[].
static void enc_grouped(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, int xdiv, id<MTLBuffer> perm,
                        id<MTLBuffer> tiles, int ntiles) {
    Eng* e = c->e;
    if (!ntiles) return;
    MmArgs a = {W->cols, W->rows, 0, xs, ys, xdiv, 0, 0};
    cpipe(c, pipe_(e, "k_mm", W->fmt, 0));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 1, W->b[0], W->o[0]);
    cbuf(c, 2, W->b[1] ? W->b[1] : W->b[0], W->o[1]);
    cbuf(c, 3, W->b[2] ? W->b[2] : W->b[0], W->o[2]);
    cbuf(c, 4, X, 0);
    cbuf(c, 5, Y, 0);
    cbuf(c, 6, perm, 0);
    cbuf(c, 7, stab_for(e, W), 0);
    cbuf(c, 8, tiles, 0);
    bind_nibbles(c, W, 9);
    cbuf(c, 10, Y, 0);   // the gate's (unused)
    crun(c, (uint64_t) (W->rows + MM_BM - 1) / MM_BM, (uint64_t) ntiles, 1, 128);
}
// Host bucketing of T x k selections (inds) by expert: perm = pair ids grouped by expert (ascending pair id within an
// expert), tiles = runs of <= MM_BN pairs.  Returns the tile count.
static int bucket(const int32_t* inds, int T, int k, int n, id<MTLBuffer> permb, id<MTLBuffer> tilesb) {
    int cnt[128] = {0}, off[129];
    const int P = T * k;
    for (int p = 0; p < P; ++p) cnt[inds[p]]++;
    off[0] = 0;
    for (int x = 0; x < n; ++x) off[x + 1] = off[x] + cnt[x];
    int pos[128];
    memcpy(pos, off, sizeof(int) * (size_t) n);
    int32_t* perm = (int32_t*) permb.contents;
    for (int p = 0; p < P; ++p) perm[pos[inds[p]]++] = p;
    MmTile* tl = (MmTile*) tilesb.contents;
    int nt = 0;
    for (int x = 0; x < n; ++x)
        for (int s = off[x]; s < off[x + 1]; s += MM_BN) tl[nt++] = (MmTile){x, s, off[x + 1] - s < MM_BN ? off[x + 1] - s : MM_BN, 0};
    return nt;
}
// Prompt chunks: finish the command buffer so the host can read the router's choices, then continue in a new one.
static int sync_cmd(Cmd* c) {
    const int grp = c->group;
    if (cmd_wait(c)) return -1;
    *c = cmd_begin(c->e);
    cmd_group(c, grp);
    return 0;
}

// One forward of T rows (T <= MAX_ROWS): tokens at e->ids, rows at e->ri.  Head rows [h0, T) get logits (h0 = T: none);
// with T - h0 <= MAX_LOGIT_ROWS the head runs here (the caller copies e->logits and e->am), else the caller runs it.
// MLA per-head maps (k_heads_mv): y[t][h][o] = W_h[o] . x[t * xs + h * hs ..], optionally gated by g (same layout as y)
static void enc_heads_mv(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, int hs, id<MTLBuffer> G, id<MTLBuffer> Y, int T) {
    const HmvArgs a = {W->slices, W->rows, W->cols, xs, hs, G != nil, {0}};
    cpipe(c, pipe_(c->e, "k_heads_mv", 0, 0));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 1, W->b[0], W->o[0]);
    cbuf(c, 2, X, 0);
    cbuf(c, 3, G ? G : X, 0);
    cbuf(c, 4, Y, 0);
    [c->enc dispatchThreads:MTLSizeMake((NSUInteger) W->rows * 32, (NSUInteger) W->slices, (NSUInteger) T)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// The per-head maps for prompt rows: one GEMM per head (k_mm on its slice), with the gate when G is given; tr: by
// W_h transposed (read in place).
static void enc_heads_gemm(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, int hs, id<MTLBuffer> G, id<MTLBuffer> Y, int T, int tr) {
    Eng* e = c->e;
    const int H = W->slices, O = tr ? W->cols : W->rows, I = tr ? W->rows : W->cols;
    const MmArgs a = {I, O, T, xs, H * O, 1, G ? 2 : 0, 0};
    cpipe(c, pipe_(e, "k_mm", MF_BF16, tr ? 2 : 1));
    cbytes(c, 0, &a, sizeof a);
    cbuf(c, 2, W->b[0], 0);
    cbuf(c, 3, W->b[0], 0);
    cbuf(c, 6, e->ids, 0);
    cbuf(c, 7, e->stab, 0);
    cbuf(c, 8, e->ids, 0);
    cbuf(c, 9, e->stab, 0);
    for (int h = 0; h < H; ++h) {
        cbuf(c, 1, W->b[0], W->o[0] + (uint64_t) h * O * I * 2);
        cbuf(c, 4, X, (uint64_t) h * hs * 4);
        cbuf(c, 5, Y, (uint64_t) h * O * 4);
        cbuf(c, 10, G ? G : Y, (uint64_t) h * O * 4);
        crun(c, (uint64_t) (O + MM_BM - 1) / MM_BM, (uint64_t) (T + MM_BN - 1) / MM_BN, 1, 128);
    }
}
static void enc_heads_mm(Cmd* c, const MW* W, id<MTLBuffer> X, int xs, int hs, id<MTLBuffer> G, id<MTLBuffer> Y, int T) {
    enc_heads_gemm(c, W, X, xs, hs, G, Y, T, 0);
}

// MLA attention of prompt rows (consecutive positions of one slot) with the latent expanded per head: per pass of
// MLP_KB cached keys, the latent rows to f32, the heads' keys (q_lat^T c) and values (v_up c) as GEMMs, then
// k_mla_prefill (the online softmax carried across passes; the last writes the gated output into ao).
static void encode_mla_prefill(Eng* e, Cmd* c, int l, int T) {
    const MovaCfg* g = &e->c;
    Layer* L = &e->L[l];
    const int r = L->mla_r, H = g->n_head;
    const RowInfo* ri = (const RowInfo*) e->ri.contents;
    const int nk = ri[T - 1].pos + 1;
    const uint64_t kv0 = (uint64_t) ri[0].kv0;
    for (int kb0 = 0; kb0 < nk; kb0 += MLP_KB) {
        const int kb1 = kb0 + MLP_KB < nk ? kb0 + MLP_KB : nk;
        const uint32_t n = (uint32_t) ((kb1 - kb0) * r);
        cmd_group(c, MOVA_TG_ATTN_PROJ);
        const uint32_t len = (uint32_t) r;
        cpipe(c, e->kv_fmt >= ENG_KV_FP8 ? pipe_(e, "k_kv_f32", e->kv_fmt, 0) : pipe_(e, "k_bf16_f32", 0, 0));
        cbuf(c, 0, e->Vc[l], (kv0 + (uint64_t) kb0) * kv_row(e, r));
        cbuf(c, 1, e->latf, 0);
        cbytes(c, 2, &n, 4);
        if (e->kv_fmt >= ENG_KV_FP8) { cbuf(c, 3, e->Vs[l], (kv0 + (uint64_t) kb0) * kv_srow(e, r)); cbytes(c, 4, &len, 4); }
        [c->enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        enc_heads_gemm(c, &L->ql, e->latf, r, 0, nil, e->kn, kb1 - kb0, 1);
        enc_heads_mm(c, &L->vu, e->latf, r, 0, nil, e->vn, kb1 - kb0);
        cmd_group(c, MOVA_TG_ATTN);
        const MlpArgs a = {H, kb0, kb1, kb0 == 0, kb1 == nk, 1.0f / sqrtf((float) g->head_dim), T, 0};
        cpipe(c, pipe_(e, "k_mla_prefill", e->kv_fmt, 0));
        cbytes(c, 0, &a, sizeof a);
        cbuf(c, 1, e->q, 0);
        cbuf(c, 2, e->qrp, 0);
        cbuf(c, 3, e->kn, 0);
        cbuf(c, 4, e->vn, 0);
        cbuf(c, 5, e->Kc[l], 0);
        cbuf(c, 6, e->ri, 0);
        cbuf(c, 7, e->mst, 0);
        cbuf(c, 8, e->lst, 0);
        cbuf(c, 9, e->ost, 0);
        cbuf(c, 10, e->gq, 0);
        cbuf(c, 11, e->ao, 0);
        cbuf(c, 12, e->Ks[l] ? e->Ks[l] : e->Kc[l], 0);
        crun(c, (uint64_t) (T + 8 * MLPF_R - 1) / (8 * MLPF_R), (uint64_t) H, 1, 32 * MLPF_R);
    }
}

// MLA attention of layer l (after q, the gate projection and, on MoVA layers, the value experts' v): the latent
// c = kv_a_x xn (+ kv_a_v v), the RoPE key, the per-head query maps, RoPE + cache write, latent attention (split-key +
// reduce for decode, one pass for prompt chunks), then v_up with the gate into ao.
static void encode_mla_attn(Eng* e, Cmd* c, int l, int T, int ns) {
    const MovaCfg* g = &e->c;
    Layer* L = &e->L[l];
    const int d = g->d, qd = g->n_head * g->head_dim, kvd = g->n_kv * g->head_dim, r = L->mla_r, H = g->n_head;
    cmd_group(c, MOVA_TG_ATTN_PROJ);
    enc_dense(c, &L->ka_x, e->xn, d, e->lat, r, T, false);
    if (L->sparse) enc_dense(c, &L->ka_v, e->v, kvd, e->lat, r, T, true);   // c = bf16(bf16(kv_a_x x) + bf16(kv_a_v v))
    enc_dense(c, &L->kr, e->xn, d, e->k, g->mla_rope, T, false);
    const int big = (T > MV_MAXT || e->prompt) && !e->decode_rows, nsp = big ? 1 : ns, xp = big && e->expand;
    void (*heads)(Cmd*, const MW*, id<MTLBuffer>, int, int, id<MTLBuffer>, id<MTLBuffer>, int) = big ? enc_heads_mm : enc_heads_mv;
    heads(c, &L->qm, e->q, qd, g->head_dim, nil, e->qrp, T);   // prompt rows: GEMMs
    if (!xp) heads(c, &L->ql, e->q, qd, g->head_dim, nil, e->qlat, T);   // expanded: the keys instead
    cmd_group(c, MOVA_TG_ATTN);
    const MlaArgs ma = {H, r, 128, nsp, 1.0f / sqrtf((float) g->head_dim), {0}};
    cpipe(c, pipe_(e, "k_mla_rope", e->kv_fmt, 0));
    cbytes(c, 0, &ma, sizeof ma);
    cbuf(c, 1, e->qrp, 0);
    cbuf(c, 2, e->k, 0);
    cbuf(c, 3, e->lat, 0);
    cbuf(c, 4, e->Kc[l], 0);
    cbuf(c, 5, e->Vc[l], 0);
    cbuf(c, 6, e->ri, 0);
    cbuf(c, 7, e->inv, 0);
    cbuf(c, 8, e->Ks[l] ? e->Ks[l] : e->Kc[l], 0);
    cbuf(c, 9, e->Vs[l] ? e->Vs[l] : e->Vc[l], 0);
    const int nx = (H + 1) * 64 > r ? (H + 1) * 64 : r;
    [c->enc dispatchThreads:MTLSizeMake((NSUInteger) nx, (NSUInteger) T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (xp) { encode_mla_prefill(e, c, l, T); return; }
    cpipe(c, pipe_(e, "k_mla_attn", e->kv_fmt, (r + MLAF_DC - 1) / MLAF_DC));
    cbytes(c, 0, &ma, sizeof ma);
    cbuf(c, 1, e->qlat, 0);
    cbuf(c, 2, e->qrp, 0);
    cbuf(c, 3, e->Kc[l], 0);
    cbuf(c, 4, e->Vc[l], 0);
    cbuf(c, 5, e->ri, 0);
    cbuf(c, 6, nsp > 1 ? e->part : e->olat, 0);
    cbuf(c, 7, e->Ks[l] ? e->Ks[l] : e->Kc[l], 0);
    cbuf(c, 8, e->Vs[l] ? e->Vs[l] : e->Vc[l], 0);
    crun(c, (uint64_t) nsp, (uint64_t) ((H + MLAF_Q - 1) / MLAF_Q), (uint64_t) T, 32 * MLAF_SG);
    if (nsp > 1) {
        cpipe(c, pipe_(e, "k_mla_reduce", 0, 0));
        cbytes(c, 0, &ma, sizeof ma);
        cbuf(c, 1, e->part, 0);
        cbuf(c, 2, e->olat, 0);
        crun(c, (uint64_t) H, (uint64_t) T, 1, 256);
    }
    cmd_group(c, MOVA_TG_ATTN_PROJ);
    heads(c, &L->vu, e->olat, H * r, r, e->gq, e->ao, T);
    (void) kvd;
}

static int encode_forward(Eng* e, Cmd* c, int T, int max_ctx, int h0) {
    const MovaCfg* g = &e->c;
    const int d = g->d, qd = g->n_head * g->head_dim, kvd = g->n_kv * g->head_dim;
    const bool big = T > MV_MAXT || e->prompt;
    cmd_group(c, MOVA_TG_EMBED_NORM);
    {
        const int32_t dd = d;
        cpipe(c, pipe_(e, "k_embed", e->embed.fmt, 0));
        cbytes(c, 0, &dd, 4);
        cbuf(c, 1, e->embed.b[0], 0);
        cbuf(c, 2, e->embed.b[0], 0);
        cbuf(c, 3, e->embed.b[1] ? e->embed.b[1] : e->embed.b[0], 0);
        cbuf(c, 4, e->embed.b[2] ? e->embed.b[2] : e->embed.b[0], 0);
        cbuf(c, 5, e->ids, 0);
        cbuf(c, 6, e->x, 0);
        [c->enc dispatchThreads:MTLSizeMake((NSUInteger) d, (NSUInteger) T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    int ns = (max_ctx + 127) / 128;
    if (ns > MAX_SPLITS) ns = MAX_SPLITS;
    if (g->mla && e->gpu_cores > 0) {   // k_mla_attn runs one threadgroup per GPU core at a time (threadgroup memory): the
        // splits that make each row's (split, head block) threadgroups one wave (M4 Max, 8k context: 32 -> 20, +16%)
        const int w = e->gpu_cores / ((g->n_head + MLAF_Q - 1) / MLAF_Q), nk = (max_ctx + 63) / 64;
        if (w >= 1) ns = (w < nk ? w : nk) < MAX_SPLITS ? (w < nk ? w : nk) : MAX_SPLITS;
    }
    while (ns > 1 && T * ns > 1024) --ns;
    if (ns < 1) ns = 1;
    AttnArgs aa = {g->n_head, g->n_kv, 128, ns, 1.0f / sqrtf((float) g->head_dim)};   // per-row splits (k_attn)
    const char* dump = getenv("MOVA_DUMP");
    if (dump && sync_cmd(c) == 0) { FILE* f = fopen(dump, "ab"); fwrite(e->x.contents, 4, (size_t) T * d, f); fclose(f); }
    for (int l = 0; l < g->n_layer; ++l) {
        Layer* L = &e->L[l];
        if (dump && l > 0 && sync_cmd(c) == 0) { FILE* f = fopen(dump, "ab"); fwrite(e->x.contents, 4, (size_t) T * d, f); fclose(f); }
        cmd_group(c, MOVA_TG_EMBED_NORM);
        enc_norm(c, e->x, &L->ln1, e->xn, T);
        cmd_group(c, MOVA_TG_ATTN_PROJ);
        enc_dense(c, &L->q, e->xn, d, e->q, qd, T, false);
        if (!g->mla) enc_dense(c, &L->k, e->xn, d, e->k, kvd, T, false);
        enc_dense(c, &L->g, e->xn, d, e->gq, qd, T, false);
        cmd_group(c, MOVA_TG_VALUES);
        if (!L->sparse) {
            if (!g->mla) enc_dense(c, &L->v, e->xn, d, e->v, kvd, T, false);   // MLA: v_proj is folded into kv_a_x
        } else {
            const uint64_t ik = (uint64_t) l * MAX_ROWS * g->top_kv * 4, so = (uint64_t) l * MAX_ROWS * g->n_vexp * 4;
            enc_router(c, &L->vr, &L->vb, e->xn, g->n_vexp, g->top_kv, e->vinds, e->vwts, e->vsel, ik, so, T);
            if (!big) enc_mv_sel(c, &L->vx, e->xn, d, e->V, kvd, e->vinds, ik, T * g->top_kv, g->top_kv);
            else {
                if (sync_cmd(c)) return -1;
                const int nt = bucket((const int32_t*) ((const uint8_t*) e->vinds.contents + ik), T, g->top_kv, g->n_vexp, e->vperm, e->vtiles);
                enc_grouped(c, &L->vx, e->xn, d, e->V, kvd, g->top_kv, e->vperm, e->vtiles, nt);
            }
            const int32_t dk[2] = {kvd, g->top_kv};
            cpipe(c, pipe_(e, "k_vcombine", 0, 0));
            cbuf(c, 0, e->V, 0);
            cbuf(c, 1, e->vwts, ik);
            cbuf(c, 2, e->v, 0);
            cbytes(c, 3, dk, 8);
            [c->enc dispatchThreads:MTLSizeMake((NSUInteger) kvd, (NSUInteger) T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (g->mla) encode_mla_attn(e, c, l, T, ns);
        else {
            cmd_group(c, MOVA_TG_ATTN);
            const int32_t hk[2] = {g->n_head, g->n_kv};
            const int q8 = e->kv_q8;   // the 8-bit cache: the _q8 kernels, scales at buffers 8, 9 (rope, prefill) / 6, 7 (decode)
            cpipe(c, pipe_(e, q8 ? "k_rope_kv_q8" : "k_rope_kv", 0, 0));
            cbuf(c, 0, e->q, 0);
            cbuf(c, 1, e->k, 0);
            cbuf(c, 2, e->v, 0);
            cbuf(c, 3, e->Kc[l], 0);
            cbuf(c, 4, e->Vc[l], 0);
            cbuf(c, 5, e->ri, 0);
            cbuf(c, 6, e->inv, 0);
            cbytes(c, 7, hk, 8);
            if (q8) { cbuf(c, 8, e->Ks[l], 0); cbuf(c, 9, e->Vs[l], 0); }
            [c->enc dispatchThreads:MTLSizeMake((NSUInteger) g->n_head * 64, (NSUInteger) T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            if (big && !e->decode_rows) {   // prefill: key tiles on simdgroup matrices, the gate fused
                const int32_t TT = T;
                cpipe(c, pipe_(e, q8 ? "k_attn_prefill_q8" : "k_attn_prefill", 0, 0));
                cbytes(c, 0, &aa, sizeof aa);
                cbuf(c, 1, e->q, 0);
                cbuf(c, 2, e->Kc[l], 0);
                cbuf(c, 3, e->Vc[l], 0);
                cbuf(c, 4, e->ri, 0);
                cbuf(c, 5, e->gq, 0);
                cbuf(c, 6, e->ao, 0);
                cbytes(c, 7, &TT, 4);
                if (q8) { cbuf(c, 8, e->Ks[l], 0); cbuf(c, 9, e->Vs[l], 0); }
                crun(c, (uint64_t) (T + 8 * ATTF_RS - 1) / (8 * ATTF_RS), (uint64_t) g->n_kv, 1, 32 * ATTF_G * ATTF_RS);
            } else {
                cpipe(c, pipe_(e, q8 ? "k_attn_q8" : "k_attn", 0, 0));
                cbytes(c, 0, &aa, sizeof aa);
                cbuf(c, 1, e->q, 0);
                cbuf(c, 2, e->Kc[l], 0);
                cbuf(c, 3, e->Vc[l], 0);
                cbuf(c, 4, e->ri, 0);
                cbuf(c, 5, e->part, 0);
                if (q8) { cbuf(c, 6, e->Ks[l], 0); cbuf(c, 7, e->Vs[l], 0); }
                crun(c, (uint64_t) ns, (uint64_t) g->n_kv, (uint64_t) T, 32 * ATT_SG);
                cpipe(c, pipe_(e, "k_attn_reduce", 0, 0));
                cbytes(c, 0, &aa, sizeof aa);
                cbuf(c, 1, e->part, 0);
                cbuf(c, 2, e->gq, 0);
                cbuf(c, 3, e->ao, 0);
                crun(c, (uint64_t) g->n_head, (uint64_t) T, 1, ATT_HD);
            }
        }
        cmd_group(c, MOVA_TG_ATTN_PROJ);
        enc_dense(c, &L->o, e->ao, qd, e->x, d, T, true);
        cmd_group(c, MOVA_TG_EMBED_NORM);
        enc_norm(c, e->x, &L->ln2, e->xn, T);
        if (!L->sparse) {
            cmd_group(c, MOVA_TG_SHARED);
            enc_dense(c, &L->mg, e->xn, d, e->ga, g->ff_dense, T, false);
            enc_dense(c, &L->mu, e->xn, d, e->ua, g->ff_dense, T, false);
            enc_swiglu(c, e->ga, e->ua, e->aa, T * g->ff_dense);
            enc_dense(c, &L->md, e->aa, g->ff_dense, e->x, d, T, true);
            continue;
        }
        cmd_group(c, MOVA_TG_ROUTER);
        const uint64_t ik = (uint64_t) l * MAX_ROWS * g->top_k * 4, so = (uint64_t) l * MAX_ROWS * g->n_exp * 4;
        enc_router(c, &L->r, &L->rb, e->xn, g->n_exp, g->top_k, e->inds, e->wts, e->rsel, ik, so, T);
        cmd_group(c, MOVA_TG_EXPERTS);
        const int P = T * g->top_k;
        if (!big) {
            if (L->eg.fmt == L->eu.fmt) enc_mv_gu(c, &L->eg, &L->eu, e->xn, d, e->A, g->ff_exp, e->inds, ik, P, g->top_k);
            else {
                enc_mv_sel(c, &L->eg, e->xn, d, e->G, g->ff_exp, e->inds, ik, P, g->top_k);
                enc_mv_sel(c, &L->eu, e->xn, d, e->U, g->ff_exp, e->inds, ik, P, g->top_k);
                enc_swiglu(c, e->G, e->U, e->A, P * g->ff_exp);
            }
            enc_mv_sel(c, &L->ed, e->A, g->ff_exp, e->D, d, e->inds, ik, P, 1);
        } else {
            if (sync_cmd(c)) return -1;
            const int nt = bucket((const int32_t*) ((const uint8_t*) e->inds.contents + ik), T, g->top_k, g->n_exp, e->perm, e->tiles);
            enc_grouped(c, &L->eg, e->xn, d, e->G, g->ff_exp, g->top_k, e->perm, e->tiles, nt);
            enc_grouped(c, &L->eu, e->xn, d, e->U, g->ff_exp, g->top_k, e->perm, e->tiles, nt);
            enc_swiglu(c, e->G, e->U, e->A, P * g->ff_exp);
            enc_grouped(c, &L->ed, e->A, g->ff_exp, e->D, d, 1, e->perm, e->tiles, nt);
        }
        cmd_group(c, MOVA_TG_SHARED);
        enc_dense(c, &L->sg, e->xn, d, e->ga, g->ff_exp, T, false);
        enc_dense(c, &L->su, e->xn, d, e->ua, g->ff_exp, T, false);
        enc_swiglu(c, e->ga, e->ua, e->aa, T * g->ff_exp);
        enc_dense(c, &L->sd, e->aa, g->ff_exp, e->sh, d, T, false);
        cmd_group(c, MOVA_TG_EXPERTS);
        {
            const int32_t dk[2] = {d, g->top_k};
            cpipe(c, pipe_(e, "k_moe_combine", 0, 0));
            cbuf(c, 0, e->D, 0);
            cbuf(c, 1, e->wts, ik);
            cbuf(c, 2, e->sh, 0);
            cbuf(c, 3, e->x, 0);
            cbytes(c, 4, dk, 8);
            [c->enc dispatchThreads:MTLSizeMake((NSUInteger) d, (NSUInteger) T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }
    if (h0 < T) {
        cmd_group(c, MOVA_TG_HEAD);
        enc_norm(c, e->x, &e->norm, e->xn, T);
    }
    return 0;
}

// The LM head for rows [h0, h0 + n) (n <= MAX_LOGIT_ROWS) of the normed activations: logits + argmax.
static void encode_head(Eng* e, Cmd* c, int h0, int n) {
    const MovaCfg* g = &e->c;
    cmd_group(c, MOVA_TG_HEAD);
    // the head reads xn from row h0: bind with an offset by copying the rows' view (xs stride d, base offset)
    MW H = e->head;
    if (n <= MV_MAXT) {
        enc_mv_bind(c, &H, e->xn, g->d, e->logits, g->vocab, n, false, nil, 0, 1);
        [c->enc setBuffer:e->xn offset:(uint64_t) h0 * g->d * 4 atIndex:4];
        crun(c, (uint64_t) (H.rows + MV_RPT(H.fmt) - 1) / MV_RPT(H.fmt), 1, 1, 32 * MV_ROWS);
    } else {
        enc_dense(c, &H, e->xn, g->d, e->logits, g->vocab, n, false);
        [c->enc setBuffer:e->xn offset:(uint64_t) h0 * g->d * 4 atIndex:4];
        crun(c, (uint64_t) (H.rows + MM_BM - 1) / MM_BM, (uint64_t) (n + MM_BN - 1) / MM_BN, 1, 128);
    }
    const int32_t V = g->vocab;
    cpipe(c, pipe_(e, "k_argmax", 0, 0));
    cbuf(c, 0, e->logits, 0);
    cbuf(c, 1, e->am, 0);
    cbytes(c, 2, &V, 4);
    crun(c, (uint64_t) n, 1, 1, 1024);
}

static void route_collect(Eng* e, int T);

// Forward of T rows (tokens tok[0..T-1], rows[t]: position and slot cache base).  Rows [h0, T) get logits (into
// logits_out, (T - h0) x vocab, may be NULL) and arg max (into am, may be NULL); h0 = T: no head.  Rows of more than
// MV_MAXT (the prefill kernels) must be consecutive positions of one slot.
static int forward_rows(Eng* e, const int32_t* tok, const RowInfo* rows, int T, int h0, float* logits_out, int32_t* am) {
    @autoreleasepool {
        if (T < 1 || T > MAX_ROWS) return -1;
        int max_ctx = 0;
        for (int t = 0; t < T; ++t) {
            if (rows[t].pos >= e->slot_cap) { fprintf(stderr, "mova engine: KV capacity %lld exceeded\n", (long long) e->slot_cap); return -1; }
            if (rows[t].pos + 1 > max_ctx) max_ctx = rows[t].pos + 1;
            const int seq = (int) (rows[t].kv0 / e->slot_stride);
            if (e->kvp && rows[t].pos >= e->seqs[seq].backed && kvp_back(e, seq, rows[t].pos + 1)) {
                fprintf(stderr, "mova engine: KV pages: out of memory\n");
                return -1;
            }
        }
        memcpy(e->ids.contents, tok, (size_t) T * 4);
        memcpy(e->ri.contents, rows, (size_t) T * sizeof(RowInfo));
        Cmd c = cmd_begin(e);
        if (encode_forward(e, &c, T, max_ctx, h0)) { [c.enc endEncoding]; return -1; }
        for (int r = h0; r < T; r += MAX_LOGIT_ROWS) {
            const int n = T - r < MAX_LOGIT_ROWS ? T - r : MAX_LOGIT_ROWS;
            encode_head(e, &c, r, n);
            if (cmd_wait(&c)) return -1;
            if (logits_out) memcpy(logits_out + (size_t) (r - h0) * e->c.vocab, e->logits.contents, (size_t) n * e->c.vocab * 4);
            if (am) memcpy(am + (r - h0), e->am.contents, (size_t) n * 4);
            c = cmd_begin(e);
        }
        if (cmd_wait(&c)) return -1;
        if (e->route_on) route_collect(e, T);
        return 0;
    }
}
// Forward of T rows of slot seq at positions pos0 .. pos0 + T - 1.
static int forward(Eng* e, int seq, const int32_t* tok, int T, int pos0, int h0, float* logits_out, int32_t* am) {
    RowInfo ri[MAX_ROWS];
    if (T < 1 || T > MAX_ROWS) return -1;
    for (int t = 0; t < T; ++t) ri[t] = (RowInfo){pos0 + t, (int) (seq * e->slot_stride), {0}};
    return forward_rows(e, tok, ri, T, h0, logits_out, am);
}

// ---- sequences ------------------------------------------------------------------------------------------------------

static void hist_push(Seq* s, int32_t t) {
    if (s->len == s->cap) { s->cap *= 2; s->hist = (int32_t*) realloc(s->hist, sizeof(int32_t) * (size_t) s->cap); }
    s->hist[s->len++] = t;
}
static Seq* seq_of(Eng* e, int seq) { return seq >= 0 && seq < e->nseqs ? &e->seqs[seq] : NULL; }
// Distinct slots, each prefilled (only the pending token uncached).
static int ready(Eng* e, const int* seqs, int n) {
    if (n < 1) return 0;
    for (int i = 0; i < n; ++i) {
        const Seq* s = seq_of(e, seqs[i]);
        if (!s || s->len < 1 || s->done < s->len - 1) return 0;
        for (int k = 0; k < i; ++k) if (seqs[k] == seqs[i]) return 0;
    }
    return 1;
}

int eng_prefill_begin(Eng* e, int seq, const int32_t* ids, int n, int* reused) {
    Seq* s = seq_of(e, seq);
    if (!s || n < 1 || n > e->slot_cap) return -1;
    int c = 0;
    while (c < s->done && c < n - 1 && s->hist[c] == ids[c]) ++c;
    if (s->backed > n) kvp_back(e, seq, n);   // paged KV: the old conversation's pages past the new prompt go back
    s->len = s->done = c;
    s->xend = n - 1 - c >= e->expand_min ? (n - 1) / e->expand_min * e->expand_min : 0;
    for (int i = c; i < n; ++i) hist_push(s, ids[i]);
    if (reused) *reused = c;
    return 0;
}
int eng_prefill_next(Eng* e, int seq, int max_rows) {
    Seq* s = seq_of(e, seq);
    if (!s || s->len < 1 || max_rows < 1) return -1;
    const int left = s->len - 1 - s->done;
    if (left <= 0) return 0;
    int T = left < max_rows ? (left < MAX_ROWS ? left : MAX_ROWS) : (max_rows < MAX_ROWS ? max_rows : MAX_ROWS);
    if (s->done < s->xend && s->done + T > s->xend) T = s->xend - s->done;   // a forward on one side of xend
    e->prompt = 1;
    e->expand = s->done < s->xend;
    const int rc = forward(e, seq, s->hist + s->done, T, s->done, T, NULL, NULL);
    e->prompt = 0;
    e->expand = 1;
    if (rc) { s->len = s->done = 0; return -1; }   // no half-written KV
    s->done += T;
    return left - T;
}
int eng_prefill_cached(Eng* e, int seq, const int32_t* ids, int n, int* reused) {
    if (eng_prefill_begin(e, seq, ids, n, reused)) return -1;
    int left;
    while ((left = eng_prefill_next(e, seq, MAX_ROWS)) > 0) {}
    return left;
}
int eng_prefill(Eng* e, int seq, const int32_t* ids, int n) {
    Seq* s = seq_of(e, seq);
    if (!s) return -1;
    s->len = s->done = 0;
    return eng_prefill_cached(e, seq, ids, n, NULL);
}
int eng_rewind(Eng* e, int seq, int n) {
    Seq* s = seq_of(e, seq);
    if (!s || n < 1 || n > s->len) return -1;
    s->len = n;
    if (s->done > n) s->done = n;
    return 0;
}
int eng_fork(Eng* e, int dst, int src) { (void) e; return dst == src ? 0 : -1; }
void eng_free(Eng* e, int seq) {
    Seq* s = seq_of(e, seq);
    if (!s) return;
    s->len = s->done = 0;
    kvp_back(e, seq, 0);
}
int eng_len(Eng* e, int seq) { const Seq* s = seq_of(e, seq); return s ? s->len : 0; }

// A slot's positions [p0, p1) to (out) or from host bytes: per layer K rows, V rows, then (8-bit, FP8, FP4 caches) their
// scales.  Paged KV (private buffers): through a staging buffer and blits.  0, or -1.
static int kv_copy(Eng* e, int seq, int p0, int p1, uint8_t* h, int out) {
    const uint64_t r0 = (uint64_t) seq * (uint64_t) e->slot_stride + (uint64_t) p0, n = (uint64_t) (p1 - p0);
    if (!e->kvp) {
        for (NSUInteger i = 0; i < e->kv_bufs.count; ++i) {
            id<MTLBuffer> b = e->kv_bufs[i];
            const uint64_t rb = e->kv_rb[i];
            uint8_t* d = (uint8_t*) b.contents + r0 * rb;
            if (out) memcpy(h, d, n * rb); else memcpy(d, h, n * rb);
            h += n * rb;
        }
        return 0;
    }
    @autoreleasepool {
        uint64_t total = 0;
        for (NSUInteger i = 0; i < e->kv_bufs.count; ++i) total += n * e->kv_rb[i];
        if (!total) return 0;
        id<MTLBuffer> st = [e->dev newBufferWithLength:total options:MTLResourceStorageModeShared];
        if (!st) return -1;
        if (!out) memcpy(st.contents, h, total);
        id<MTLCommandBuffer> cb = [e->queue commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        uint64_t o = 0;
        for (NSUInteger i = 0; i < e->kv_bufs.count; ++i) {
            const uint64_t rb = e->kv_rb[i];
            if (out) [bl copyFromBuffer:e->kv_bufs[i] sourceOffset:r0 * rb toBuffer:st destinationOffset:o size:n * rb];
            else [bl copyFromBuffer:st sourceOffset:o toBuffer:e->kv_bufs[i] destinationOffset:r0 * rb size:n * rb];
            o += n * rb;
        }
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) return -1;
        if (out) memcpy(h, st.contents, total);
    }
    return 0;
}
int64_t eng_kv_bytes(Eng* e) {
    const MovaCfg* c = &e->c;
    int64_t b = 0;
    for (int l = 0; l < c->n_layer; ++l)
        b += c->mla ? (int64_t) (kv_rowk(e, c->mla_rope) + kv_srowk(e, c->mla_rope) + kv_row(e, e->L[l].mla_r) + kv_srow(e, e->L[l].mla_r))
                    : 2 * (int64_t) c->n_kv * (e->kv_q8 ? c->head_dim + 4 : 2 * c->head_dim);
    return b;
}
int eng_kv_read(Eng* e, int seq, int p0, int p1, void* dst) {
    const Seq* s = seq_of(e, seq);
    if (!s || p0 < 0 || p1 < p0 || p1 > s->done) return -1;
    return kv_copy(e, seq, p0, p1, (uint8_t*) dst, 1);
}
int eng_kv_write(Eng* e, int seq, const int32_t* ids, int p0, int p1, const void* src) {
    Seq* s = seq_of(e, seq);
    if (!s || p0 < 0 || p0 > s->done || p1 < p0 || p1 > e->slot_cap) return -1;
    if ((p1 > s->backed && kvp_back(e, seq, p1)) || kv_copy(e, seq, p0, p1, (uint8_t*) src, 0)) return -1;
    s->len = 0;
    for (int i = 0; i < p1; ++i) hist_push(s, ids[i]);
    s->done = p1;
    return 0;
}

int eng_step(Eng* e, int seq, float* logits) { return eng_step_batch(e, &seq, 1, logits); }
int eng_push(Eng* e, int seq, int32_t tok) {
    Seq* s = seq_of(e, seq);
    if (!s) return -1;
    hist_push(s, tok);
    return 0;
}
// The pending tokens of seqs[0..n-1] (ready()) in forwards of up to MV_MAXT rows (the decode kernels compute each row
// as a one-row forward does): logits (n x vocab) and / or arg max (n).
static int step_rows(Eng* e, const int* seqs, int n, float* logits, int32_t* am) {
    const int per = e->batch_gemm && n >= e->batch_gemm ? MAX_LOGIT_ROWS : MV_MAXT;
    for (int i0 = 0; i0 < n; i0 += per) {
        const int T = n - i0 < per ? n - i0 : per;
        int32_t tok[MAX_LOGIT_ROWS];
        RowInfo ri[MAX_LOGIT_ROWS];
        for (int t = 0; t < T; ++t) {
            const Seq* s = &e->seqs[seqs[i0 + t]];
            tok[t] = s->hist[s->len - 1];
            ri[t] = (RowInfo){s->len - 1, (int) (seqs[i0 + t] * e->slot_stride), {0}};
        }
        e->decode_rows = T > MV_MAXT;
        const int rc = forward_rows(e, tok, ri, T, 0, logits ? logits + (size_t) i0 * e->c.vocab : NULL, am ? am + i0 : NULL);
        e->decode_rows = 0;
        if (rc) return -1;
        for (int t = 0; t < T; ++t) e->seqs[seqs[i0 + t]].done = e->seqs[seqs[i0 + t]].len;
    }
    return 0;
}
int eng_step_batch(Eng* e, const int* seqs, int n, float* logits) { return ready(e, seqs, n) ? step_rows(e, seqs, n, logits, NULL) : -1; }

// Prompt-lookup speculative decoding (single stream; ENG_MODE_PL): the draft continues the most
// recent earlier occurrence of the history's last PL_NMAX..PL_NMIN tokens (up to PL_K tokens); one forward verifies
// [pending, d_1 .. d_k] (k + 1 <= MV_MAXT rows: the decode kernels, per-row attention splits, per-(token, expert)
// gathers), and the matching prefix plus the correction / bonus token is committed.  Every kernel computes a row as a
// single AR step does, so the committed tokens are AR's.
#ifndef PL_K
#define PL_K 4
#endif
#define PL_NMAX 4
#define PL_NMIN 3
static int pl_draft(const Seq* s, int32_t* d) {
    const int L = s->len;
    for (int n = PL_NMAX; n >= PL_NMIN; --n) {
        if (L < n + 1) continue;
        const int32_t* suf = s->hist + L - n;
        for (int i = L - n - 1; i >= 0; --i) {   // most recent earlier occurrence first
            if (memcmp(s->hist + i, suf, (size_t) n * 4)) continue;
            int k = 0;
            while (k < PL_K && i + n + k < L) { d[k] = s->hist[i + n + k]; ++k; }
            if (k) return k;
        }
    }
    return 0;
}
static int gen_pl(Eng* e, int seq, int n_new, int32_t* out, EngStats* st) {
    int32_t tok[PL_K + 1], am[PL_K + 1], d[PL_K];
    Seq* s = &e->seqs[seq];
    for (int done = 0; done < n_new;) {
        const int nd = pl_draft(s, d), p = s->len - 1;
        tok[0] = s->hist[p];
        for (int j = 0; j < nd; ++j) tok[j + 1] = d[j];
        if (forward(e, seq, tok, nd + 1, p, 0, NULL, am)) return -1;
        int a = 0;
        while (a < nd && d[a] == am[a]) ++a;
        for (int j = 0; j <= a && done < n_new; ++j) {
            hist_push(s, am[j]);
            out[done++] = am[j];
            if (st) st->tokens++;
        }
        s->done = s->len - 1;
        if (st) { st->forwards++; st->rows += nd + 1; st->cycles++; st->proposals += nd; st->accepted += a; }
    }
    return 0;
}

int eng_generate(Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st) {
    if (!ready(e, seqs, nseq)) return -1;
    if (mode == ENG_MODE_PL) return nseq == 1 ? gen_pl(e, seqs[0], n_new, out, st) : -1;
    int32_t* am = (int32_t*) malloc(sizeof(int32_t) * (size_t) nseq);
    for (int j = 0; j < n_new; ++j) {
        if (step_rows(e, seqs, nseq, NULL, am)) { free(am); return -1; }
        for (int i = 0; i < nseq; ++i) {
            hist_push(&e->seqs[seqs[i]], am[i]);
            out[(size_t) i * n_new + j] = am[i];
        }
        if (st) { st->tokens += nseq; st->forwards += (nseq + MV_MAXT - 1) / MV_MAXT; st->rows += nseq; }
    }
    free(am);
    return 0;
}

int eng_score(Eng* e, int seq, const int32_t* ids, int from, int count, float* logits) {
    // rows 0 .. from + count - 2; logits for rows from - 1 .. from + count - 2
    Seq* s = seq_of(e, seq);
    if (!s || from < 1 || count < 1) return -1;
    s->len = s->done = 0;   // the slot's cache is overwritten
    const int last = from + count - 2;
    for (int p = 0; p <= last; p += MAX_ROWS) {
        const int T = last + 1 - p < MAX_ROWS ? last + 1 - p : MAX_ROWS;
        int h0 = from - 1 - p;
        if (h0 < 0) h0 = 0;
        if (h0 > T) h0 = T;
        float* dst = h0 < T ? logits + (size_t) (p + h0 - (from - 1)) * e->c.vocab : NULL;
        if (forward(e, seq, ids + p, T, p, h0, dst, NULL)) return -1;
    }
    s->len = s->done = 0;
    return 0;
}

int eng_mova_routes(Eng* e, int on, int max_rows) {
    free(e->route_mlp); free(e->route_val); free(e->route_mlp_sel); free(e->route_val_sel);
    e->route_mlp = NULL; e->route_val = NULL; e->route_mlp_sel = NULL; e->route_val_sel = NULL;
    e->route_on = on;
    e->route_rows = 0;
    e->route_max = max_rows;
    if (!on) return 0;
    const MovaCfg* c = &e->c;
    const size_t ns = (size_t) (c->n_layer - c->first_sparse);
    e->route_mlp = (int32_t*) malloc((size_t) max_rows * ns * (size_t) c->top_k * 4);
    e->route_val = (int32_t*) malloc((size_t) max_rows * ns * (size_t) c->top_kv * 4);
    e->route_mlp_sel = (float*) malloc((size_t) max_rows * ns * (size_t) c->n_exp * 4);
    e->route_val_sel = (float*) malloc((size_t) max_rows * ns * (size_t) c->n_vexp * 4);
    return 0;
}
// After a forward of T rows: append their selections (called by forward()).
static void route_collect(Eng* e, int T) {
    const MovaCfg* c = &e->c;
    const int ns = c->n_layer - c->first_sparse;
    for (int t = 0; t < T && e->route_rows < e->route_max; ++t, ++e->route_rows) {
        const size_t r = (size_t) e->route_rows;
        for (int s = 0; s < ns; ++s) {
            const int l = c->first_sparse + s;
            memcpy(e->route_mlp + (r * ns + s) * c->top_k, (int32_t*) e->inds.contents + ((size_t) l * MAX_ROWS + t) * c->top_k, (size_t) c->top_k * 4);
            memcpy(e->route_val + (r * ns + s) * c->top_kv, (int32_t*) e->vinds.contents + ((size_t) l * MAX_ROWS + t) * c->top_kv, (size_t) c->top_kv * 4);
            memcpy(e->route_mlp_sel + (r * ns + s) * c->n_exp, (float*) e->rsel.contents + ((size_t) l * MAX_ROWS + t) * c->n_exp, (size_t) c->n_exp * 4);
            memcpy(e->route_val_sel + (r * ns + s) * c->n_vexp, (float*) e->vsel.contents + ((size_t) l * MAX_ROWS + t) * c->n_vexp, (size_t) c->n_vexp * 4);
        }
    }
}
int eng_mova_routes_read(Eng* e, int rows, int32_t* mlp, int32_t* val, float* mlp_sel, float* val_sel) {
    if (!e->route_on || rows > e->route_rows) return -1;
    const MovaCfg* c = &e->c;
    const size_t ns = (size_t) (c->n_layer - c->first_sparse), r0 = (size_t) (e->route_rows - rows);
    if (mlp) memcpy(mlp, e->route_mlp + r0 * ns * c->top_k, (size_t) rows * ns * c->top_k * 4);
    if (val) memcpy(val, e->route_val + r0 * ns * c->top_kv, (size_t) rows * ns * c->top_kv * 4);
    if (mlp_sel) memcpy(mlp_sel, e->route_mlp_sel + r0 * ns * c->n_exp, (size_t) rows * ns * c->n_exp * 4);
    if (val_sel) memcpy(val_sel, e->route_val_sel + r0 * ns * c->n_vexp, (size_t) rows * ns * c->n_vexp * 4);
    return 0;
}
int eng_mova_timing(Eng* e, int on) {
    if (on && !e->csb) {
        id<MTLCounterSet> ts = nil;
        for (id<MTLCounterSet> s in e->dev.counterSets) if ([s.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = s;
        if (!ts) return -1;
        MTLCounterSampleBufferDescriptor* sd = [MTLCounterSampleBufferDescriptor new];
        sd.counterSet = ts;
        sd.storageMode = MTLStorageModeShared;
        sd.sampleCount = 2 * MAX_ENC;
        NSError* err = nil;
        e->csb = [e->dev newCounterSampleBufferWithDescriptor:sd error:&err];
        if (!e->csb) return -1;
    }
    e->timing_on = on;
    if (on) memset(e->tg_sec, 0, sizeof e->tg_sec);
    return 0;
}
int eng_mova_timing_read(Eng* e, double* seconds) {
    memcpy(seconds, e->tg_sec, sizeof e->tg_sec);
    return 0;
}
