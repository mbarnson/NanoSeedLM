// engine/kernels_cuda.h - the C interface of the CUDA kernels (engine/kernels_moe.cu) for the CUDA engine
// (engine/mova_cuda.c).  The argument structs and constants are kernels_moe.metal's, shared by both backends.
//
// Weights are addressed per slice: a WSlice holds the device addresses of one slice's streams (nslm/model_st.h order:
// BF16 0 values; Q8/Q4 0 packed words, 1 scales, 2 biases; SEED4 0 seeds, 1 nibble words; SEED4P4 0 seeds,
// 1 coefficient words, 3 exponent codes) and the slice's exponent bias.  A dense tensor is one WSlice; stacked experts
// are a device array of them, so each expert's streams can live anywhere the GPU can read: VRAM or mapped host memory.
#pragma once
#include <stdint.h>

#include <cuda_runtime_api.h>

#include "kernels_moe.metal"   // MF_*, MvArgs, MmArgs, MmTile, RouterArgs, AttnArgs, RowInfo, tile constants

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const void* p[4];
    int32_t eb;        // seeds: the slice's exponent bias
    int32_t pad[3];
} WSlice;

// ---- the expert cache: a pool of fixed-size VRAM slots over a host arena that holds every expert ----------------------
// Unit u (an MLP expert's gate, up and down slices, or one value expert) occupies bytes [u * unit_bytes, + unit_bytes)
// of the host arena and, when resident, the same layout in slot unit_slot[u] of the VRAM pool.  The admit kernel runs
// after a router: it marks the selected units used, gives every missing one the least recently used slot that this
// call does not need, points the unit's slice tables (tab) at the slot, and queues a copy job; the copy kernel then
// brings the queued units over PCIe.  The kernels that follow read only VRAM.
typedef struct {
    int32_t* unit_slot;     // [units]: slot, or -1
    int32_t* slot_unit;     // [slots]: unit, or -1
    uint32_t* slot_last;    // [slots]: the admit tick of the last use
    uint32_t* tick;         // [4]: the admit tick (from CACHE_TICK_BASE), unused
    uint32_t* stats;        // [2]: hits, misses (a prefetching view of a pool counts apart)
    int32_t* jobs;          // [2 * per_layer]: (unit, slot) pairs queued by the last admit
    int32_t* njobs;         // [1]
    uint8_t* vram;          // the pool
    const uint8_t* host;    // device address of the host arena
    uint64_t unit_bytes;
    int32_t slots, per_layer, ntens;
    WSlice* tab;            // [sparse layer][ntens][per_layer]
    uint32_t off[3][4];     // byte offset of tensor i's stream s in a unit (0xFFFFFFFF: no stream)
    uint64_t hits, misses;  // host-side statistics (not used by the kernels)
} CachePool;
#define CACHE_TICK_BASE 0x40000000u
// rows x k selections inds[] of sparse layer sl.  flags: CACHE_PROTECT_PREV (never replace the units of the pool's
// previous admit: their layer may still be computing beside a prefetch).
enum { CACHE_PROTECT_PREV = 2 };
void kc_cache_admit(cudaStream_t s, const CachePool* p, int sl, const int32_t* inds, int count, int flags);
// blocks: the copy's grid (fewer leave SMs to kernels running beside it on another stream)
void kc_cache_copy(cudaStream_t s, const CachePool* p, int blocks);

// ---- the KV cache: per layer, positions [0, nv) in segment a and [nv, ...) in segment b (row 0 of b = position nv) ----
// Rows hold n_kv heads of 128 values: BF16, or Q8 (int8 with one f32 scale per row and head: x = q * scale, scale =
// max|x| / 127).  Segments live in VRAM or mapped host memory; a prompt stages b in VRAM for the layer it computes.
enum { KV_BF16 = 0, KV_Q8 = 1 };
typedef struct {
    void* k;
    void* v;
    float* ks;   // Q8: [rows][n_kv]
    float* vs;
} KvSeg;
typedef struct {
    KvSeg a, b;
    int32_t nv, fmt;
} KvView;

// x[t][:] = the embedding row of ids[t] (BF16 or Q8)
void kc_embed(cudaStream_t s, int fmt, WSlice w, int d, const int32_t* ids, float* x, int T);
// grouped RMSNorm (2 groups)
void kc_gnorm(cudaStream_t s, int d, float eps, const float* x, const uint16_t* w, float* y, int T);
// Dense matvec, T <= MV_MAXT tokens: y[t][r] (+)= W[r] . x[t]
void kc_mv(cudaStream_t s, int fmt, WSlice w, int K, int R, const float* x, int xs, float* y, int ys, int T, int add,
           const uint32_t* G);
// Gather matvec: P (token, expert) pairs, pair p uses slice sel[p] and input row p / xdiv: y[p][r] = W_sel[r] . x
void kc_mv_sel(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
               const int32_t* sel, int P, int xdiv, const uint32_t* G);
// Routed experts' gate + up + SwiGLU in one pass over the selected pairs: a[p][r]
void kc_mv_gu(cudaStream_t s, int fmt, const WSlice* wg, const WSlice* wu, int K, int R, const float* x, int xs, float* a,
              int ys, const int32_t* sel, int P, int xdiv, const uint32_t* G);
// GEMM: dense (T tokens) or grouped (ntiles entries of the tile table; pairs perm[])
// prefill GEMM: seed weights exact in f32 (three BF16 terms) instead of rounded to BF16 (see kernels_moe.cu)
void kc_seed_gemm_f32(int on);
void kc_mm(cudaStream_t s, int fmt, WSlice w, int K, int R, const float* x, int xs, float* y, int ys, int T, int add,
           const uint32_t* G);
void kc_mm_grouped(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
                   int xdiv, const int32_t* perm, const MmTile* tiles, int ntiles, const uint32_t* G);
// Grouping on the GPU: the count = rows x k selections inds[] (n experts) -> perm (pair ids by expert, ascending within
// an expert), tiles (runs of <= MMT_BN pairs of one expert) and *ntiles.  kc_mm_grouped_dev takes the tile count from
// the device (grid = the upper bound, extra blocks exit).
void kc_bucket(cudaStream_t s, const int32_t* inds, int count, int n, int32_t* perm, MmTile* tiles, int32_t* ntiles);
void kc_mm_grouped_dev(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
                       int xdiv, const int32_t* perm, const MmTile* tiles, const int32_t* ntiles, int max_tiles,
                       const uint32_t* G);
#define MMT_BM 64   // tensor-core GEMM: weight rows per block
#define MMT_BN 128   // tokens (or pairs) per block
#define MMT_THREADS (2 * MMT_BN)   // threads per block: (BM / 32) x (BN / 32) warps of 32 x 32
#define MMT_BK 32
// Router: sigmoid scores, selection scores (score + bias), top-k ids and weights
void kc_router(cudaStream_t s, RouterArgs a, const uint16_t* W, const uint16_t* bias, const float* x, float* score,
               float* sel, int32_t* inds, float* wts, int T);
// the top-k step alone, from scores and selection scores (tests/test_mova_kernels.c)
void kc_router_topk(cudaStream_t s, RouterArgs a, const float* score, const float* sel, int32_t* inds, float* wts, int T);
void kc_swiglu(cudaStream_t s, const float* g, const float* u, float* a, int n);
void kc_moe_combine(cudaStream_t s, const float* D, const float* w, const float* shared, float* x, int d, int k, int T);
void kc_vcombine(cudaStream_t s, const float* V, const float* w, float* v, int d, int k, int T);
void kc_rope_kv(cudaStream_t s, float* q, const float* k, const float* v, KvView kv, const RowInfo* ri, const float* inv,
                int n_head, int n_kv, int T);
// split-key attention (rows <= MV_MAXT, or any T) + reduce with the softplus output gate
void kc_attn(cudaStream_t s, AttnArgs a, const float* q, KvView kv, const RowInfo* ri, float* part, const float* g, float* o,
             int T);
// prefill attention: causal, tiles of FA_BK keys shared by a KV head's query heads (the online softmax rescales per
// tile, so the tile size is a rounding point: tests/test_mova_kernels.c), the gate fused
#define FA_BK 64
void kc_attn_prefill(cudaStream_t s, AttnArgs a, const float* q, KvView kv, const RowInfo* ri, const float* g, float* o, int T);
// MLA (TransMLA; kernels_moe.metal MlaArgs, HmvArgs).  The KvView of an MLA layer (BF16): k the RoPE key (128 per
// position), v the latent (a.r per position).
// per-head maps: y[t][h][o] = bf16(W_h[o] . x[t * a.xs + h * a.hs ..]) (W BF16 [H][O][I], I a multiple of 32); with g
// (y's layout): bf16(that * bf16(softplus_ln2(g)))
void kc_heads_mv(cudaStream_t s, HmvArgs a, const uint16_t* W, const float* x, const float* g, float* y, int T);
// RoPE of the query RoPE parts qr [T][n_head][128] (in place) and of kr [T][128] into kv.k; the latent c [T][r] into kv.v
void kc_mla_rope(cudaStream_t s, MlaArgs a, float* qr, const float* kr, const float* c, KvView kv, const RowInfo* ri,
                 const float* inv, int T);
// latent attention into olat [T][n_head][r]: a.n_splits > 1 splits the keys (partials in part: T x n_head x n_splits x
// (r + 2) floats) and reduces; 1 is one pass
void kc_mla_attn(cudaStream_t s, MlaArgs a, const float* ql, const float* qr, KvView kv, const RowInfo* ri, float* part,
                 float* olat, int T);
void kc_argmax(cudaStream_t s, const float* logits, int32_t* out, int V, int n);

#ifdef __cplusplus
}
#endif
