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
void kc_mm(cudaStream_t s, int fmt, WSlice w, int K, int R, const float* x, int xs, float* y, int ys, int T, int add,
           const uint32_t* G);
void kc_mm_grouped(cudaStream_t s, int fmt, const WSlice* ws, int K, int R, const float* x, int xs, float* y, int ys,
                   int xdiv, const int32_t* perm, const MmTile* tiles, int ntiles, const uint32_t* G);
// Router: sigmoid scores, selection scores (score + bias), top-k ids and weights
void kc_router(cudaStream_t s, RouterArgs a, const uint16_t* W, const uint16_t* bias, const float* x, float* score,
               float* sel, int32_t* inds, float* wts, int T);
void kc_swiglu(cudaStream_t s, const float* g, const float* u, float* a, int n);
void kc_moe_combine(cudaStream_t s, const float* D, const float* w, const float* shared, float* x, int d, int k, int T);
void kc_vcombine(cudaStream_t s, const float* V, const float* w, float* v, int d, int k, int T);
void kc_rope_kv(cudaStream_t s, float* q, const float* k, const float* v, uint16_t* Kc, uint16_t* Vc, const RowInfo* ri,
                const float* inv, int n_head, int n_kv, int T);
// split-key attention (rows <= MV_MAXT, or any T) + reduce with the softplus output gate
void kc_attn(cudaStream_t s, AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc, const RowInfo* ri,
             float* part, const float* g, float* o, int T);
// prefill attention: causal, tiles of keys shared by a KV head's query heads, the gate fused
void kc_attn_prefill(cudaStream_t s, AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc,
                     const RowInfo* ri, const float* g, float* o, int T);
void kc_argmax(cudaStream_t s, const float* logits, int32_t* out, int V, int n);

#ifdef __cplusplus
}
#endif
