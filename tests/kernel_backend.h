// tests/kernel_backend.h - the GPU side of tests/test_mova_kernels.c: one call per engine kernel (or fused group of
// kernels), on host arrays.  The backend uploads the inputs, runs the kernel the engine runs and copies the outputs back.
// Implementations: tests/kernel_backend_cuda.c (engine/kernels_moe.cu) and tests/kernel_backend_metal.m
// (engine/kernels_moe.metal).  Every call returns 0, or -1 after printing what failed.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "kernels_moe.metal"   // MF_*, MvArgs, MmArgs, MmTile, RouterArgs, AttnArgs, RowInfo, MV_MAXT

#ifdef __cplusplus
extern "C" {
#endif

// A stacked weight: S slices of R x K in the model's streams, slice after slice in each stream (nslm/model_st.h):
// BF16 w = values; Q8 / Q4 w = packed words, s = scales, b = biases (BF16, one per 64 values); SEED4 w = seeds,
// s = nibble words, b = int32 exponent bias per slice; SEED4P4 w = seeds, s = coefficient words, b = int32 exponent bias
// per slice, e = exponent codes (4 bits per block, two blocks per byte, low first).
typedef struct {
    int fmt, S, R, K;
    const void *w, *s, *b, *e;
    size_t wn, sn, bn, en;   // bytes
} KtWeight;

// 0, or -1 with err ("no CUDA device" / "no Metal device" when there is no GPU: the test is then skipped)
int kt_open(char* err, int errlen);
void kt_close(void);
const char* kt_name(void);
int kt_mm_tile(void);   // pairs per tile of the grouped GEMM (MmTile.count <= this)
int kt_attn_key_tile(void);   // keys per tile of prefill attention (its online softmax rescales per tile)
// The GEMM's seed weights: exact (f32) when on; returns whether the backend also has a BF16-rounded mode (off)
int kt_seed_gemm_exact(int on);

// y[t][r] (+)= W_0[r] . x[t] for t < T <= MV_MAXT (slice 0; add: y holds the residual)
int kt_mv(const KtWeight* w, int T, int add, const float* x, float* y);
// y[p][r] = W_sel[p][r] . x[p / xdiv]
int kt_mv_sel(const KtWeight* w, int P, int xdiv, const int32_t* sel, const float* x, float* y);
// a[p][r] = bf16(silu(bf16 g) * bf16 u): gate slice sel[p], up slice sel[p] + 1
int kt_mv_gu(const KtWeight* w, int P, int xdiv, const int32_t* sel, const float* x, float* a);
// GEMM, dense on slice 0 with the residual: y[t][r] = bf16(y[t][r] + bf16(W_0[r] . x[t]))
int kt_mm(const KtWeight* w, int T, const float* x, float* y);
// GEMM, grouped: the tiles' pairs perm[start ..), y[pair][r] = W_slice[r] . x[pair / xdiv]
int kt_mm_grouped(const KtWeight* w, int P, int xdiv, const int32_t* perm, const MmTile* tiles, int nt, const float* x,
                  float* y);

int kt_gnorm(int d, float eps, const float* x, const uint16_t* w, float* y, int T);
int kt_router(RouterArgs a, const uint16_t* W, const uint16_t* bias, const float* x, int T, int32_t* ind, float* wt);
int kt_router_topk(RouterArgs a, const float* score, const float* sel, int T, int32_t* ind, float* wt);
int kt_swiglu(const float* g, const float* u, float* a, int n);
// x[t] += bf16(sum_j bf16(D[t][j] w[t][j]) + shared[t]) (k_moe_combine); v[t] = sum_j bf16(silu(D) w) (k_vcombine)
int kt_combine(const float* D, const float* w, const float* shared, float* x, float* v, int d, int k, int T);
// rope of q (in place) and of k into the BF16 caches at the rows' positions (with v), then decode attention + gate
int kt_rope_attn(AttnArgs a, float* q, const float* k, const float* v, uint16_t* Kc, uint16_t* Vc, int npos,
                 const RowInfo* ri, const float* inv, const float* g, float* o, int T);
// decode (split-key) attention + reduce with the gate; BF16 caches of npos positions x n_kv heads x 128
int kt_attn(AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
            const float* g, float* o, int T);
// prefill attention (causal, gate fused)
int kt_attn_prefill(AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
                    const float* g, float* o, int T);
// The 8-bit KV cache (EngOpts.kv_format = ENG_KV_Q8): npos positions x n_kv heads x 128 int8 values and one f32 scale
// per (position, head): a value is q * scale, which the attention kernels read rounded to BF16.
typedef struct {
    int8_t *k, *v;
    float *ks, *vs;
} KtKvQ8;
// rope of q (in place) and of k, and v, quantized into the 8-bit caches at the rows' positions (scale = max |x| / 127 of
// the head's BF16 values); the caches are updated
int kt_rope_kv_q8(AttnArgs a, float* q, const float* k, const float* v, KtKvQ8 kv, int npos, const RowInfo* ri,
                  const float* inv, int T);
// decode and prefill attention (+ gate) on 8-bit caches
int kt_attn_q8(AttnArgs a, const float* q, KtKvQ8 kv, int npos, const RowInfo* ri, const float* g, float* o, int T);
int kt_attn_prefill_q8(AttnArgs a, const float* q, KtKvQ8 kv, int npos, const RowInfo* ri, const float* g, float* o, int T);
// MLA (a TransMLA conversion; nslm/mova_cfg.h).  Per-head maps (k_heads_mv): y[t][h][o] = bf16(W_h[o] . x_th) with W
// BF16 [H][O][I] and x_th = x + t * xs + h * hs (floats); with g (same layout as y): bf16(that * bf16(softplus_ln2(g))).
int kt_heads_mv(int H, int O, int I, const uint16_t* W, const float* x, int xs, int hs, const float* g, float* y, int T);
// rope of the query RoPE parts qr [T][n_head][128] (in place) and of kr [T][128] into Kc [npos][128], and the latent
// c [T][r] into Vc [npos][r], at the rows' positions (k_mla_rope); the caches are updated
int kt_mla_rope(MlaArgs a, float* qr, const float* kr, const float* c, uint16_t* Kc, uint16_t* Vc, int npos,
                const RowInfo* ri, const float* inv, int T);
// latent attention: olat[t][h] = bf16(sum_p softmax_p(scale (ql_th . c_p + qr_th . k_p)) c_p) over p <= pos_t, from
// ql [T][n_head][r], qr [T][n_head][128] (rotated) and the caches; split-key + reduce when a.n_splits > 1
int kt_mla_attn(MlaArgs a, const float* ql, const float* qr, const uint16_t* Kc, const uint16_t* Vc, int npos,
                const RowInfo* ri, float* olat, int T);
// fmt MF_BF16 (E) or MF_Q8 (q8, s8, b8): x[t] = row ids[t]
int kt_embed(int fmt, const uint16_t* E, const uint32_t* q8, const uint16_t* s8, const uint16_t* b8, int V, int d,
             const int32_t* ids, int n, float* x);
int kt_argmax(const float* logits, int V, int n, int32_t* out);

#ifdef __cplusplus
}
#endif
