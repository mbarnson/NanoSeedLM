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
// per slice, e = exponent codes (4 bits per block, two blocks per byte, low first); SEED6P8 as SEED4P4 with s = 32-bit
// coefficient words.
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
// 1 when the backend's kernels take weights in fmt (MF_*)
int kt_has_fmt(int fmt);

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
// The same with W stored transposed, BF16 [H][I][O] (k_mm, prompt rows): y[t][h][o] = bf16(sum_i W_h[i][o] x_th[i]);
// 1 where the backend lacks it
int kt_heads_mm_t(int H, int O, int I, const uint16_t* W, const float* x, int xs, int hs, float* y, int T);
// Per-head maps with W in fmt MF_BF16 / MF_Q8 / MF_Q4 (streams as nslm/model_st.h: codes, then a BF16 scale and bias
// per 64 values; each head's O x I a whole number of groups): kt_heads_mv's maps (tr 0: W [H][O][I]; T <= 8 the decode
// matvec, else the prompt GEMMs) or kt_heads_mm_t's (tr 1: W [H][I][O], T > 8; no gate).  1 where a backend lacks them.
int kt_heads_q(int fmt, int tr, int H, int O, int I, const void* codes, const uint16_t* scales, const uint16_t* biases,
               const float* x, int xs, int hs, const float* g, float* y, int T);
// The same with W in SEED4P4 (nslm/search4.h): seeds and coefficients [H][rows][cols / 8], an exponent bias per head,
// exponent codes [H][rows][cols / 8] as nibbles (low first).  1 where a backend lacks it.
int kt_heads_seed(int tr, int H, int O, int I, const uint16_t* seeds, const uint16_t* coefs, const int32_t* ebias,
                  const uint8_t* ecodes, const float* x, int xs, int hs, const float* g, float* y, int T);
// rope of the query RoPE parts qr [T][n_head][128] (in place) and of kr [T][128] into Kc [npos][128], and the latent
// c [T][r] into Vc [npos][r], at the rows' positions (k_mla_rope); the caches are updated
int kt_mla_rope(MlaArgs a, float* qr, const float* kr, const float* c, uint16_t* Kc, uint16_t* Vc, int npos,
                const RowInfo* ri, const float* inv, int T);
// latent attention: olat[t][h] = bf16(sum_p softmax_p(scale (ql_th . c_p + qr_th . k_p)) c_p) over p <= pos_t, from
// ql [T][n_head][r], qr [T][n_head][128] (rotated) and the caches; split-key + reduce when a.n_splits > 1
int kt_mla_attn(MlaArgs a, const float* ql, const float* qr, const uint16_t* Kc, const uint16_t* Vc, int npos,
                const RowInfo* ri, float* olat, int T);
// MLA prompt rows without absorption (CUDA; 1 where a backend has no such path): keys 0 .. pos of T consecutive rows
// decompressed per head (Kn = bf16(q_lat_h^T c), Vd = bf16(v_up_h c)) dec_keys at a time, then o[t][h] =
// bf16(bf16(sum_p softmax_p(scale (q_th . Kn_ph + qr_th . k_p)) Vd_ph) * bf16(softplus_ln2(g))), from q, qr [T][n_head][128],
// the caches and Wql [n_head][r][128], Wvu [n_head][128][r] (BF16); Kn, Vd ([pos + 1][n_head][128]) receive the
// decompressed keys and values
int kt_mla_prefill(MlaArgs a, const float* q, const float* qr, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
                   const uint16_t* Wql, const uint16_t* Wvu, const float* g, float* o, int T, int dec_keys, uint16_t* Kn, uint16_t* Vd);
// kt_mla_prefill's decompression alone, each map in its own format (MF_BF16 / MF_Q8 / MF_Q4, streams as kt_heads_q's;
// q_lat [n_head][r][128] grouped along 128, v_up [n_head][128][r] along r): Kn, Vd [n][n_head][128] from the latent rows
// c [n][r] (BF16).  1 where a backend has no such path.
int kt_mla_decomp_q(MlaArgs a, int qfmt, const void* qc, const uint16_t* qs, const uint16_t* qb, int vfmt, const void* vc,
                    const uint16_t* vs, const uint16_t* vb, const uint16_t* c, int n, uint16_t* Kn, uint16_t* Vd);
// MLA prompt attention over keys and values already expanded per head (Metal's k_mla_prefill): o[t][h*128+d] = bf16(bf16(sum_p softmax_p(scale (qn_th . kn_ph
// + qr_th . kr_p)) vn_ph[d]) * bf16(softplus_ln2(g))) over p <= pos_t, from qn [T][n_head*128] (the heads' queries),
// qr [T][n_head][128] (rotated), kn / vn [npos][n_head][128] and the RoPE keys Kc [npos][128] (BF16, at the rows'
// kv0); the keys in nb passes [kb[i], kb[i + 1]) carrying the softmax state.  Returns 1 where the backend lacks it.
int kt_mla_attn_x(int n_head, float scale, const int* kb, int nb, const float* qn, const float* qr, const float* kn,
                  const float* vn, const uint16_t* Kc, int npos, const RowInfo* ri, const float* g, float* o, int T);
// MLA caches in FP8 / FP4 (fmt ENG_KV_FP8 / ENG_KV_FP4, nslm/kvq.h): per position the RoPE key's codes k (128 or 64
// bytes) and block scales ks (4 or 8), the latent's v (r or r / 2) and vs (r / 32 or r / 16).  1 where a backend lacks them.
typedef struct {
    uint8_t *k, *v, *ks, *vs;
} KtKvMla;
// kt_mla_rope with the cache written in fmt
int kt_mla_rope_q(MlaArgs a, int fmt, float* qr, const float* kr, const float* c, KtKvMla kv, int npos, const RowInfo* ri,
                  const float* inv, int T);
// kt_mla_attn over a cache in fmt
int kt_mla_attn_q(MlaArgs a, int fmt, const float* ql, const float* qr, KtKvMla kv, int npos, const RowInfo* ri, float* olat, int T);
// kt_mla_attn_x with the RoPE keys in fmt (codes Kq, scales Ks)
int kt_mla_attn_xq(int fmt, int n_head, float scale, const int* kb, int nb, const float* qn, const float* qr, const float* kn,
                   const float* vn, const uint8_t* Kq, const uint8_t* Ks, int npos, const RowInfo* ri, const float* g, float* o, int T);
// y[row][d] = value d of `rows` cache rows of len values in fmt (k_kv_f32)
int kt_kv_f32(int fmt, const uint8_t* codes, const uint8_t* scales, int len, int rows, float* y);
// fmt MF_BF16 (E) or MF_Q8 (q8, s8, b8): x[t] = row ids[t]
int kt_embed(int fmt, const uint16_t* E, const uint32_t* q8, const uint16_t* s8, const uint16_t* b8, int V, int d,
             const int32_t* ids, int n, float* x);
// x[t] = row ids[t] of a SEED4P4 / SEED6P8 embedding (slice 0 of w); 1 where the backend lacks it
int kt_embed_seed(const KtWeight* w, const int32_t* ids, int n, float* x);
int kt_argmax(const float* logits, int V, int n, int32_t* out);
// y[i] = x[i] (BF16 to f32, k_bf16_f32); 1 where the backend lacks it
int kt_bf16_f32(const uint16_t* x, float* y, int n);
// X^T X (k_xtx, the MLA capture for GPTQ): H[b][i][j] += sum_t x[t * xs + b * D + i] x[t * xs + b * D + j] over T rows, for
// nb blocks of D columns (D a multiple of 32), on H's lower 32 x 32 tiles (i / 32 >= j / 32); the rest of H unchanged.
// 1 where a backend lacks it.
int kt_xtx(const float* x, int T, int xs, int D, int nb, float* H);

#ifdef __cplusplus
}
#endif
