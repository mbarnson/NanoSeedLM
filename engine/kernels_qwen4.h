// engine/kernels_qwen4.h - the CUDA kernels of qwen4 (engine/kernels_qwen4.cu), counterparts of nslm/qwen4_ops.h.
// f32 device arrays, T tokens of consecutive rows.
#pragma once
#include <cuda_runtime_api.h>

#include "qwen4_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *conv_w, *A_log, *dt_bias, *norm_w;   // as Qwen4GdnW, in f32
} Q4GdnDev;
// qwen4_gdn_token for T tokens in order (conv and S updated; cv: T x conv_dim scratch).  -1 unless dk is 32, 64 or
// 128, dv a multiple of 32 up to 1024 and the conv kernel at most 8.
int kq4_gdn(cudaStream_t s, const Qwen4Cfg* c, Q4GdnDev w, const float* qkv, const float* z, const float* a, const float* b,
            float* conv, float* S, float* cv, float* y, int T);
// y[t] = qwen4_norm1p(x[t], len, g, w, eps) for t < T (y may be x)
void kq4_norm1p(cudaStream_t s, const float* x, int len, int g, const float* w, float eps, float* y, int T);
// the hyper-connection pointwise steps of qwen4_ops.h for T tokens: lo[T][rank], up and normed [T][hc * d], mixed [T][d],
// inj [T][hc], H [T][hc * d], branch [T][d]
void kq4_hc_act(cudaStream_t s, int hc, float* lo, int n);
void kq4_hc_mix(cudaStream_t s, int hc, int d, const float* normed, const float* up, float* mixed, int T);
void kq4_hc_gate(cudaStream_t s, int hc, float* inj, int n);
void kq4_hc_add(cudaStream_t s, int hc, int d, float* H, const float* branch, const float* inj, int T);

// H[t][s][i] = x[t][i] for each of hc streams
void kq4_tile(cudaStream_t s, int hc, int d, const float* x, float* H, int T);
// qwen4_head_norm_rope on row t of x (row stride rs) at position pos0 + t: nh heads of hd at hs apart; w [hd]
void kq4_head_norm_rope(cudaStream_t s, const Qwen4Cfg* c, float* x, int rs, int nh, int hs, int hd, const float* w, int pos0, int T);
// qwen4_route of logits[T][n_exp] into ind, wt [T][top_k] (top_k <= 32)
void kq4_route(cudaStream_t s, const Qwen4Cfg* c, const float* logits, int32_t* ind, float* wt, int T);
// out[t] = sum_k wt[t][k] D[t * top_k + k] + sigmoid(sg[t]) sh[t]
void kq4_moe_combine(cudaStream_t s, int d, int k, const float* D, const float* wt, const float* sh, const float* sg, float* out, int T);
// qwen4_ple_gate and qwen4_ple_conv for T tokens in order (hist updated)
void kq4_ple_gate(cudaStream_t s, const Qwen4Cfg* c, const float* keys, const float* qs, const float* vals, float* gated, int T);
void kq4_ple_conv(cudaStream_t s, const Qwen4Cfg* c, const float* w, const float* cin, const float* gated, float* hist, float* H, int T);
// QSA: block keys j0 .. j1 - 1 (qwen4_qsa_block) from the raw key cache raw[pos][ix_dim] into blk[j][ix_dim]
void kq4_qsa_blocks(cudaStream_t s, const Qwen4Cfg* c, const float* raw, const float* w, int j0, int j1, float* blk);
// qwen4_qsa_select for row t (iq[t][ix_heads][ix_dim] at row stride rs, done[t] blocks) into sel[t][budget / ratio];
// sc is scratch of T x maxdone floats.  Rows with done[t] <= budget / ratio are left as they are.
void kq4_qsa_select(cudaStream_t s, const Qwen4Cfg* c, const float* iq, int rs, const float* blk, const int32_t* done, int maxdone,
                    float* sc, int32_t* sel, int T);
// qwen4_attend for row t at position pos0 + t: q[t][n_head][2 head_dim], caches K, V [pos][n_kv][head_dim], nsel[t] blocks
// of sel[t][budget / ratio]; o[t][n_head][head_dim].  -1 unless head_dim is a multiple of 32, at most 256.
int kq4_attend(cudaStream_t s, const Qwen4Cfg* c, const float* q, const float* K, const float* V, int pos0, const int32_t* sel,
               const int32_t* nsel, float* o, int T);

#ifdef __cplusplus
}
#endif
