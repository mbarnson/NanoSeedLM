// engine/kernels_qwen4.h - the CUDA kernels of Qwen3.8 (engine/kernels_qwen4.cu), counterparts of nslm/qwen4_ops.h.
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
// T tokens in order through a Gated DeltaNet layer (qwen4_gdn_token): qkv[T][conv_dim], z[T][hv * dv], a and b [T][hv]
// -> y[T][hv * dv].  conv and S (layouts of qwen4_ops.h) are updated; cv is scratch of T x conv_dim floats.
// -1 when the head widths are not supported (dk 32, 64 or 128; dv a multiple of 32, at most 1024; conv kernel <= 8).
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

#ifdef __cplusplus
}
#endif
