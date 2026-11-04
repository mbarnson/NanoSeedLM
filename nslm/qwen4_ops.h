// nslm/qwen4_ops.h - the qwen4_exp operations of one token, in C: the reference forward (nslm/qwen4_ref.h) is built from
// them, and the GPU kernels are tested against them.  f32 values, sums in double.
//
// Gated DeltaNet keeps, for each value head, a matrix state S that maps keys to values.  Each token decays S by g, then
// corrects it toward the token's value along its key (the delta rule), and reads it with the query:
//     S = g S,   S += beta (v - S k) k^T,   y = S q
// with g = exp(-exp(A_log) softplus(a + dt_bias)) and beta = sigmoid(b) per head.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "qwen4_cfg.h"

// Qwen4 RMSNorm (weights centred at 0): each group of g values of x[len] times rsqrt(mean square + eps) * (1 + w)
void qwen4_norm1p(const float* x, int len, int g, const float* w, float eps, float* y);

typedef struct {
    const float *conv_w;    // [conv_dim][K], conv_dim = 2 * hk * dk + hv * dv
    const float *A_log, *dt_bias;   // [hv]
    const float *norm_w;    // [dv], a plain RMSNorm weight
} Qwen4GdnW;
// The state of one layer: conv[(K - 1)][conv_dim] (the last K - 1 inputs, oldest first) and S[hv][dk][dv]
size_t qwen4_gdn_conv_len(const Qwen4Cfg* c);
size_t qwen4_gdn_state_len(const Qwen4Cfg* c);
// One token: qkv[conv_dim], z[hv * dv], a[hv], b[hv] (the projections) -> y[hv * dv] (normed and gated by sigmoid(z));
// conv and S are updated
void qwen4_gdn_token(const Qwen4Cfg* c, const Qwen4GdnW* w, const float* qkv, const float* z, const float* a, const float* b,
                     float* conv, float* S, float* y);

// Hyper-connections of one token, around the low-rank maps (matrix products, not here):
//   normed = qwen4_norm1p(H, hc * d, d, hc_norm)   lo = down(normed)   act: lo = silu(lo / hc)
//   up = up(lo)   mixed = qwen4_hc_mix(normed, up)   inj = block_inject(normed)   gate: inj = 2 sigmoid(inj / hc)
//   H += qwen4_hc_add(branch, inj)
void qwen4_hc_act(const Qwen4Cfg* c, float* lo);
void qwen4_hc_mix(const Qwen4Cfg* c, const float* normed, const float* up, float* mixed);
void qwen4_hc_gate(const Qwen4Cfg* c, float* inj);
void qwen4_hc_add(const Qwen4Cfg* c, float* H, const float* branch, const float* inj);
