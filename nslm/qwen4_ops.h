// nslm/qwen4_ops.h - the qwen4_exp operations of one token: the reference forward is built from them, the GPU kernels
// are tested against them.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "model_st.h"
#include "ple.h"
#include "qwen4_cfg.h"
#include "rows.h"

// Qwen4 RMSNorm (weights centred at 0): each group of g values of x[len] times rsqrt(mean square + eps) * (1 + w)
void qwen4_norm1p(const float* x, int len, int g, const float* w, float eps, float* y);

typedef struct {
    const float *conv_w;    // [conv_dim][K], conv_dim = 2 * hk * dk + hv * dv
    const float *A_log, *dt_bias;   // [hv]
    const float *norm_w;    // [dv], a plain RMSNorm weight
} Qwen4GdnW;
// Gated DeltaNet (delta rule) per value head: S = g S; S += beta (v - S k) k^T; y = S q, with g = exp(-exp(A_log)
// softplus(a + dt_bias)), beta = sigmoid(b).  A layer's state: conv[(K - 1)][conv_dim] (oldest first), S[hv][dk][dv].
size_t qwen4_gdn_conv_len(const Qwen4Cfg* c);
size_t qwen4_gdn_state_len(const Qwen4Cfg* c);
// One token: qkv[conv_dim], z[hv * dv], a[hv], b[hv] (the projections) -> y[hv * dv] (normed and gated by sigmoid(z));
// conv and S are updated
void qwen4_gdn_token(const Qwen4Cfg* c, const Qwen4GdnW* w, const float* qkv, const float* z, const float* a, const float* b,
                     float* conv, float* S, float* y);

// Hyper-connections around the low-rank maps: lo = silu(down(normed) / hc), mixed = mean_s sigmoid(up(lo)) normed,
// inj = 2 sigmoid(inject(normed) / hc), H_s += inj_s branch
void qwen4_hc_act(const Qwen4Cfg* c, float* lo);
void qwen4_hc_mix(const Qwen4Cfg* c, const float* normed, const float* up, float* mixed);
void qwen4_hc_gate(const Qwen4Cfg* c, float* inj);
void qwen4_hc_add(const Qwen4Cfg* c, float* H, const float* branch, const float* inj);

// RMSNorm (1 + w over head_dim values) of nh heads at x + h * hs, then RoPE at pos on each head's first rot_dim values
void qwen4_head_norm_rope(const Qwen4Cfg* c, float* x, int nh, int hs, int hd, const float* w, int pos);
// MoE router: softmax of logits[n_exp], the top k (ties to the higher expert), weights renormalised over the k
void qwen4_route(const Qwen4Cfg* c, const float* logits, int32_t* ind, float* wt);

// PLE of token t: emb[ple_dim], the rows of its n-grams from table (a GGUF table, e.g. Unsloth's), or when NULL from the
// folder's table shards of `layer`; -1 when a row is missing
int qwen4_ple_embed(const NsModel* m, const NsRows* rw, const PleTable* table, const Qwen4Cfg* c, const Qwen4Ngram* g, int layer,
                    const int32_t* ids, int t, float* emb);
// gated[s] = sigmoid(signed sqrt(keys_s . qs_s / sqrt(d))) vals, for each stream s of the normed keys and queries
void qwen4_ple_gate(const Qwen4Cfg* c, const float* keys, const float* qs, const float* vals, float* gated);
// H += gated + silu(causal conv of cin: kernel ple_conv, dilation ngram); hist (the last inputs, oldest first) updated
size_t qwen4_ple_hist_len(const Qwen4Cfg* c);
void qwen4_ple_conv(const Qwen4Cfg* c, const float* w, const float* cin, const float* gated, float* hist, float* H);

// QSA block j's key: the mean of its raw keys raw[ix_ratio][ix_dim], RMSNorm (1 + w), RoPE at j * ix_ratio
void qwen4_qsa_block(const Qwen4Cfg* c, const float* raw, const float* w, int j, float* out);
// The budget / ratio best of `done` blocks for query iq[ix_heads][ix_dim] by sum_h relu(iq_h . blk_j), ties to the
// higher block, into sel; their count, or 0 (all keys) while done is not above that
int qwen4_qsa_select(const Qwen4Cfg* c, const float* iq, const float* blk, int done, int32_t* sel);
// Attention of the query at pos, q[n_head][2 * head_dim] (query, gate), over K, V [pos + 1][n_kv][head_dim]: keys 0 .. pos
// when nsel is 0, else the nsel blocks and the incomplete tail; o[n_head][head_dim], gated by sigmoid
void qwen4_attend(const Qwen4Cfg* c, const float* q, const float* K, const float* V, int pos, const int32_t* sel, int nsel,
                  float* o);

