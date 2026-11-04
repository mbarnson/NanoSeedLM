// nslm/qwen4_cfg.h - Qwen3.8-Flash-Next (qwen4_exp): the validated text configuration, and the PLE n-gram row ids.
//
// qwen4_cfg_parse accepts only what the reference implements: text input (MRoPE with equal positions is plain partial
// RoPE), silu experts, sigmoid output gates, one indexer key head, untied head, top-k renormalisation.
//
// PLE n-grams.  A PLE layer adds rows of a large embedding table chosen by a hash of the last `ngram` tokens.  Head h of
// the n-gram of length n (n = 2 .. ngram, heads_per_ngram heads each) reads row
//     off[h] + (XOR over p < n of tok[t - p] * mult[p]) mod size[h],
// where size[h] is a prime, mult[p] an odd multiplier from splitmix64, and tok[t - p] is EOS once the walk back from t
// passes an EOS (a document starts a new context).  The rows of all heads, side by side, form the layer's embedding.
#pragma once
#include <stdint.h>

#define QWEN4_MAX_LAYERS 128
#define QWEN4_MAX_NGRAM 8
#define QWEN4_MAX_PLE_HEADS 64

typedef struct {
    int n_layer, d, vocab, eos;
    int hc, hc_rank;                                  // hyper-connection streams, low rank of their mixing
    int n_head, n_kv, head_dim, rot_dim;              // attention: RoPE on the first rot_dim of each head
    float rope_theta, eps;
    int ix_heads, ix_dim, ix_budget, ix_ratio;        // QSA indexer: query heads, head width, token budget, block size
    int lin_hk, lin_hv, lin_dk, lin_dv, lin_conv;     // Gated DeltaNet heads, head widths, conv kernel
    int n_exp, top_k, ff_exp, ff_shared;
    int ple_dim, ple_conv, ngram, heads_per_ngram, ngram_parts;
    int64_t ngram_base, ngram_div, seed;
    uint8_t attn[QWEN4_MAX_LAYERS];                   // 1: a QSA attention layer, 0: a Gated DeltaNet layer
    int8_t ple[QWEN4_MAX_LAYERS];                     // the PLE index of the layer (0, 1, ..), or -1
} Qwen4Cfg;

int qwen4_cfg_parse(Qwen4Cfg* c, const char* json, char* err, int errlen);   // config.json text (text_config or flat)
int qwen4_cfg_load(Qwen4Cfg* c, const char* model_dir, char* err, int errlen);

typedef struct {
    int heads;                                        // (ngram - 1) * heads_per_ngram
    int64_t mult[QWEN4_MAX_NGRAM], size[QWEN4_MAX_PLE_HEADS], off[QWEN4_MAX_PLE_HEADS];
    int64_t rows;                                     // the table's rows, padded to a multiple of ngram_div
} Qwen4Ngram;

void qwen4_ngram_init(const Qwen4Cfg* c, int ple_index, Qwen4Ngram* g);
// The rows of token t of ids[0..t] (the context before ids[0] is EOS): rows[g->heads].
void qwen4_ngram_rows(const Qwen4Cfg* c, const Qwen4Ngram* g, const int32_t* ids, int t, int64_t* rows);
