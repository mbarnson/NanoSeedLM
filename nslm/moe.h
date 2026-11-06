// nslm/moe.h - MoVA helpers: sharded-checkpoint index lookup, per-expert calibration statistics, and the
// expanded-expert safetensors files.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "model_st.h"

// The header of nslm-moe's .blk / .blk4 / .blk8 / .blks block files.
typedef struct {
    char magic[8];   // "NSLMBLKM", "NSLMBLK4", "NSLMBLK8", "NSLMBLKS" (S3: C = 12, P = 4), "NSLMBLKT" (trellis)
    uint32_t rows, cols, n_experts;
    uint32_t n_seeds, n_exp, refit;   // .blkt: L, K, T (nslm/trellis.h)
    int32_t exp_delta[3];
    float n0;
    double seconds;
} MoeBlkHeader;

// Stream s (model_st.h order) of an nslm-moe block file of E slices of rows x cols: P = 3 (.blk, SEED4) or P = 4
// (.blk4, SEED4P4, exponent codes low nibble first).  -1 when blk is not that file.
int nslm_blk_stream(const uint8_t* blk, size_t len, int p, int E, int rows, int cols, int s, uint8_t* dst);

// The banks of a folder as tensors language_model.model.layers.L.mlp.switch_mlp.PROJ.weight, mapped for the process:
// S3 seeds as NS_SEEDS3 (nslm-moe --codec s3: L{l}_{gate,up,down}_proj.blks, as .blk4 with blocks of 12, a row's last
// block padded), trellis-coded rows as NS_TCQ (--codec tq: .blkt).  0, 1 with err (no bank), or -1 with err (a wrong file).
int ns_open_bank(NsModel* m, const char* dir, char* err, int errlen);
// The folder's banks added to m (ns_open's), each in place of m's tensor of its name.  0 (also with no bank), or -1 with err.
// The banks stay mapped for the process.
int ns_add_banks(NsModel* m, const char* dir, char* err, int errlen);

// Copies an MLX loader and the nanoseedlm_seeds.py beside it into dir.  -1 when either is missing.
int nslm_copy_loader(const char* loader, const char* dir);
// Copies the tokenizer, template and processor files that exist in from (all optional) into to.
void nslm_copy_model_files(const char* from, const char* to);

// The shard file of tensor `name` in a model.safetensors.index.json text (the weight_map entry).  0 when found.
int nslm_moe_index_lookup(const char* index_json, const char* name, char* file, int filelen);

// Per-channel mean squared activation of one expert with a shrinkage prior toward the layer's mean:
//   h[c] = (sum[c] + n0 * prior[c]) / (n + n0)      (n routed tokens, sum = their sum of x^2)
// With n + n0 == 0 the prior is returned.
void nslm_moe_blend(const double* sum, int64_t n, const double* prior, double n0, int dim, float* h);

// qwen4_exp (Qwen3.8-Flash-Next) routed-expert shapes from a config.json text: 1 when found, 0 for another model,
// -1 when a shape is missing or not a multiple of 8.
int nslm_qwen_experts(const char* config_json, int* d, int* ff, int* n_exp, int* n_layer);

// h of expert e from llama.cpp imatrix sums (sum [E][cols] of x^2, cnt [E] tokens): nslm_moe_blend toward the mean
// over all experts' tokens.
void nslm_imatrix_h(const float* sum, const float* cnt, int E, int cols, int e, double n0, float* h);

// Writes one BF16 tensor `name` of shape [d0][d1][d2] as a safetensors file (atomic: temp file + rename).
int nslm_st_write_bf16_3d(const char* path, const char* name, int d0, int d1, int d2, const uint16_t* data, char* err,
                          int errlen);
