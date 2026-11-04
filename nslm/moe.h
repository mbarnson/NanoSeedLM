// nslm/moe.h - MoVA helpers: sharded-checkpoint index lookup, per-expert calibration statistics, and the
// expanded-expert safetensors files.
#pragma once
#include <stddef.h>
#include <stdint.h>

// The header of nslm-moe's .blk / .blk4 / .blk8 block files.
typedef struct {
    char magic[8];   // "NSLMBLKM", "NSLMBLK4", "NSLMBLK8"
    uint32_t rows, cols, n_experts;
    uint32_t n_seeds, n_exp, refit;
    int32_t exp_delta[3];
    float n0;
    double seconds;
} MoeBlkHeader;

// Stream s (model_st.h) of an nslm-moe block file image of E slices of rows x cols.  P = 3 (.blk, SEED4): 0 seeds,
// 1 nibble words, 2 int32 bias per slice.  P = 4 (.blk4, SEED4P4): the same with coefficient words, then 3 exponent
// codes (block 2j in the low nibble of byte j).  -1 when blk is not that file.
int nslm_blk_stream(const uint8_t* blk, size_t len, int p, int E, int rows, int cols, int s, uint8_t* dst);

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
