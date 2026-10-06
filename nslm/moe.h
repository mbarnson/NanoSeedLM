// nslm/moe.h - MoVA helpers: sharded-checkpoint index lookup, per-expert calibration statistics, and the
// expanded-expert safetensors files.
#pragma once
#include <stdint.h>

// The shard file of tensor `name` in a model.safetensors.index.json text (the weight_map entry).  0 when found.
int nslm_moe_index_lookup(const char* index_json, const char* name, char* file, int filelen);

// Per-channel mean squared activation of one expert with a shrinkage prior toward the layer's mean:
//   h[c] = (sum[c] + n0 * prior[c]) / (n + n0)      (n routed tokens, sum = their sum of x^2)
// With n + n0 == 0 the prior is returned.
void nslm_moe_blend(const double* sum, int64_t n, const double* prior, double n0, int dim, float* h);

// Writes one BF16 tensor `name` of shape [d0][d1][d2] as a safetensors file (atomic: temp file + rename).
int nslm_st_write_bf16_3d(const char* path, const char* name, int d0, int d1, int d2, const uint16_t* data, char* err,
                          int errlen);
