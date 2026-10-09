// nslm/gptq4.h - GPTQ-style error feedback over the P = 4 seed search (nslm/search4.h), C99; the loop of nslm-dense
// (O-02), shared by nslm-moe --scope mla --xtx.
#pragma once
#include <stdint.h>

// One search of rows x cols (cols / 8 column groups, each with its lower-triangular 8 x 8 transform A[g * 64 ..]) at one
// exponent bias: nslm4_gpu_search_a, or nslm4_search_ref_a per group.  0 on success.
typedef int (*Nslm4SearchA)(void* ctx, const float* w, int rows, int cols, const float* A, int bias, uint16_t* seed,
                            uint16_t* coef, uint8_t* ecode, char* err, int errlen);

// ns slices searched together (the heads of a per-head map, or one matrix): slice s is R x C (W[s], row-major; changed in
// place by the feedback) with U[s] = nslm_gptq_factor of its input's H (C x C) and exponent bias bias[s].  Column groups
// in order; group b of every slice with the same bias in one search (as the column groups of an R x 8n matrix), each with
// A = T^T, T the inverse of U's 8 x 8 diagonal block at b; its error E = (W_b - Q_b) T fed forward, within each batch of
// 16 groups at once and to the later batches after it: W[:, after] -= E U[b, after].  Seeds, coefficients and exponent
// codes (e - bias) per slice, [R][C / 8].  C a multiple of 8.  0, or -1 with err set.
int nslm4_gptq(Nslm4SearchA search, void* ctx, int ns, int R, int C, float* const* W, const double* const* U, const int* bias,
               uint16_t* const* seed, uint16_t* const* coef, uint8_t* const* ecode, char* err, int errlen);
