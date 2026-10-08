// nslm/search8_gpu.h - the P = 8 seed search on the GPU (nslm/search8.metal + nslm/search8_metal.m): bit-identical to
// nslm8_search_ref.  Search8Args is shared with the Metal compiler.
#pragma once
#ifdef __METAL_VERSION__
#define S8_INT int
#else
#include <stdint.h>
#define S8_INT int32_t
#endif

#define S8_NCH 32            // seeds per table chunk in threadgroup memory
#define S8_TPB 256           // threads per threadgroup
#define S8_BPT 2             // blocks (rows) per thread
#define S8_ROWS (S8_TPB * S8_BPT)
#define S8_ENT 112           // floats per table entry: U[64], L[36], 1/diag[8], ok, pad

typedef struct {
    S8_INT rows, cols;       // the tensor
    S8_INT g0;               // first column group of this dispatch
    S8_INT bias;             // exponent window [bias, bias + 15]
    S8_INT n_seeds, n_exp, refit;
    S8_INT exp_delta[3];
} Search8Args;

#ifndef __METAL_VERSION__
#include "search8.h"
typedef struct Nslm8Gpu Nslm8Gpu;
Nslm8Gpu* nslm8_gpu_open(const char* metallib, char* err, int errlen);
// One tensor w[rows][cols] (cols a multiple of 8): seed / coef / ecode / err in row-major block order ([r][cols / 8]).
// sh: sqrt(h) per input channel (cols floats) or NULL; A: a lower-triangular 8 x 8 transform per column group
// (cols / 8 x 64 floats) or NULL (then sh).
int nslm8_gpu_search(Nslm8Gpu* g, const float* w, int rows, int cols, const float* sh, const float* A, int bias, const Search4Opts* o,
                     uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err, char* msg, int msglen);
void nslm8_gpu_close(Nslm8Gpu* g);
#endif
