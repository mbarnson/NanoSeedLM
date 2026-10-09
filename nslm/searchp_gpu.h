// nslm/searchp_gpu.h - the P = 3 / 8 seed search on the GPU (nslm/searchp.metal + nslm/searchp_metal.m): bit-identical
// to nslmp_search_ref.  SearchPArgs is shared with the Metal compiler.
#pragma once
#ifdef __METAL_VERSION__
#define SP_INT int
#else
#include <stdint.h>
#define SP_INT int32_t
#endif

#define SP_NCH 32            // seeds per table chunk in threadgroup memory
#define SP_TPB 256           // threads per threadgroup
#define SP_BPT 2             // blocks (rows) per thread
#define SP_ROWS (SP_TPB * SP_BPT)
#define SP_ENT 112           // floats per table entry (P = 8's size): U[8 P], L[P (P + 1) / 2], 1/diag[P]; ok at [108]
#define SP_OK 108

typedef struct {
    SP_INT rows, cols;       // the tensor
    SP_INT g0;               // first column group of this dispatch
    SP_INT bias;             // exponent window [bias, bias + 15]
    SP_INT n_seeds, n_exp, refit;
    SP_INT exp_delta[3];
} SearchPArgs;

#ifndef __METAL_VERSION__
#include "searchp.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct NslmPGpu NslmPGpu;
NslmPGpu* nslmp_gpu_open(const char* metallib, char* err, int errlen);
// One tensor w[rows][cols] (cols a multiple of 8) in P = 3 or 8 blocks: seed / coef / ecode / err in row-major block
// order ([r][cols / 8]).  sh: sqrt(h) per input channel (cols floats) or NULL; A: a lower-triangular 8 x 8 transform per
// column group (cols / 8 x 64 floats) or NULL (then sh).
int nslmp_gpu_search(NslmPGpu* g, int P, const float* w, int rows, int cols, const float* sh, const float* A, int bias,
                     const Search4Opts* o, uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err, char* msg, int msglen);
void nslmp_gpu_close(NslmPGpu* g);
#ifdef __cplusplus
}
#endif
#endif
