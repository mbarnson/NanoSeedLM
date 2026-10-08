// nslm/search4_gpu.h - the P = 4 seed search on the GPU (nslm/search4.metal + nslm/search4_metal.m): bit-identical to
// nslm4_search_ref.  Search4Args is shared with the Metal compiler.
#pragma once
#ifdef __METAL_VERSION__
#define S4_INT int
#else
#include <stdint.h>
#define S4_INT int32_t
#endif

#define S4_NCH 64            // seeds per table chunk in threadgroup memory
#define S4_TPB 256           // threads per threadgroup
#define S4_BPT 2             // blocks (rows) per thread
#define S4_ROWS (S4_TPB * S4_BPT)
#define S4_ENT 48            // floats per table entry (= NSLM4_ENT): U[32], L[10], 1/diag[4], ok, pad

typedef struct {
    S4_INT rows, cols;       // the tensor
    S4_INT g0;               // first column group of this dispatch
    S4_INT bias;             // exponent window [bias, bias + 15]
    S4_INT n_seeds, n_exp, refit;
    S4_INT exp_delta[3];
} Search4Args;

#ifndef __METAL_VERSION__
#include "search4.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct Nslm4Gpu Nslm4Gpu;
// library: the Metal build's search4.metallib (the CUDA build links the kernel and ignores it).
Nslm4Gpu* nslm4_gpu_open(const char* library, char* err, int errlen);
// One tensor w[rows][cols] (cols a multiple of 8): seed / coef / ecode / err in row-major block order ([r][cols / 8]).
// sh: sqrt(h) per input channel (cols floats), or NULL for 1.  Returns 0 on success.
int nslm4_gpu_search(Nslm4Gpu* g, const float* w, int rows, int cols, const float* sh, int bias, const Search4Opts* o,
                     uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err, char* msg, int msglen);
void nslm4_gpu_close(Nslm4Gpu* g);
#ifdef __cplusplus
}
#endif
#endif
