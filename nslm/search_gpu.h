// nslm/search_gpu.h - the P = 3 seed search on the GPU (nslm/search.metal + nslm/search_metal.m): the same search as
// nslm/lib_search.c for every block of a tensor.  Tables are built per column group on the GPU (float-float, or exact
// emulated double for fragile seeds; the CPU uses double); a least-squares lower bound skips seeds that cannot beat a
// block's best; surviving candidates use the CPU's exact f32 expressions (no fast math, no contraction).
// SearchArgs is shared with the Metal compiler.
#pragma once
#ifdef __METAL_VERSION__
#define SG_INT int
#else
#include <stdint.h>
#define SG_INT int32_t
#endif

#define SG_NCH 64           // seeds per table chunk in threadgroup memory
#define SG_TPB 256          // threads per threadgroup
#define SG_BPT 2            // blocks (rows) per thread
#define SG_ROWS (SG_TPB * SG_BPT)
#define SG_ENT 40           // floats per table entry: U[24], Gi[6], R[6], prune margin, pad[3]

typedef struct {
    SG_INT rows, cols;      // the tensor
    SG_INT g0;              // first column group of this dispatch (grid x = column groups from g0)
    SG_INT bias;            // exponent window [bias, bias + 15]
    SG_INT n_seeds, n_exp, refit;
    SG_INT exp_delta[3];
    SG_INT prune;           // 0: evaluate every seed exactly (the CPU's loop, for checking the bound)
    float exact_kappa;      // seeds with float-float condition estimate (sum|G| sum|Gi|) > this get the exact double
                            // emulation of the CPU's table (0: every seed)
    SG_INT full;            // 1: full 8 x 8 transform per column group instead of sqrt(h) (GPTQ-style feedback):
                            // buffer 2 holds A (row-major, lower triangular) for column group g0 + tg.x at 64 * tg.x;
                            // objective |A (w - w_hat)|^2, x = A w, U' = A U (no CPU reference)
} SearchArgs;

#ifndef __METAL_VERSION__
#include "search.h"
typedef struct NslmGpu NslmGpu;
// metallib: the search library (out/res/search.metallib).  NULL and a message on failure.
NslmGpu* nslm_gpu_open(const char* metallib, char* err, int errlen);
// One tensor w[rows][cols] (cols a multiple of 8): seed / nib / err in row-major block order ([r][cols / 8]).
// sh: sqrt(h) per input channel (cols floats), or NULL for the unweighted search.  Returns 0 on success.
int nslm_gpu_search(NslmGpu* g, const float* w, int rows, int cols, const float* sh, int bias, const SearchOpts* o,
                    int prune, uint16_t* seed, uint16_t* nib, float* err, char* msg, int msglen);
// GPTQ-style feedback: one column block w8[rows][8] with transform a[64] (row-major, lower triangular), objective
// |a (w - w_hat)|^2.  Thread-safe across distinct NslmGpu handles (one per worker).
int nslm_gpu_search_block(NslmGpu* g, const float* w8, int rows, const float a[64], int bias, const SearchOpts* o,
                          uint16_t* seed, uint16_t* nib, float* err, char* msg, int msglen);
void nslm_gpu_close(NslmGpu* g);
#endif
