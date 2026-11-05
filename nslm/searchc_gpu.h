// nslm/searchc_gpu.h - the seed search of nslm/searchc.h on CUDA (nslm/searchc.cu): bit-identical to nslmc_search_ref,
// for (C, P) = (12, 4), (16, 7) and (8, 3).  No Metal kernel yet (docs/concepts/parity.md).
#pragma once
#include <stdint.h>

#include "searchc.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct NslmCGpu NslmCGpu;
NslmCGpu* nslmc_gpu_open(char* err, int errlen);
// One tensor w[rows][cols] with input scales sh[cols] (NULL: 1) in blocks of C along each row; a row's last block is
// padded with zero weights and scales when C does not divide cols.  Outputs in row-major block order [r][ceil(cols / C)].
int nslmc_gpu_search(NslmCGpu* g, int C, int P, const float* w, int rows, int cols, const float* sh, int bias, const Search4Opts* o,
                     uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err, char* msg, int msglen);
void nslmc_gpu_close(NslmCGpu* g);
#ifdef __cplusplus
}
#endif
