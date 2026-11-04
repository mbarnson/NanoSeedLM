// nslm/rows.h - one weight row of a model-folder tensor (nslm/model_st.h) as f32, with the kernels' definitions: BF16, MLX
// affine (s * q + b), seed blocks (isum * R32 * 2^e, nslm/lfsr.h and nslm/search4.h).  The CPU references read weights here.
#pragma once
#include <stdint.h>

#include "model_st.h"

typedef struct { uint32_t g24[65536], g32[65536]; } NsRows;   // LFSR stream tables of P = 3 and P = 4 seeds

void ns_rows_init(NsRows* r);
// Row `row` of slice `slice` into out[cols].  round = 1 rounds affine values to BF16, as the MoVA kernels dequantize.
void ns_row_f32(const NsRows* r, const NsTensor* t, int slice, int row, int round, float* out);
// Start reading that row from the files (st_willneed): BF16 and affine tensors; the others are not prefetched
void ns_row_prefetch(const NsTensor* t, int slice, int row);
