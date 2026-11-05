// nslm/trellis_gpu.h - the trellis encoder of nslm/trellis.h on CUDA (nslm/trellis.cu), plan T2: the same words, scales
// and error as nslm_tq_encode_row, bit for bit (tests/test_trellis_gpu.c).  No Metal kernel yet (docs/concepts/parity.md).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct NslmTqGpu NslmTqGpu;
NslmTqGpu* nslm_tq_gpu_open(char* err, int errlen);   // NULL with "no CUDA device" when there is none
// A tensor w[rows][cols] with importance h[cols] (NULL: all 1; the same for every row, as an imatrix), each row as
// nslm_tq_encode_row(w + r cols, h, cols, T, ...) with L state bits and K bits a weight would encode it:
// words[rows][nslm_tq_row_words(cols, T, L, K)], scales[rows][ceil(cols / T)] (BF16), err[rows] (optional: the row's
// sum h (w - decoded)^2).  Needs L <= 16, 2 K <= L, K <= 8.  Returns 0, or -1 with a message.
int nslm_tq_gpu_encode(NslmTqGpu* g, const float* w, const float* h, int rows, int cols, int T, int L, int K, uint32_t* words,
                       uint16_t* scales, double* err, char* msg, int msglen);
void nslm_tq_gpu_close(NslmTqGpu* g);
#ifdef __cplusplus
}
#endif
