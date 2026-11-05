// nslm/trellis_none.c - nslm/trellis_gpu.h for builds with no GPU trellis encoder (the Mac until the Metal one lands):
// nslm_tq_gpu_open says so, and nslm-moe --codec tq runs the CPU encoder.
#include <stdio.h>

#include "trellis_gpu.h"

NslmTqGpu* nslm_tq_gpu_open(char* err, int errlen) { snprintf(err, (size_t) errlen, "no GPU trellis encoder in this build"); return NULL; }
int nslm_tq_gpu_encode(NslmTqGpu* g, const float* w, const float* h, int rows, int cols, int T, int L, int K, uint32_t* words,
                       uint16_t* scales, double* err, char* msg, int msglen) {
    (void) g; (void) w; (void) h; (void) rows; (void) cols; (void) T; (void) L; (void) K; (void) words; (void) scales; (void) err;
    snprintf(msg, (size_t) msglen, "no GPU trellis encoder in this build");
    return -1;
}
void nslm_tq_gpu_close(NslmTqGpu* g) { (void) g; }
