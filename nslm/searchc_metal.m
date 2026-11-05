// nslm/searchc_metal.m - no Metal kernel for the any-C seed search yet (nslm/searchc_gpu.h is CUDA only,
// docs/concepts/parity.md): nslmc_gpu_open fails, so nslm-moe --codec s3 stops with a message on a Mac.
#include <stdio.h>

#include "searchc_gpu.h"

struct NslmCGpu { int unused; };

NslmCGpu* nslmc_gpu_open(char* err, int errlen) {
    snprintf(err, (size_t) errlen, "no Metal kernel for --codec s3 (CUDA only)");
    return NULL;
}
int nslmc_gpu_search(NslmCGpu* g, int C, int P, const float* w, int rows, int cols, const float* sh, int bias, const Search4Opts* o,
                     uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err, char* msg, int msglen) {
    (void) g; (void) C; (void) P; (void) w; (void) rows; (void) cols; (void) sh; (void) bias; (void) o;
    (void) seed; (void) coef; (void) ecode; (void) err;
    snprintf(msg, (size_t) msglen, "no Metal kernel for --codec s3");
    return -1;
}
void nslmc_gpu_close(NslmCGpu* g) { (void) g; }
