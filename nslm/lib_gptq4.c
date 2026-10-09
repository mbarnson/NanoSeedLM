// nslm/lib_gptq4.c - nslm4_gptq (gptq4.h): nslm_gptq (gptq.h) over the P = 4 search's 16-bit coefficient words.
#include "gptq4.h"

#include <stdlib.h>

#include "gptq.h"
#include "search4.h"

typedef struct { Nslm4SearchA search; void* ctx; } Ctx4;
static int search32(void* ctx, const float* w, int rows, int cols, const float* A, int bias, uint16_t* seed, uint32_t* coef,
                    uint8_t* ecode, char* err, int errlen) {
    const Ctx4* c = (const Ctx4*) ctx;
    const size_t nb = (size_t) rows * cols / 8;
    uint16_t* c16 = malloc(2 * nb);
    const int rc = c->search(c->ctx, w, rows, cols, A, bias, seed, c16, ecode, err, errlen);
    for (size_t k = 0; k < nb; ++k) coef[k] = c16[k];
    free(c16);
    return rc;
}
static void decode4(uint16_t seed, uint32_t coef, int e, uint16_t bf[8]) { nslm4_decode_block(seed, (uint16_t) coef, e, bf); }

int nslm4_gptq(Nslm4SearchA search, void* ctx, int ns, int R, int C, float* const* W, const double* const* U, const int* bias,
               uint16_t* const* seed, uint16_t* const* coef, uint8_t* const* ecode, char* err, int errlen) {
    Ctx4 c = {search, ctx};
    const NslmGptq q = {search32, decode4, &c, 1};
    const size_t nb = (size_t) R * (C / 8);
    uint32_t** c32 = malloc(sizeof(uint32_t*) * (size_t) (ns > 0 ? ns : 1));
    for (int s = 0; s < ns; ++s) c32[s] = malloc(4 * nb);
    const int rc = nslm_gptq(&q, ns, R, C, W, U, bias, seed, c32, ecode, err, errlen);
    for (int s = 0; s < ns; ++s) {
        if (!rc)
            for (size_t k = 0; k < nb; ++k) coef[s][k] = (uint16_t) c32[s][k];
        free(c32[s]);
    }
    free(c32);
    return rc;
}
