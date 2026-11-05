// nslm/qwen4_ref.h - the qwen4_exp forward on the CPU from an MLX model folder: the reference of the GPU engines.
// f32 values, sums in double, one thread; for short sequences.
#pragma once
#include <stdint.h>

#include "qwen4_cfg.h"

typedef struct Qwen4Ref Qwen4Ref;

Qwen4Ref* qwen4_ref_open(const char* model_dir, char* err, int errlen);
void qwen4_ref_close(Qwen4Ref* r);
const Qwen4Cfg* qwen4_ref_cfg(const Qwen4Ref* r);
// The logits of every row of ids[0..n-1] (a fresh context): logits[t * vocab ..].  0 on success, else see the error.
int qwen4_ref_forward(Qwen4Ref* r, const int32_t* ids, int n, float* logits);
// The same, and the last layer's streams before the final mixer: Hout[n][hc d] (the MTP head's input; may be NULL)
int qwen4_ref_forward_h(Qwen4Ref* r, const int32_t* ids, int n, float* logits, float* Hout);
// The MTP head on rows [0, n): tok[t] the token after row t, Hin[t] its streams (forward_h's, or the head's own for a
// later draft) -> logits[n][vocab] and the head's streams before its mixer, Hout[n][hc d] (may be NULL)
int qwen4_ref_mtp(Qwen4Ref* r, const int32_t* tok, const float* Hin, int n, float* logits, float* Hout);
const char* qwen4_ref_error(const Qwen4Ref* r);
