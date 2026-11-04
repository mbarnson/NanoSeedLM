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
const char* qwen4_ref_error(const Qwen4Ref* r);
