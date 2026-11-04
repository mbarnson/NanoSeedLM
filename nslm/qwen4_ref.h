// nslm/qwen4_ref.h - the Qwen3.8-Flash-Next (qwen4_exp) forward on the CPU, in C: the reference the GPU engines are
// checked against.  It reads an MLX model folder (BF16, MLX affine Q4 / Q8, seed experts; nslm/rows.h) and computes the
// logits of a token sequence in f32, with dot products in double.  It is slow and single-threaded: for short sequences.
//
// One layer is two branches on a state of hc streams (hyper-connections).  Each branch normalises the streams, mixes
// them into one input (sigmoid weights from a low-rank map), runs the block (Gated DeltaNet or QSA attention, then the
// MoE) and adds the output back to every stream with its own weight 2 * sigmoid(.).  A final mixer gives the head input.
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
