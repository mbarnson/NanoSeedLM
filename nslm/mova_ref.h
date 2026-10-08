// nslm/mova_ref.h - the MoVA forward on the CPU, in C: the reference the GPU engines (Metal, CUDA) are checked against.
//
// It reads a model folder (nslm/model_st.h) and computes the logits of a token sequence with the engines' rounding
// points (BF16 activations wherever the MLX reference implementation produces a BF16 tensor) but every dot product in
// double, so its own rounding error is far below the engines'.  Weights are the kernels' definitions: BF16, MLX affine
// Q8 / Q4 dequantized to BF16, seed blocks as isum * R32 * 2^e (lfsr.h, search4.h).  Rows are spread over threads.
// It is slow (a full matrix pass per layer for every expert used), meant for short sequences.
#pragma once
#include <stdint.h>

typedef struct MovaRef MovaRef;

MovaRef* mova_ref_open(const char* model_dir, int threads, char* err, int errlen);
void mova_ref_close(MovaRef* r);
int mova_ref_vocab(const MovaRef* r);
// Logits (BF16 values, as the engines store them) for rows [h0, n) of ids[0..n-1]: logits[(t - h0) * vocab ..].
// Router choices per row and sparse layer go to mlp_sel [n][n_sparse][top_k] and val_sel [n][n_sparse][top_kv] when
// non-NULL.  0 on success.
int mova_ref_forward(MovaRef* r, const int32_t* ids, int n, int h0, float* logits, int32_t* mlp_sel, int32_t* val_sel);
