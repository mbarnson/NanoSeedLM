// nslm/emulate.h - emulated packs: a pack's routed experts, decoded (nslm/rows.h) and rounded to Q8_0, written over the
// Q8_0 expert tensors of a llama.cpp GGUF shard in place, so that llama.cpp runs the seed weights (RAM-tier plan T1).
#pragma once
#include <stdint.h>

#include "model_st.h"

uint16_t ns_f32_to_f16(float f);                         // IEEE half: nearest, ties to even
void ns_q8_0_row(const float* x, int n, uint8_t* out);   // llama.cpp quantize_row_q8_0_ref: n / 32 blocks of 34 bytes
#define NS_EMU_PACKS 8
// Each blk.L.ffn_{gate,up,down}_exps.weight of the shard from the packs' language_model.model.layers.L.mlp.switch_mlp
// tensor of that projection: expert e of projection p (0 gate, 1 up, 2 down) from packs[map[(3 L + p) E + e]], or from
// packs[0] when map is NULL.  All are checked before any is written.  *n: tensors written.  0, or -1 with err.
int ns_emulate_gguf(const NsModel* const* packs, int np, const uint8_t* map, size_t map_len, const char* gguf, int workers,
                    int* n, char* err, int errlen);
