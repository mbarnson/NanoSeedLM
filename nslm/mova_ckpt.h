// nslm/mova_ckpt.h - MoVA's HF checkpoint (48 safetensors shards + index), mmapped, for the packer and the engine.
#pragma once
#include <stdint.h>

typedef struct MovaCkpt MovaCkpt;

MovaCkpt* mova_ckpt_open(const char* model_dir, char* err, int errlen);
void mova_ckpt_close(MovaCkpt* k);
// The BF16 data of checkpoint tensor `name` with exactly rows x cols elements (a 1-D tensor of n is 1 x n); NULL and
// a message if missing, not BF16, or a different shape.  Thread-safe.
const uint16_t* mova_ckpt_bf16(MovaCkpt* k, const char* name, int rows, int cols, char* err, int errlen);
// The same for a stacked tensor of slices x rows x cols (3-D; 2-D when slices is 1), e.g. MLA's per-head maps.
const uint16_t* mova_ckpt_bf16_3d(MovaCkpt* k, const char* name, int slices, int rows, int cols, char* err, int errlen);
