// nslm/ple.h - rows of a GGUF table as f32: the qwen4 PLE n-gram table of Unsloth's engram files
// (per_layer_token_embd.weight, 160 x 320 M) in BF16, Q8_0, Q5_1 or IQ4_NL, read through the file mapping.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

typedef struct {
    Gguf g;
    const uint8_t* data;
    int64_t rows, width;
    uint32_t type;
    size_t row_bytes;
} PleTable;

int ple_open(PleTable* t, const char* path, const char* name, char* err, int errlen);
void ple_close(PleTable* t);
int ple_row(const PleTable* t, int64_t row, float* out);   // width floats; -1 when row is out of range
