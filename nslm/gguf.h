// nslm/gguf.h - a minimal read-only GGUF (v2, v3) reader (mmap): tensor lookup by name, for llama.cpp imatrix files
// and tables.  Dimensions are in GGUF order: dims[0] varies fastest.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define GGUF_F32 0

typedef struct {
    char name[128];
    int ndim;
    int64_t dims[4];
    uint32_t type;
    uint64_t offset;   // relative to the data section
} GgufTensor;

typedef struct {
    const uint8_t* map;
    size_t size;
    uint64_t data;     // file offset of the data section
    GgufTensor* t;
    int n;
} Gguf;

int gguf_open(Gguf* g, const char* path, char* err, int errlen);
void gguf_close(Gguf* g);
const GgufTensor* gguf_find(const Gguf* g, const char* name);
static inline const void* gguf_data(const Gguf* g, const GgufTensor* t) { return g->map + g->data + t->offset; }
// An F32 tensor of exactly n elements that lies inside the file, or NULL.
const float* gguf_f32(const Gguf* g, const char* name, int64_t n);
