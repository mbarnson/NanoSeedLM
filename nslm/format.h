// nslm/format.h - a minimal read-only safetensors reader (mmap) for the C tools.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char name[96];
    char dtype[8];
    int ndim;
    int64_t shape[4];
    uint64_t begin, end;     // byte offsets relative to the data section
} StEntry;

typedef struct {
    const uint8_t* map;
    size_t size;
    uint64_t data;           // file offset of the data section
    StEntry* e;
    int n;
} StFile;

int st_open_file(StFile* s, const char* path, char* err, int errlen);
void st_close_file(StFile* s);
const StEntry* st_find(const StFile* s, const char* name);
static inline const uint8_t* st_data(const StFile* s, const StEntry* e) { return s->map + s->data + e->begin; }
