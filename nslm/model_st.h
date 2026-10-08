// nslm/model_st.h - NanoSeedLM model folders: safetensors shards and model.safetensors.index.json.
//
// A logical tensor NAME.weight is stored as safetensors entries by encoding (lead = [slices] when slices > 1):
//   BF16     NAME.weight   BF16  lead + [rows, cols]   (any name; 1 x cols tensors are [cols])
//   Q8, Q4   NAME.weight   U32   lead + [rows, cols * bits / 32]   (MLX affine, group 64)
//            NAME.scales   BF16  lead + [rows, cols / 64]
//            NAME.biases   BF16  lead + [rows, cols / 64]
//            (a Q8 / Q4 tensor whose name does not end in .weight, as MLA's projections: codes as NAME, then
//            NAME.scales and NAME.biases)
//   SEED4    NAME.seeds    U16   [slices, rows, cols / 8]   (P = 3; lfsr.h)
//            NAME.nibbles  U16   [slices, rows, cols / 8]
//            NAME.exp_bias I32   [slices]
//   SEED4P4  NAME.seeds    U16   [slices, rows, cols / 8]   (P = 4; search4.h)
//            NAME.coefs    U16   [slices, rows, cols / 8]
//            NAME.exp_bias I32   [slices]
//            NAME.codes    U8    [slices, rows, cols / 16]   (4-bit exponent codes, low nibble first)
//   SEED6P8  as SEED4P4 with NAME.coefs U32 [slices, rows, cols / 8]   (P = 8; search8.h)
// The writer pads each header so the data starts on a 16 KiB page and puts entries whose size is a page multiple
// first: those map into GPU buffers without a copy.
#pragma once
#include <stdint.h>

#include "format.h"

enum { NS_BF16 = 0, NS_SEED4 = 1, NS_Q8 = 2, NS_Q4 = 3, NS_SEED4P4 = 4, NS_SEED6P8 = 5 };   // = MF_* (engine/kernels_moe.metal)
#define NS_PAGE 16384

typedef struct {
    const uint8_t* p;      // data in the mapping
    uint64_t len;
    uint64_t off;          // file offset
    int shard;
} NsStream;

// Streams: BF16 0 values; Q8/Q4 0 packed words, 1 scales, 2 biases; SEED4 0 seeds, 1 nibble words, 2 exponent biases;
// SEED4P4 / SEED6P8 0 seeds, 1 coefficient words, 2 exponent biases, 3 exponent codes.
typedef struct {
    char name[96];
    int enc, slices, rows, cols;
    NsStream s[4];
} NsTensor;

typedef struct {
    StFile* sh;
    int nsh;
    NsTensor* t;
    int n;
} NsModel;

uint64_t ns_stream_len(int enc, int slices, int rows, int cols, int s);   // 0 = unused stream
int ns_open(NsModel* m, const char* dir, char* err, int errlen);         // index, else model*.safetensors
void ns_close(NsModel* m);
const NsTensor* ns_find(const NsModel* m, const char* name);
// The mapping that holds stream s can back a GPU buffer directly (page-aligned offset, padded length inside it).
int ns_mappable(const NsModel* m, const NsStream* s);

// Writer.  fill() is called once per stream, in tensor order and stream order 0..3.
typedef struct {
    const char* name;
    int enc, slices, rows, cols;
} NsSpec;
typedef int (*NsFill)(void* ctx, int tensor, int stream, uint8_t* dst, uint64_t len);
// Shards of at most shard_bytes (a larger tensor gets its own shard) in dir, and the index.  meta: extra
// "__metadata__" string pairs as a JSON object body (e.g. "\"k\": \"v\""), or NULL.
int ns_write(const char* dir, const NsSpec* t, int n, uint64_t shard_bytes, const char* meta, NsFill fill, void* ctx,
             char* err, int errlen);
