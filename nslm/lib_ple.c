// nslm/lib_ple.c - see ple.h.  The ggml block formats: Q8_0 {f16 d; int8 q[32]}, Q5_1 {f16 d, m; u32 qh; u8 qs[16]},
// IQ4_NL {f16 d; u8 qs[16]} with a fixed 16-entry codebook.  Elements j and j + 16 share byte j of qs.
#include "ple.h"

#include "format.h"

#include <stdio.h>
#include <string.h>

enum { T_Q5_1 = 7, T_Q8_0 = 8, T_IQ4_NL = 20, T_BF16 = 30 };
static const int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static float f16(const uint8_t* p) {
    const uint32_t h = (uint32_t) p[0] | (uint32_t) p[1] << 8, s = (h & 0x8000u) << 16, e = h >> 10 & 31, m = h & 0x3FF;
    uint32_t u;
    if (e == 31) u = s | 0x7F800000u | m << 13;
    else if (e) u = s | (e + 112) << 23 | m << 13;
    else if (!m) u = s;
    else {   // subnormal: normalize
        int k = -1;
        uint32_t mm = m;
        do { ++k; mm <<= 1; } while (!(mm & 0x400));
        u = s | (uint32_t) (112 - k) << 23 | (mm & 0x3FF) << 13;
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static size_t block_bytes(uint32_t type) {
    return type == T_Q8_0 ? 34 : type == T_Q5_1 ? 24 : type == T_IQ4_NL ? 18 : type == T_BF16 ? 64 : 0;
}

int ple_open(PleTable* t, const char* path, const char* name, char* err, int errlen) {
    memset(t, 0, sizeof *t);
    if (gguf_open(&t->g, path, err, errlen)) return -1;
    const GgufTensor* g = gguf_find(&t->g, name);
    const size_t bb = g ? block_bytes(g->type) : 0;
    if (!g || g->ndim != 2 || !bb || g->dims[0] % 32) {
        snprintf(err, (size_t) errlen, "%s: no %s table in BF16, Q8_0, Q5_1 or IQ4_NL with whole 32-value blocks", path, name);
        gguf_close(&t->g);
        return -1;
    }
    t->width = g->dims[0];
    t->rows = g->dims[1];
    t->type = g->type;
    t->row_bytes = (size_t) t->width / 32 * bb;
    t->data = (const uint8_t*) gguf_data(&t->g, g);
    if (t->g.data + g->offset + (uint64_t) t->rows * t->row_bytes > t->g.size) {
        snprintf(err, (size_t) errlen, "%s: %s runs past the end of the file", path, name);
        gguf_close(&t->g);
        return -1;
    }
    return 0;
}

void ple_close(PleTable* t) { gguf_close(&t->g); memset(t, 0, sizeof *t); }

int ple_prefetch(const PleTable* t, int64_t row) {
    if (row < 0 || row >= t->rows) return -1;
    st_willneed(t->data + (size_t) row * t->row_bytes, t->row_bytes);
    return 0;
}

int ple_row(const PleTable* t, int64_t row, float* out) {
    if (row < 0 || row >= t->rows) return -1;
    const uint8_t* b = t->data + (size_t) row * t->row_bytes;
    for (int64_t k = 0; k < t->width / 32; ++k, out += 32) {
        if (t->type == T_BF16) {
            for (int j = 0; j < 32; ++j) {
                const uint32_t u = ((uint32_t) b[2 * j] | (uint32_t) b[2 * j + 1] << 8) << 16;
                memcpy(&out[j], &u, 4);
            }
            b += 64;
        } else if (t->type == T_Q8_0) {
            const float d = f16(b);
            for (int j = 0; j < 32; ++j) out[j] = d * (float) (int8_t) b[2 + j];
            b += 34;
        } else if (t->type == T_Q5_1) {
            const float d = f16(b), m = f16(b + 2);
            const uint32_t qh = (uint32_t) b[4] | (uint32_t) b[5] << 8 | (uint32_t) b[6] << 16 | (uint32_t) b[7] << 24;
            for (int j = 0; j < 16; ++j) {
                out[j] = d * (float) ((b[8 + j] & 15) | (qh >> j & 1) << 4) + m;
                out[j + 16] = d * (float) ((b[8 + j] >> 4) | (qh >> (j + 16) & 1) << 4) + m;
            }
            b += 24;
        } else {
            const float d = f16(b);
            for (int j = 0; j < 16; ++j) { out[j] = d * kIq4nl[b[2 + j] & 15]; out[j + 16] = d * kIq4nl[b[2 + j] >> 4]; }
            b += 18;
        }
    }
    return 0;
}
