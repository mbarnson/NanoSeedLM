// nslm/lib_affine.c - MLX-compatible affine quantization (nslm/affine.h).
#include "affine.h"

#include <math.h>
#include <string.h>

static inline float bf2f(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static inline float round_away(float x) { return x < 0 ? -floorf(-x + 0.5f) : floorf(x + 0.5f); }

void nslm_affine_quantize(const uint16_t* w, int rows, int cols, int bits, uint32_t* words, uint16_t* scales,
                          uint16_t* biases) {
    const float nb = (float) ((1 << bits) - 1);
    const int per = 32 / bits;
    const int64_t groups = (int64_t) rows * cols / 64;
    for (int64_t g = 0; g < groups; ++g) {
        const uint16_t* x = w + g * 64;
        float mx = bf2f(x[0]), mn = mx;
        for (int i = 1; i < 64; ++i) {
            const float v = bf2f(x[i]);
            mx = v > mx ? v : mx;
            mn = v < mn ? v : mn;
        }
        float delta = (mx - mn) / nb;
        if (delta < 1e-7f) delta = 1e-7f;
        const int neg = fabsf(mn) > fabsf(mx);
        float scale = neg ? delta : -delta;
        const float edge = neg ? mn : mx;
        const float q0 = round_away(edge / scale);
        float bias = 0.0f;
        if (q0 != 0.0f) { scale = edge / q0; bias = edge; }
        scales[g] = f2bf(scale);
        biases[g] = f2bf(bias);
        uint32_t* wd = words + g * 64 / per;
        for (int k = 0; k < 64 / per; ++k) wd[k] = 0;
        for (int i = 0; i < 64; ++i) {
            float q = round_away((bf2f(x[i]) - bias) / scale);
            q = q < 0 ? 0 : q > nb ? nb : q;
            wd[i / per] |= (uint32_t) q << (bits * (i % per));
        }
    }
}

void nslm_affine_dequantize(const uint32_t* words, const uint16_t* scales, const uint16_t* biases, int rows, int cols,
                            int bits, uint16_t* out) {
    const int per = 32 / bits;
    const uint32_t mask = (1u << bits) - 1u;
    const int64_t n = (int64_t) rows * cols;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t q = (words[i / per] >> (bits * (i % per))) & mask;
        const float s = bf2f(scales[i / 64]), b = bf2f(biases[i / 64]);
        out[i] = f2bf(s * (float) q + b);
    }
}
