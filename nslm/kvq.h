// nslm/kvq.h - low-bit KV cache codecs (header-only C99), DeepSeek-V4.1's formats.  The scalar functions are the spec;
// the Metal kernels (engine/kernels_moe.metal) must match them bit for bit.
//
// FP8: E4M3 values (bias 7, max 448, no infinities), one E8M0 scale (2^(byte - 127)) per 32 values: the smallest power
//   of two s with 448 s >= max |x|; codes = e4m3_rne(x / s).  8.25 bits a value.
// FP4: E2M1 values (0, 0.5, 1, 1.5, 2, 3, 4, 6; sign in bit 3), one E4M3 scale per 16 values: the E4M3 value nearest
//   max |x| / 6 (ties to the even code); codes = e2m1_rne(x / s), saturated at 6.  Two codes a byte, the even value in
//   the low nibble.  4.5 bits a value.  The scale's range bounds a block's max |x| to about [0.006, 2688]: smaller
//   blocks decode as zeros, larger ones saturate (MLA latents and RoPE keys: 0.1 .. 130).
// Encoding uses only exact operations (powers of two, comparisons against exact products), so any IEEE single-precision
// implementation gets the same codes.  Decoded values (code value times scale) are exact in BF16.
#pragma once
#include <math.h>
#include <stdint.h>

#define KVQ_FP8_BLOCK 32
#define KVQ_FP4_BLOCK 16

static inline float kvq_e4m3_dec(uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const float v = e ? ldexpf(1.0f + (float) m / 8, e - 7) : ldexpf((float) m, -9);
    return c & 0x80 ? -v : v;
}
// a >= 0: the E4M3 code of a, rounded to nearest even, saturated at 448
static inline uint8_t kvq_e4m3_enc_abs(float a) {
    if (!(a < 448.0f)) return 0x7E;
    if (a < 0.015625f) return (uint8_t) rintf(a * 512.0f);   // subnormals (8: the smallest normal)
    int e;
    const float f = frexpf(a, &e);   // a = 2f 2^(e - 1), 2f in [1, 2)
    int E = e - 1, m = (int) rintf((2 * f - 1) * 8);
    if (m == 8) { ++E; m = 0; }
    const int c = ((E + 7) << 3) | m;
    return (uint8_t) (c > 0x7E ? 0x7E : c);
}
static inline float kvq_e2m1_dec(uint8_t n) {
    static const float v[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return n & 8 ? -v[n & 7] : v[n & 7];
}
// the E2M1 code of x / s (s > 0, E4M3): x against the midpoints times s (exact), ties to the even code
static inline uint8_t kvq_e2m1_enc(float x, float s) {
    static const float mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    const float a = fabsf(x);
    int k = 0;
    while (k < 7 && a > mid[k] * s) ++k;
    if (k < 7 && a == mid[k] * s && (k & 1)) ++k;   // a tie between codes k and k + 1: the even one
    return (uint8_t) (k && x < 0 ? k | 8 : k);
}
// the E8M0 scale byte of an FP8 block with max |x| = amax
static inline uint8_t kvq_fp8_scale(float amax) {
    if (!(amax > 0)) return 127;
    int k;
    frexpf(amax, &k);   // amax in [2^(k-1), 2^k); 448 2^(k-9) = 0.875 2^k
    int e = amax <= ldexpf(448.0f, k - 9) ? k - 9 : k - 8;
    if (e < -126) e = -126;   // a normal scale
    if (e > 127) e = 127;
    return (uint8_t) (e + 127);
}
// the E4M3 scale code of an FP4 block with max |x| = amax: nearest to amax / 6 (6 dec(c) is exact), ties to even
static inline uint8_t kvq_fp4_scale(float amax) {
    if (!(amax > 0)) return 0;
    const int c0 = kvq_e4m3_enc_abs(amax / 6);
    int best = -1;
    float bd = 0;
    for (int c = c0 > 0 ? c0 - 1 : 0; c <= c0 + 1 && c <= 0x7E; ++c) {
        const float d = fabsf(6 * kvq_e4m3_dec((uint8_t) c) - amax);
        if (best < 0 || d < bd || (d == bd && !(c & 1))) { best = c; bd = d; }
    }
    return (uint8_t) best;
}
// One FP8 block: n (<= 32) values x into codes[n] and its scale byte
static inline void kvq_fp8_block(const float* x, int n, uint8_t* codes, uint8_t* scale) {
    float amax = 0;
    for (int i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(x[i]));
    *scale = kvq_fp8_scale(amax);
    const int e = (int) *scale - 127;
    for (int i = 0; i < n; ++i) {
        const uint8_t c = kvq_e4m3_enc_abs(ldexpf(fabsf(x[i]), -e));
        codes[i] = (uint8_t) (c && x[i] < 0 ? c | 0x80 : c);
    }
}
// One FP4 block: n (even, <= 16) values x into n / 2 bytes and its scale code
static inline void kvq_fp4_block(const float* x, int n, uint8_t* codes, uint8_t* scale) {
    float amax = 0;
    for (int i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(x[i]));
    *scale = kvq_fp4_scale(amax);
    const float s = kvq_e4m3_dec(*scale);
    for (int i = 0; i < n; i += 2) {
        const uint8_t lo = s > 0 ? kvq_e2m1_enc(x[i], s) : 0, hi = s > 0 ? kvq_e2m1_enc(x[i + 1], s) : 0;
        codes[i / 2] = (uint8_t) (lo | hi << 4);
    }
}
// A cache row of n values whose first `lead` (a multiple of 32) are FP8 and the rest FP4: lead + (n - lead) / 2 code
// bytes, then lead / 32 + (n - lead) / 16 scales.  The MLA cache (engine KV formats fp8 / fp4): the RoPE key all FP8;
// the latent all FP8 (fp8) or its first KVQ_FP4_LEAD values FP8 and the rest FP4 (fp4).  TransMLA latents put most of
// their variance in the leading dims, and the RoPE key in FP4 cost most of all-FP4's KL divergence: held-out KLD of the
// J768 P=4 model (BF16 cache 0.2199): fp8 0.2221; fp4 with 256 / 128 / 0 leading FP8 dims 0.2223 / 0.2264 / 0.2307;
// everything FP4 (DeepSeek's layout) 0.2677.
#define KVQ_FP4_LEAD 256
static inline int kvq_row_bytes(int lead, int n) { return lead + (n - lead) / 2; }
static inline int kvq_row_scales(int lead, int n) { return lead / 32 + (n - lead) / 16; }
static inline void kvq_row(const float* x, int n, int lead, uint8_t* codes, uint8_t* scales) {
    for (int b = 0; b < lead; b += KVQ_FP8_BLOCK) kvq_fp8_block(x + b, KVQ_FP8_BLOCK, codes + b, scales + b / KVQ_FP8_BLOCK);
    for (int b = 0; b < n - lead; b += KVQ_FP4_BLOCK)
        kvq_fp4_block(x + lead + b, KVQ_FP4_BLOCK, codes + lead + b / 2, scales + lead / KVQ_FP8_BLOCK + b / KVQ_FP4_BLOCK);
}
// value i of a row written by kvq_row
static inline float kvq_row_get(const uint8_t* codes, const uint8_t* scales, int lead, int i);
// value i of an FP8 / FP4 row (codes and per-block scales as written by the block functions over the row)
static inline float kvq_fp8_get(const uint8_t* codes, const uint8_t* scales, int i) {
    return ldexpf(kvq_e4m3_dec(codes[i]), (int) scales[i / KVQ_FP8_BLOCK] - 127);
}
static inline float kvq_fp4_get(const uint8_t* codes, const uint8_t* scales, int i) {
    return kvq_e2m1_dec((uint8_t) (codes[i / 2] >> (4 * (i & 1)) & 15)) * kvq_e4m3_dec(scales[i / KVQ_FP4_BLOCK]);
}
static inline float kvq_row_get(const uint8_t* codes, const uint8_t* scales, int lead, int i) {
    return i < lead ? kvq_fp8_get(codes, scales, i) : kvq_fp4_get(codes + lead, scales + lead / KVQ_FP8_BLOCK, i - lead);
}
