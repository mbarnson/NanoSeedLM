// tests/test_kvq.c - the low-bit KV cache codecs (nslm/kvq.h) against brute-force nearest-value rounding.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kvq.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static float bf16r(float x) {   // round to BF16, nearest even
    uint32_t u;
    memcpy(&u, &x, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}
// brute force: the code (of n_codes non-negative ones, dec ascending) nearest a, ties to the even code
static int nearest(float a, int n_codes, float (*dec)(uint8_t)) {
    int best = 0;
    for (int c = 1; c < n_codes; ++c) {
        const double d = fabs((double) dec((uint8_t) c) - a), db = fabs((double) dec((uint8_t) best) - a);
        if (d < db || (d == db && !(c & 1))) best = c;
    }
    return best;
}

int main(void) {
    for (int c = 0; c <= 0x7E; ++c) {   // E4M3: every finite code round-trips, and decodes exactly to BF16
        const float v = kvq_e4m3_dec((uint8_t) c);
        CHECK(kvq_e4m3_enc_abs(v) == c, "e4m3 %d -> %g -> %d", c, v, kvq_e4m3_enc_abs(v));
        CHECK(bf16r(v) == v, "e4m3 %d not exact in BF16", c);
    }
    CHECK(kvq_e4m3_dec(0x7E) == 448 && kvq_e4m3_dec(0x01) == 0.001953125f && kvq_e4m3_dec(0xB8) == -1, "e4m3 values");
    uint32_t st = 12345;
    int bad = 0;
    for (int i = 0; i < 200000; ++i) {   // E4M3 rounding: the nearest code, ties even, saturated
        st = st * 1664525u + 1013904223u;
        float a = ldexpf((float) (st >> 8) / (1 << 24), (int) (st % 23) - 12);
        if (i % 7 == 0) a = (kvq_e4m3_dec((uint8_t) (st % 126)) + kvq_e4m3_dec((uint8_t) (st % 126 + 1))) / 2;   // a midpoint
        const int want = a >= 448 ? 0x7E : nearest(a, 0x7F, kvq_e4m3_dec);
        bad += kvq_e4m3_enc_abs(a) != want;
    }
    CHECK(!bad, "e4m3 rounding: %d wrong codes", bad);
    for (int c = 0; c < 16; ++c) CHECK(bf16r(kvq_e2m1_dec((uint8_t) c)) == kvq_e2m1_dec((uint8_t) c), "e2m1 %d", c);
    bad = 0;
    for (int i = 0; i < 200000; ++i) {   // E2M1 against a scale: the nearest code of x / s, ties even, saturated at 6
        st = st * 1664525u + 1013904223u;
        const float s = kvq_e4m3_dec((uint8_t) (1 + st % 0x7E));
        float x = ((float) (st >> 8) / (1 << 24) * 14 - 7) * s;
        if (i % 5 == 0) x = (kvq_e2m1_dec((uint8_t) (st % 7)) + kvq_e2m1_dec((uint8_t) (st % 7 + 1))) / 2 * s;
        const float a = fabsf(x) / s;
        const int k = a >= 6 ? 7 : nearest(a, 8, kvq_e2m1_dec), want = k && x < 0 ? k | 8 : k;
        bad += kvq_e2m1_enc(x, s) != want;
    }
    CHECK(!bad, "e2m1 rounding: %d wrong codes", bad);
    bad = 0;
    int bad4 = 0;
    double e8 = 0, e4 = 0, n2 = 0;
    for (int b = 0; b < 4000; ++b) {   // blocks: FP8 scale (smallest power of two with 448 s >= max), FP4 scale (nearest
        float x[32];                     // max / 6, ties even); decoded values within half a step of x
        const int sc = (int) (st % 11) - 4;   // block maxima within FP4's range (E4M3 scales: about 0.006 .. 2688)
        for (int i = 0; i < 32; ++i) {
            st = st * 1664525u + 1013904223u;
            x[i] = bf16r(ldexpf((float) (int32_t) st / 2147483648.0f, sc) * (1 + (i == 3) * 20));
        }
        if (b == 0) memset(x, 0, sizeof x);
        uint8_t c8[32], s8, c4[16], s4[2];
        kvq_fp8_block(x, 32, c8, &s8);
        kvq_fp4_block(x, 16, c4, &s4[0]);
        kvq_fp4_block(x + 16, 16, c4 + 8, &s4[1]);
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
        const double s = ldexp(1, s8 - 127);
        bad += amax > 0 ? !(448 * s >= amax && 448 * s / 2 < amax) : s8 != 127;
        for (int h = 0; h < 2; ++h) {
            float m = 0;
            for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(x[16 * h + i]));
            int want = 0;
            if (m > 0) {
                for (int c = 1; c <= 0x7E; ++c) {
                    const double d = fabs(6.0 * kvq_e4m3_dec((uint8_t) c) - m), dw = fabs(6.0 * kvq_e4m3_dec((uint8_t) want) - m);
                    if (d < dw || (d == dw && !(c & 1))) want = c;
                }
            }
            bad4 += s4[h] != want;
        }
        for (int i = 0; i < 32; ++i) {
            const float v8 = kvq_fp8_get(c8, &s8, i), v4 = kvq_fp4_get(c4, s4, i);
            CHECK(bf16r(v8) == v8 && bf16r(v4) == v4, "decoded values exact in BF16");
            e8 += ((double) v8 - x[i]) * ((double) v8 - x[i]);
            e4 += ((double) v4 - x[i]) * ((double) v4 - x[i]);
            n2 += (double) x[i] * x[i];
        }
    }
    CHECK(!bad && !bad4, "block scales: %d FP8, %d FP4 wrong", bad, bad4);
    printf("kvq: FP8 relative error %.4f, FP4 %.4f (uniform blocks with an outlier)\n", sqrt(e8 / n2), sqrt(e4 / n2));
    CHECK(sqrt(e8 / n2) < 0.04 && sqrt(e4 / n2) < 0.2, "block errors");
    printf("test_kvq: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
