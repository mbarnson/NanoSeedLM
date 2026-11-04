// tests/test_ple.c - PLE table rows (nslm/ple.h) from a small GGUF written here: one 2 x 32 table per type (BF16, Q8_0,
// Q5_1, IQ4_NL) with values checked against the ggml block formats, and row bounds.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ple.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void put(FILE* f, const void* p, size_t n) { fwrite(p, 1, n, f); }
static void put_u32(FILE* f, uint32_t v) { put(f, &v, 4); }
static void put_u64(FILE* f, uint64_t v) { put(f, &v, 8); }
static void put_str(FILE* f, const char* s) { put_u64(f, strlen(s)); put(f, s, strlen(s)); }

int main(void) {
    // Row r of each table: BF16 v = r + c / 8; Q8_0 d = 0.5, q = c - 16 + r; Q5_1 d = 0.25, m = -1, q = (c + r) & 31;
    // IQ4_NL d = 2, index = (c + r) & 15.  f16 0.5 = 0x3800, 0.25 = 0x3400, -1 = 0xBC00, 2 = 0x4000.
    static const int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    uint8_t bf[2][64], q8[2][34], q5[2][24], iq[2][18];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 32; ++c) {
            float v = (float) r + (float) c / 8;
            uint32_t u;
            memcpy(&u, &v, 4);
            const uint16_t h = (uint16_t) (u >> 16);   // exact: few mantissa bits
            memcpy(&bf[r][2 * c], &h, 2);
        }
        q8[r][0] = 0x00; q8[r][1] = 0x38;
        for (int c = 0; c < 32; ++c) q8[r][2 + c] = (uint8_t) (int8_t) (c - 16 + r);
        memset(q5[r], 0, sizeof q5[r]);
        q5[r][0] = 0x00; q5[r][1] = 0x34; q5[r][2] = 0x00; q5[r][3] = 0xBC;
        for (int j = 0; j < 16; ++j) {
            const int lo = (j + r) & 31, hi = (j + 16 + r) & 31;
            q5[r][8 + j] = (uint8_t) ((lo & 15) | (hi & 15) << 4);
            uint32_t qh;   // a little-endian uint32: bit j is element j's fifth bit, bit j + 16 element j + 16's
            memcpy(&qh, &q5[r][4], 4);
            qh |= (uint32_t) (lo >> 4) << j | (uint32_t) (hi >> 4) << (j + 16);
            memcpy(&q5[r][4], &qh, 4);
        }
        iq[r][0] = 0x00; iq[r][1] = 0x40;
        for (int j = 0; j < 16; ++j) iq[r][2 + j] = (uint8_t) (((j + r) & 15) | ((j + 16 + r) & 15) << 4);
    }
    const char* names[4] = {"t.bf16", "t.q8_0", "t.q5_1", "t.iq4_nl"};
    const uint32_t types[4] = {30, 8, 7, 20};
    const void* data[4] = {bf, q8, q5, iq};
    const size_t bytes[4] = {sizeof bf, sizeof q8, sizeof q5, sizeof iq};
    const char* path = "out/test_ple.gguf";
    FILE* f = fopen(path, "wb");
    put(f, "GGUF", 4); put_u32(f, 3); put_u64(f, 4); put_u64(f, 0);
    uint64_t off = 0;
    for (int k = 0; k < 4; ++k) {
        put_str(f, names[k]); put_u32(f, 2); put_u64(f, 32); put_u64(f, 2); put_u32(f, types[k]); put_u64(f, off);
        off += (bytes[k] + 31) / 32 * 32;
    }
    while (ftell(f) % 32) fputc(0, f);
    for (int k = 0; k < 4; ++k) {
        put(f, data[k], bytes[k]);
        for (size_t z = bytes[k]; z % 32; ++z) fputc(0, f);
    }
    fclose(f);

    char err[256] = "";
    float row[32];
    for (int k = 0; k < 4; ++k) {
        PleTable t;
        CHECK(ple_open(&t, path, names[k], err, sizeof err) == 0, "%s: %s", names[k], err);
        CHECK(t.rows == 2 && t.width == 32, "%s shape", names[k]);
        for (int r = 0; r < 2; ++r) {
            CHECK(ple_row(&t, r, row) == 0, "%s row %d", names[k], r);
            for (int c = 0; c < 32; ++c) {
                const float want = k == 0 ? (float) r + (float) c / 8 : k == 1 ? 0.5f * (float) (c - 16 + r)
                                 : k == 2 ? 0.25f * (float) ((c + r) & 31) - 1.0f : 2.0f * kIq4nl[(c + r) & 15];
                CHECK(fabsf(row[c] - want) < 1e-6f, "%s row %d col %d: %g want %g", names[k], r, c, row[c], want);
            }
        }
        CHECK(ple_row(&t, 2, row) != 0, "%s: row 2 of 2 read", names[k]);
        ple_close(&t);
    }
    PleTable t;
    CHECK(ple_open(&t, path, "t.missing", err, sizeof err) != 0, "missing table opened");
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
