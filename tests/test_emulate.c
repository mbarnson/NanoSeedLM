// tests/test_emulate.c - emulated packs (nslm/emulate.h): f32 -> f16 rounding, Q8_0 rows as llama.cpp quantizes them,
// and a pack's experts (and dense tensors) written over the Q8_0 tensors of a small GGUF, in place.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "emulate.h"
#include "gguf.h"
#include "rows.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void put(FILE* f, const void* p, size_t n) { fwrite(p, 1, n, f); }
static void put_u32(FILE* f, uint32_t v) { put(f, &v, 4); }
static void put_u64(FILE* f, uint64_t v) { put(f, &v, 8); }
static void put_str(FILE* f, const char* s) { put_u64(f, strlen(s)); put(f, s, strlen(s)); }

static float h2f(uint16_t h) {   // IEEE half, by its definition
    const int e = h >> 10 & 31, m = h & 0x3FF;
    const float v = e ? ldexpf(1.0f + m / 1024.0f, e - 15) : ldexpf(m / 1024.0f, -14);
    return h & 0x8000 ? -v : v;
}

static uint32_t rng = 12345;
static uint32_t next(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

enum { E = 2, R = 4, CG = 64, CD = 32 };   // experts, rows; gate and down columns
static int fill(void* ctx, int t, int s, uint8_t* dst, uint64_t len) {
    (void) ctx;
    if (s == 2) { for (uint64_t i = 0; i < len / 4; ++i) { const int32_t b = -12 - t; memcpy(dst + 4 * i, &b, 4); } return 0; }
    for (uint64_t i = 0; i < len; ++i) dst[i] = (uint8_t) next();
    return 0;
}

// Tensors in GGUF order: gate (Q8_0, CG x R x E), a norm (F32, kept), down (Q8_0, CD x R x E) at aligned offsets.
static const uint64_t kOff[3] = {0, 576, 640};
static void write_gguf(const char* path, int rows) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("FAIL: cannot write %s\n", path); exit(1); }
    put(f, "GGUF", 4); put_u32(f, 3); put_u64(f, 3); put_u64(f, 1);
    put_str(f, "general.alignment"); put_u32(f, 4); put_u32(f, 32);
    put_str(f, "blk.0.ffn_gate_exps.weight"); put_u32(f, 3); put_u64(f, CG); put_u64(f, rows); put_u64(f, E); put_u32(f, 8); put_u64(f, kOff[0]);
    put_str(f, "blk.0.attn_norm.weight"); put_u32(f, 1); put_u64(f, 16); put_u32(f, 0); put_u64(f, kOff[1]);
    put_str(f, "blk.0.ffn_down_exps.weight"); put_u32(f, 3); put_u64(f, CD); put_u64(f, rows); put_u64(f, E); put_u32(f, 8); put_u64(f, kOff[2]);
    while (ftell(f) % 32) fputc(0, f);
    for (int i = 0; i < 640 + E * R * 34; ++i) fputc(0xAB, f);   // 544 + 32 pad, 64 norm, 272 down
    fclose(f);
}

static int bf16_fill(void* ctx, int t, int s, uint8_t* dst, uint64_t len) {   // BF16 weights in [-0.05, 0.05)
    (void) ctx; (void) t; (void) s;
    for (uint64_t i = 0; i + 2 <= len; i += 2) {
        const float v = ((float) (next() % 1000) - 500.0f) * 1e-4f;
        uint32_t u;
        memcpy(&u, &v, 4);
        dst[i] = (uint8_t) (u >> 16);
        dst[i + 1] = (uint8_t) (u >> 24);
    }
    return 0;
}

// A dense shard: attn_qkv (Q8_0, 64 x 3), a norm (F32, kept) and output (Q8_0, 32 x 2) at aligned offsets
static const uint64_t kDOff[3] = {0, 224, 288};
static void write_dense_gguf(const char* path, const uint8_t* qkv, size_t nq, const uint8_t* out, size_t no) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("FAIL: cannot write %s\n", path); exit(1); }
    put(f, "GGUF", 4); put_u32(f, 3); put_u64(f, 3); put_u64(f, 1);
    put_str(f, "general.alignment"); put_u32(f, 4); put_u32(f, 32);
    put_str(f, "blk.0.attn_qkv.weight"); put_u32(f, 2); put_u64(f, 64); put_u64(f, 3); put_u32(f, 8); put_u64(f, kDOff[0]);
    put_str(f, "blk.0.attn_norm.weight"); put_u32(f, 1); put_u64(f, 16); put_u32(f, 0); put_u64(f, kDOff[1]);
    put_str(f, "output.weight"); put_u32(f, 2); put_u64(f, 32); put_u64(f, 2); put_u32(f, 8); put_u64(f, kDOff[2]);
    while (ftell(f) % 32) fputc(0, f);
    const long d0 = ftell(f);
    put(f, qkv, nq);
    while (ftell(f) < d0 + (long) kDOff[1]) fputc(0, f);
    for (int i = 0; i < 64; ++i) fputc(0xAB, f);
    put(f, out, no);
    fclose(f);
}

int main(void) {
    mkdir("out", 0755);
    // f16: nearest, ties to even, subnormals, overflow
    static const struct { float f; uint16_t h; } kH[] = {
        {1.0f, 0x3C00}, {-2.0f, 0xC000}, {65504.0f, 0x7BFF}, {1.0f / 3, 0x3555}, {0.0f, 0}, {1e-7f, 0x0002},
        {0x1p-24f, 0x0001}, {0x1p-25f, 0x0000}, {0x1.8p-24f, 0x0002}, {0x1.002p0f, 0x3C00}, {0x1.006p0f, 0x3C02},
        {65520.0f, 0x7C00}, {0x1p-14f, 0x0400}, {0x1.ff8p-15f, 0x03FF}, {0x1.ffcp-15f, 0x0400}, {-0x1p-30f, 0x8000}};
    for (size_t i = 0; i < sizeof kH / sizeof kH[0]; ++i)
        CHECK(ns_f32_to_f16(kH[i].f) == kH[i].h, "f16(%g) = %04x, not %04x", kH[i].f, ns_f32_to_f16(kH[i].f), kH[i].h);

    // Q8_0: d = amax / 127 as f16, q = round(x / d); a zero block has d = 0
    float x[64];
    for (int i = 0; i < 64; ++i) x[i] = i < 32 ? sinf(0.37f * i) * 0.01f : 0.0f;
    x[5] = -0.02f;
    uint8_t q[68];
    ns_q8_0_row(x, 64, q);
    const uint16_t d = (uint16_t) (q[0] | q[1] << 8);
    CHECK(d == ns_f32_to_f16(0.02f / 127), "d %04x", d);
    CHECK((int8_t) q[2 + 5] == -127, "amax code %d", (int8_t) q[7]);
    const float id = 1.0f / (0.02f / 127);   // as llama.cpp: the inverse of the f32 step
    for (int i = 0; i < 32; ++i) CHECK((int8_t) q[2 + i] == (int8_t) roundf(x[i] * id), "q[%d] = %d", i, (int8_t) q[2 + i]);
    CHECK(q[34] == 0 && q[35] == 0 && q[36] == 0, "zero block");

    // Two packs with the encodings swapped; the map takes each expert of each projection from one of them
    const char* dirs[2] = {"out/test_emulate_pack", "out/test_emulate_pack4"};
    const char* names[2] = {"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"};
    NsModel m[2];
    const NsModel* pk[2] = {&m[0], &m[1]};
    char err[512] = "";
    for (int i = 0; i < 2; ++i) {
        mkdir(dirs[i], 0755);
        const NsSpec sp[2] = {{names[0], i ? NS_SEED4P4 : NS_SEED4, E, R, CG}, {names[1], i ? NS_SEED4 : NS_SEED4P4, E, R, CD}};
        CHECK(ns_write(dirs[i], sp, 2, 1u << 20, NULL, fill, NULL, err, sizeof err) == 0, "pack %d: %s", i, err);
        CHECK(ns_open(&m[i], dirs[i], err, sizeof err) == 0, "open pack %d: %s", i, err);
    }
    NsRows* nr = (NsRows*) malloc(sizeof *nr);
    ns_rows_init(nr);
    const uint8_t map[3 * E] = {0, 1, 1, 1, 1, 0};   // [proj gate, up, down][expert]: gate 0 1, down 1 0

    const char* path = "out/test_emulate.gguf";
    for (int run = 0; run < 2; ++run) {   // one pack, then two with the map
        write_gguf(path, R);
        int nt = 0;
        CHECK(ns_emulate_gguf(pk, run + 1, run ? map : NULL, sizeof map, path, 2, &nt, err, sizeof err) == 0 && nt == 2,
              "run %d: %d tensors, %s", run, nt, err);
        Gguf g;
        CHECK(gguf_open(&g, path, err, sizeof err) == 0, "reopen: %s", err);
        for (int k = 0; k < 2; ++k) {
            const int cols = k ? CD : CG;
            const uint8_t* got = g.map + g.data + kOff[k ? 2 : 0];
            float w[64], back;
            uint8_t want[68];
            double maxerr = 0;
            for (int e = 0; e < E; ++e)
                for (int r = 0; r < R; ++r) {
                    const int from = run ? map[(k ? 2 : 0) * E + e] : 0;
                    ns_row_f32(nr, ns_find(&m[from], names[k]), e, r, 0, w);
                    ns_q8_0_row(w, cols, want);
                    const uint8_t* row = got + ((size_t) e * R + r) * (cols / 32) * 34;
                    CHECK(!memcmp(row, want, (size_t) (cols / 32) * 34), "run %d tensor %d expert %d row %d", run, k, e, r);
                    for (int c = 0; c < cols; ++c) {   // near the decoded weight
                        const uint8_t* b = row + (c / 32) * 34;
                        const float dd = h2f((uint16_t) (b[0] | b[1] << 8));
                        back = dd * (int8_t) b[2 + c % 32];
                        const double rel = fabs(back - w[c]) / (dd > 0 ? dd : 1);
                        if (rel > maxerr) maxerr = rel;
                    }
                }
            CHECK(maxerr <= 0.5 + 127.0 / 2048, "tensor %d: error %.3f steps", k, maxerr);   // half a step, and d in f16
        }
        const uint8_t* norm = g.map + g.data + kOff[1];
        int kept = 1;
        for (int i = 0; i < 64; ++i) kept &= norm[i] == 0xAB;
        CHECK(kept, "a tensor that is not an expert changed");
        gguf_close(&g);
    }

    // Refused before any write: a shape mismatch, a map entry past the packs, a map too short
    const uint8_t bad[3 * E] = {0, 2, 0, 0, 0, 0};
    for (int c = 0; c < 3; ++c) {
        write_gguf(path, c ? R : R - 1);
        int nt = 0;
        CHECK(ns_emulate_gguf(pk, 2, c ? (c == 1 ? bad : map) : NULL, c == 2 ? 2 * E : sizeof map, path, 1, &nt, err, sizeof err) != 0,
              "case %d accepted", c);
        Gguf g;
        CHECK(gguf_open(&g, path, err, sizeof err) == 0 && g.map[g.data] == 0xAB, "case %d: the file changed", c);
        gguf_close(&g);
    }

    // Dense tensors (D0 of the dense plan): a BF16 pack's rows over a shard's Q8_0 tensors of the same values, by name
    {
        enum { RQ = 3, CQ = 64, RO = 2, CO = 32 };   // attn_qkv (in_proj_qkv), output (lm_head)
        const char* dn[2] = {"language_model.model.layers.0.linear_attn.in_proj_qkv.weight", "language_model.lm_head.weight"};
        const NsSpec sp[2] = {{dn[0], NS_BF16, 1, RQ, CQ}, {dn[1], NS_BF16, 1, RO, CO}};
        NsModel dm;
        mkdir("out/test_emulate_dense", 0755);
        CHECK(ns_write("out/test_emulate_dense", sp, 2, 1u << 20, NULL, bf16_fill, NULL, err, sizeof err) == 0 &&
              ns_open(&dm, "out/test_emulate_dense", err, sizeof err) == 0, "dense pack: %s", err);
        uint8_t qkv[RQ * 2 * 34], out[RO * 34], want[RQ * 2 * 34];
        float w[64];
        for (int r = 0; r < RQ; ++r) {   // the shard: the same weights with 1% noise; want: the pack's own rows
            ns_row_f32(nr, ns_find(&dm, dn[0]), 0, r, 0, w);
            ns_q8_0_row(w, CQ, want + r * 2 * 34);
            for (int c = 0; c < CQ; ++c) w[c] *= 1.0f + 0.01f * (float) ((int) (next() % 3) - 1);
            ns_q8_0_row(w, CQ, qkv + r * 2 * 34);
        }
        for (int r = 0; r < RO; ++r) { ns_row_f32(nr, ns_find(&dm, dn[1]), 0, r, 0, w); ns_q8_0_row(w, CO, out + r * 34); }
        const char* dp = "out/test_emulate_dense.gguf";
        for (int c = 0; c < 2; ++c) {   // as is, then with the rows of attn_qkv reversed: refused, the file unchanged
            uint8_t q2[sizeof qkv];
            for (int r = 0; r < RQ; ++r) memcpy(q2 + r * 68, qkv + (c ? RQ - 1 - r : r) * 68, 68);
            write_dense_gguf(dp, q2, sizeof q2, out, sizeof out);
            int nt = 0;
            const int rc = ns_emulate_dense(&dm, dp, &nt, err, sizeof err);
            Gguf g;
            CHECK(gguf_open(&g, dp, err + 256, 256) == 0, "reopen the dense shard");
            const uint8_t* at = g.map + g.data;
            if (!c) CHECK(rc == 0 && nt == 2 && !memcmp(at, want, sizeof want) && !memcmp(at + kDOff[2], out, sizeof out) && at[kDOff[1]] == 0xAB,
                          "dense: rc %d, %d tensors, %s", rc, nt, err);
            else CHECK(rc != 0 && strstr(err, "row order") && !memcmp(at, q2, sizeof q2), "reversed rows: rc %d, %s", rc, err);
            gguf_close(&g);
        }
        ns_close(&dm);
    }

    for (int i = 0; i < 2; ++i) ns_close(&m[i]);
    free(nr);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
