// tests/test_gguf.c - the GGUF reader (nslm/gguf.h) on a small file written here: key-value skipping, alignment,
// tensor lookup, F32 access with a size check, and truncated files.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "gguf.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void put(FILE* f, const void* p, size_t n) { fwrite(p, 1, n, f); }
static void put_u32(FILE* f, uint32_t v) { put(f, &v, 4); }
static void put_u64(FILE* f, uint64_t v) { put(f, &v, 8); }
static void put_str(FILE* f, const char* s) { put_u64(f, strlen(s)); put(f, s, strlen(s)); }

// A GGUF v3 file shaped like a llama.cpp imatrix: 2 experts x 3 inputs, alignment 64.
static long write_file(const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("FAIL: cannot write %s\n", path); exit(1); }
    put(f, "GGUF", 4); put_u32(f, 3); put_u64(f, 2); put_u64(f, 4);
    put_str(f, "general.type"); put_u32(f, 8); put_str(f, "imatrix");
    put_str(f, "imatrix.datasets"); put_u32(f, 9); put_u32(f, 8); put_u64(f, 2); put_str(f, "a.txt"); put_str(f, "b.txt");
    put_str(f, "imatrix.chunk_size"); put_u32(f, 4); put_u32(f, 512);
    put_str(f, "general.alignment"); put_u32(f, 4); put_u32(f, 64);
    put_str(f, "blk.0.ffn_down_exps.weight.in_sum2"); put_u32(f, 2); put_u64(f, 3); put_u64(f, 2); put_u32(f, 0); put_u64(f, 0);
    put_str(f, "blk.0.ffn_down_exps.weight.counts"); put_u32(f, 2); put_u64(f, 1); put_u64(f, 2); put_u32(f, 0); put_u64(f, 64);
    while (ftell(f) % 64) fputc(0, f);
    const float s[6] = {1, 2, 3, 4, 5, 6}, c[2] = {10, 20};
    put(f, s, sizeof s);
    for (int i = 0; i < 64 - (int) sizeof s; ++i) fputc(0, f);
    put(f, c, sizeof c);
    const long n = ftell(f);
    fclose(f);
    return n;
}

int main(void) {
    mkdir("out", 0755);   // a fresh checkout has no out/
    const char* path = "out/test_gguf.gguf";
    const long size = write_file(path);
    Gguf g;
    char err[256] = "";
    CHECK(gguf_open(&g, path, err, sizeof err) == 0, "open: %s", err);
    CHECK(g.n == 2, "tensors %d", g.n);
    const GgufTensor* t = gguf_find(&g, "blk.0.ffn_down_exps.weight.in_sum2");
    CHECK(t && t->ndim == 2 && t->dims[0] == 3 && t->dims[1] == 2 && t->type == GGUF_F32, "in_sum2 info");
    const float* s = gguf_f32(&g, "blk.0.ffn_down_exps.weight.in_sum2", 6);
    CHECK(s && s[0] == 1 && s[5] == 6, "in_sum2 data");
    const float* c = gguf_f32(&g, "blk.0.ffn_down_exps.weight.counts", 2);
    CHECK(c && c[0] == 10 && c[1] == 20, "counts data (aligned offset)");
    CHECK(!gguf_f32(&g, "blk.0.ffn_down_exps.weight.counts", 3), "element count must match");
    CHECK(!gguf_find(&g, "blk.1.ffn_down_exps.weight.counts"), "missing name found");
    gguf_close(&g);

    const long cut[2] = {size - 4, 40};   // the last tensor cut short, then the header cut off
    for (int i = 0; i < 2; ++i) {
        FILE* in = fopen(path, "rb"), *out = fopen("out/test_gguf_cut.gguf", "wb");
        if (!in || !out) { printf("FAIL: cannot copy %s\n", path); return 1; }
        for (long k = 0; k < cut[i]; ++k) fputc(fgetc(in), out);
        fclose(in);
        fclose(out);
        const int rc = gguf_open(&g, "out/test_gguf_cut.gguf", err, sizeof err);
        if (i == 0) {
            CHECK(rc == 0 && !gguf_f32(&g, "blk.0.ffn_down_exps.weight.counts", 2), "short tensor accepted");
            if (!rc) gguf_close(&g);
        } else {
            CHECK(rc != 0, "cut header accepted");
        }
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
