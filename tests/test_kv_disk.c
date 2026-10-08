// tests/test_kv_disk.c - the cold KV cache (harness/kv_disk.c): chained block hashes, store / count / load, the model
// fingerprint, invalid files, the byte budget (least recently used blocks go first) and a reopened cache's index.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "kv_disk.h"
#include "model_st.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { BB = 4096, NB = 5 };   // block bytes, blocks

static int fill(void* ctx, int ti, int s, uint8_t* dst, uint64_t len) {
    for (uint64_t i = 0; i < len; ++i) dst[i] = (uint8_t) (i * 7 + *(int*) ctx + ti + s);
    return 0;
}
static int write_model(const char* dir, int seed) {
    mkdir(dir, 0755);
    const NsSpec t = {"model.norm.weight", NS_BF16, 1, 1, 64};
    char err[256];
    return ns_write(dir, &t, 1, 1ull << 20, NULL, fill, &seed, err, sizeof err);
}
static void block_data(uint8_t* p, int b) { for (int i = 0; i < BB; ++i) p[i] = (uint8_t) (b * 31 + i); }

int main(void) {
    char err[256] = "";
    mkdir("out", 0755);
    mkdir("out/test", 0755);
    // the fingerprint: the same folder gives the same one; other weights or another engine description, another
    uint8_t fa[32], fb[32], fc[32];
    CHECK(write_model("out/test/kvd_model", 1) == 0, "model folder");
    CHECK(kvd_fingerprint("out/test/kvd_model", "engine A", 1000, fa) == 0, "fingerprint");
    CHECK(kvd_fingerprint("out/test/kvd_model", "engine B", 1000, fb) == 0 && memcmp(fa, fb, 32), "fingerprint: engine");
    CHECK(write_model("out/test/kvd_model", 2) == 0, "model folder");
    CHECK(kvd_fingerprint("out/test/kvd_model", "engine A", 1000, fc) == 0 && memcmp(fa, fc, 32), "fingerprint: weights");

    const char* dir = "out/test/kvdisk";
    KvDisk* d = kvd_open(dir, 0, fa, BB, err, sizeof err);   // budget 0: disabled
    CHECK(!d, "a zero budget opens no cache");
    d = kvd_open(dir, 3 * (BB + 2048), fa, BB, err, sizeof err);   // three blocks with their headers
    CHECK(d != NULL, "kvd_open: %s", err);
    if (!d) return 1;
    int32_t ids[NB * KVD_BLOCK], other[NB * KVD_BLOCK];
    for (int i = 0; i < NB * KVD_BLOCK; ++i) { ids[i] = i * 13 % 1000; other[i] = ids[i]; }
    other[5] = 999999;   // differs in block 0
    uint8_t h[NB][32], g[NB][32];
    kvd_hashes(d, ids, NB, h);
    kvd_hashes(d, other, NB, g);
    for (int b = 0; b < NB; ++b) CHECK(memcmp(h[b], g[b], 32) != 0, "chained hashes: block %d", b);   // a prefix change
    for (int b = 0; b < NB; ++b) kvd_drop(d, h[b]);   // an earlier run's blocks
    CHECK(kvd_count(d, h, NB) == 0, "empty cache");

    uint8_t src[BB], dst[BB];
    for (int b = 0; b < 2; ++b) { block_data(src, b); CHECK(kvd_store(d, h[b], ids + b * KVD_BLOCK, src) == 0, "store %d", b); }
    CHECK(kvd_count(d, h, NB) == 2, "two blocks counted, %d", kvd_count(d, h, NB));
    CHECK(kvd_count(d, h + 1, NB - 1) == 1, "a chain from block 1");
    block_data(src, 1);
    CHECK(kvd_load(d, h[1], ids + KVD_BLOCK, dst) == 0 && !memcmp(src, dst, BB), "load block 1");
    CHECK(kvd_load(d, h[2], ids + 2 * KVD_BLOCK, dst) != 0, "a missing block");
    {   // another fingerprint sees none of them
        KvDisk* d2 = kvd_open(dir, 3 * (BB + 2048), fc, BB, err, sizeof err);
        CHECK(d2 && kvd_count(d2, h, NB) == 0, "another model's cache");
        uint8_t h2[NB][32];
        if (d2) kvd_hashes(d2, ids, NB, h2);
        CHECK(d2 && memcmp(h2[0], h[0], 32), "hashes rooted in the fingerprint");
        kvd_close(d2);
    }
    // the budget: blocks 2 and 3 push out the least recently used block (0; block 1 was just loaded)
    for (int b = 2; b < 4; ++b) { block_data(src, b); CHECK(kvd_store(d, h[b], ids + b * KVD_BLOCK, src) == 0, "store %d", b); }
    CHECK(kvd_count(d, h, NB) == 0 && kvd_count(d, h + 1, NB - 1) == 3, "LRU eviction: block 0 gone, 1 .. 3 kept");
    CHECK(kvd_used(d) <= 3 * (BB + 2048), "within the budget: %llu bytes", (unsigned long long) kvd_used(d));
    const uint64_t used = kvd_used(d);
    kvd_close(d);
    // reopened: the index from the files
    d = kvd_open(dir, 3 * (BB + 2048), fa, BB, err, sizeof err);
    CHECK(d && kvd_used(d) == used, "reopened: %llu bytes, was %llu", d ? (unsigned long long) kvd_used(d) : 0ull, (unsigned long long) used);
    // an invalid file (truncated) reads as a miss and is deleted
    char path[512];
    kvd_path(d, h[3], path, sizeof path);
    FILE* f = fopen(path, "r+b");
    CHECK(f != NULL, "open %s", path);
    if (f) { fclose(f); f = fopen(path, "wb"); fwrite("short", 1, 5, f); fclose(f); }
    struct stat st;
    CHECK(kvd_load(d, h[3], ids + 3 * KVD_BLOCK, dst) != 0 && stat(path, &st) != 0, "a truncated block is a miss, deleted");
    block_data(src, 2);
    CHECK(kvd_load(d, h[2], ids + 3 * KVD_BLOCK, dst) != 0, "a block whose tokens differ is a miss");
    kvd_close(d);
    printf("test_kv_disk: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
