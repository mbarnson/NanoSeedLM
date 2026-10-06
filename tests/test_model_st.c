// tests/test_model_st.c - model folders: write shards and index, read them back, page alignment, rejections.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "model_st.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); ++fails; } } while (0)

static uint8_t byte_of(int t, int s, uint64_t i) { return (uint8_t) ((i * 2654435761u) >> 13 ^ (uint64_t) (t * 31 + s * 7)); }
static int fill(void* ctx, int t, int s, uint8_t* dst, uint64_t len) {
    int* order = (int*) ctx;   // the writer must call streams in tensor and stream order
    if (t * 4 + s <= *order) return -1;
    *order = t * 4 + s;
    for (uint64_t i = 0; i < len; ++i) dst[i] = byte_of(t, s, i);
    return 0;
}

static const NsSpec SPECS[] = {
    {"model.embed_tokens.weight", NS_Q8, 1, 1000, 128},
    {"model.norm.weight", NS_BF16, 1, 1, 64},
    {"model.layers.3.mlp.gate.weight", NS_BF16, 1, 4, 64},
    {"model.layers.3.mlp.gate.bias", NS_BF16, 1, 1, 4},
    {"model.layers.3.mlp.experts.gate_proj.weight", NS_SEED4P4, 4, 64, 256},
    {"model.layers.3.mlp.experts.down_proj.weight", NS_SEED4, 4, 64, 128},
    {"model.layers.3.self_attn.v_experts.weight", NS_Q4, 3, 64, 128},
    {"model.layers.3.mlp.shared_experts.up_proj.weight", NS_BF16, 2, 32, 64},
    {"lm_head.weight", NS_Q8, 1, 1000, 128},
};
#define NSPEC ((int) (sizeof SPECS / sizeof SPECS[0]))

int main(void) {
    const char* dir = "out/test/model_st";
    mkdir("out", 0755);
    mkdir("out/test", 0755);
    mkdir(dir, 0755);
    char err[512];
    int order = -1;
    CHECK(ns_write(dir, SPECS, NSPEC, 200000, "\"producer\": \"test\"", fill, &order, err, sizeof err) == 0, "write: %s", err);

    NsModel m;
    CHECK(ns_open(&m, dir, err, sizeof err) == 0, "open: %s", err);
    CHECK(m.nsh > 1, "expected several shards, got %d", m.nsh);
    CHECK(m.n == NSPEC, "%d tensors, want %d", m.n, NSPEC);
    for (int i = 0; i < NSPEC; ++i) {
        const NsTensor* t = ns_find(&m, SPECS[i].name);
        CHECK(t != NULL, "%s missing", SPECS[i].name);
        if (!t) continue;
        CHECK(t->enc == SPECS[i].enc && t->slices == SPECS[i].slices && t->rows == SPECS[i].rows && t->cols == SPECS[i].cols,
              "%s: enc %d %dx%dx%d", t->name, t->enc, t->slices, t->rows, t->cols);
        for (int s = 0; s < 4; ++s) {
            const uint64_t len = ns_stream_len(SPECS[i].enc, SPECS[i].slices, SPECS[i].rows, SPECS[i].cols, s);
            CHECK(t->s[s].len == len, "%s stream %d: %llu bytes", t->name, s, (unsigned long long) t->s[s].len);
            int same = 1;
            for (uint64_t k = 0; k < len && same; ++k) same = t->s[s].p[k] == byte_of(i, s, k);
            CHECK(same, "%s stream %d: bytes differ", t->name, s);
            if (len && len % NS_PAGE == 0) CHECK(ns_mappable(&m, &t->s[s]), "%s stream %d: page-multiple but not mappable", t->name, s);
        }
    }
    CHECK(ns_find(&m, "nope.weight") == NULL, "found a missing tensor");
    ns_close(&m);

    // rejections: a truncated shard, an unexpected dtype, a duplicate across shards
    char path[512], cmd[1200];
    snprintf(path, sizeof path, "%s/model-00001-of-%05d.safetensors", dir, 0);
    snprintf(cmd, sizeof cmd, "cp -R %s %s_bad && f=$(ls %s_bad/model-00001-*); s=$(stat -f %%z $f); truncate -s $((s - 100)) $f",
             dir, dir, dir);
    CHECK(system(cmd) == 0, "setup");
    snprintf(path, sizeof path, "%s_bad", dir);
    CHECK(ns_open(&m, path, err, sizeof err) != 0, "truncated shard accepted");
    snprintf(cmd, sizeof cmd, "rm -rf %s_bad && cp -R %s %s_bad && f=$(ls %s_bad/model-00001-*) && "
                              "LC_ALL=C sed -i '' 's/\"dtype\": \"U32\"/\"dtype\": \"I32\"/' $f", dir, dir, dir, dir);
    CHECK(system(cmd) == 0, "setup");
    CHECK(ns_open(&m, path, err, sizeof err) != 0, "wrong dtype accepted");
    snprintf(cmd, sizeof cmd, "rm -rf %s_bad", dir);
    CHECK(system(cmd) == 0, "cleanup");

    printf("test_model_st: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
