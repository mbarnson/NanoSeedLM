// tests/test_model_st.c - model folders: write shards and index, read them back, page alignment, rejections.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#include "json.h"
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
    {"model.layers.3.self_attn.q_proj.weight", NS_SEED6P8, 1, 64, 256},
    {"model.layers.3.mlp.experts.down_proj.weight", NS_SEED4, 4, 64, 128},
    {"model.layers.3.self_attn.v_experts.weight", NS_Q4, 3, 64, 128},
    {"model.layers.3.mlp.shared_experts.up_proj.weight", NS_BF16, 2, 32, 64},
    {"lm_head.weight", NS_Q8, 1, 1000, 128},
    {"model.layers.3.self_attn.mla.kv_a_x", NS_Q8, 1, 64, 128},     // encoded, not named *.weight (MLA projections)
    {"model.layers.3.self_attn.mla.q_lat", NS_Q4, 4, 64, 128},
};
#define NSPEC ((int) (sizeof SPECS / sizeof SPECS[0]))

// File helpers for the rejection cases (no shell): copy a flat folder, remove it, corrupt its first shard.
static int slurp_file(const char* path, uint8_t** data, long* len) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    *data = (uint8_t*) malloc((size_t) *len + 1);
    const size_t got = fread(*data, 1, (size_t) *len, f);
    fclose(f);
    return got == (size_t) *len ? 0 : -1;
}
static int spit_file(const char* path, const uint8_t* data, long len) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    const size_t w = fwrite(data, 1, (size_t) len, f);
    return fclose(f) || w != (size_t) len ? -1 : 0;
}
static int remove_dir(const char* d) {
    DIR* dd = opendir(d);
    if (!dd) return 0;
    char p[1024];
    for (struct dirent* e; (e = readdir(dd));) {
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "%s/%s", d, e->d_name);
        remove(p);
    }
    closedir(dd);
    return rmdir(d);
}
static int copy_dir(const char* src, const char* dst) {
    remove_dir(dst);
    mkdir(dst, 0755);
    DIR* dd = opendir(src);
    if (!dd) return -1;
    char a[1024], b[1024];
    int rc = 0;
    for (struct dirent* e; (e = readdir(dd));) {
        if (e->d_name[0] == '.') continue;
        snprintf(a, sizeof a, "%s/%s", src, e->d_name);
        snprintf(b, sizeof b, "%s/%s", dst, e->d_name);
        uint8_t* data = NULL;
        long len;
        if (slurp_file(a, &data, &len) || spit_file(b, data, len)) rc = -1;
        free(data);
    }
    closedir(dd);
    return rc;
}
// truncate = 1: drop the last 100 bytes of the first shard; else turn its header's U32 dtypes into I32.
static int edit_shard1(const char* d, int truncate) {
    char p[1024] = "";
    DIR* dd = opendir(d);
    if (!dd) return -1;
    for (struct dirent* e; (e = readdir(dd));)
        if (!strncmp(e->d_name, "model-00001-", 12)) snprintf(p, sizeof p, "%s/%s", d, e->d_name);
    closedir(dd);
    uint8_t* data = NULL;
    long len;
    if (!p[0] || slurp_file(p, &data, &len)) { free(data); return -1; }
    if (truncate) len -= 100;
    else {
        uint64_t hn;
        memcpy(&hn, data, 8);
        const char* pat = "\"dtype\": \"U32\"";
        const size_t pl = strlen(pat);
        for (uint64_t i = 8; i + pl <= 8 + hn; ++i)
            if (!memcmp(data + i, pat, pl)) data[i + 10] = 'I';
    }
    const int rc = spit_file(p, data, len);
    free(data);
    return rc;
}

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

    // rejections: a truncated shard, an unexpected dtype
    char path[512];
    snprintf(path, sizeof path, "%s_bad", dir);
    CHECK(copy_dir(dir, path) == 0, "setup");
    CHECK(edit_shard1(path, 1) == 0, "setup");
    CHECK(ns_open(&m, path, err, sizeof err) != 0, "truncated shard accepted");
    CHECK(copy_dir(dir, path) == 0, "setup");
    CHECK(edit_shard1(path, 0) == 0, "setup");
    CHECK(ns_open(&m, path, err, sizeof err) != 0, "wrong dtype accepted");
    CHECK(remove_dir(path) == 0, "cleanup");

    // Q8 / Q4 need whole 64-element groups per row: an MLA rank of 96 (v_up [8][128][96]) is refused at write time
    for (int enc = NS_Q8; enc <= NS_Q4; ++enc) {
        const NsSpec odd = {"model.layers.0.self_attn.mla.v_up", enc, 8, 128, 96};
        order = -1;
        snprintf(path, sizeof path, "%s_odd", dir);
        mkdir(path, 0755);
        CHECK(ns_write(path, &odd, 1, 1u << 20, NULL, fill, &order, err, sizeof err) != 0, "enc %d: 96 columns written", enc);
        remove_dir(path);
    }

    // NS_RAW: a tensor copied as it is (any dtype and shape), beside a seed tensor; read back as plain safetensors
    {
        const NsSpec two[2] = {{"vision.patch_embed.proj.weight", NS_RAW, 1, 1, 1}, {"layers.0.mlp.switch_mlp.up_proj.weight", NS_SEED4P4, 2, 16, 64}};
        const NsRaw raw[2] = {{"F32", 5, {2, 3, 1, 1, 4}, 2 * 3 * 4 * 4}, {NULL, 0, {0}, 0}};
        order = -1;
        snprintf(path, sizeof path, "%s_raw", dir);
        mkdir(path, 0755);
        CHECK(ns_write(path, two, 1, 1u << 20, NULL, fill, &order, err, sizeof err) != 0, "raw tensor without NsRaw written");
        CHECK(ns_write_raw(path, two, raw, 2, 1u << 20, NULL, fill, &order, err, sizeof err) == 0, "raw write: %s", err);
        char shard[2100];
        snprintf(shard, sizeof shard, "%s/model-00001-of-00001.safetensors", path);
        StFile st;
        CHECK(st_open_file(&st, shard, err, sizeof err) == 0, "raw read: %s", err);
        const StEntry* e = st_find(&st, "vision.patch_embed.proj.weight");
        CHECK(e && !strcmp(e->dtype, "F32") && e->ndim == 5 && e->shape[4] == 4 && e->end - e->begin == 96, "raw entry");
        if (e) CHECK(st_data(&st, e)[5] == byte_of(0, 0, 5), "raw data");
        st_dontneed(st.map, st.size);   // the pages leave the process; a read loads them again from the file
        if (e) CHECK(st_data(&st, e)[5] == byte_of(0, 0, 5) && st_data(&st, e)[95] == byte_of(0, 0, 95), "raw data after st_dontneed");
        CHECK(st_find(&st, "layers.0.mlp.switch_mlp.up_proj.seeds") && st_find(&st, "layers.0.mlp.switch_mlp.up_proj.codes"),
              "seed streams beside a raw tensor");
        st_close_file(&st);
        remove_dir(path);
    }

    // MLX affine groups (written raw, read as Q4 / Q8): Q4 group 32, Q8 group 64, Q4 group 64; codes U32, scales BF16
    {
        const NsSpec sp[9] = {
            {"a.weight", NS_RAW, 1, 1, 1}, {"a.scales", NS_RAW, 1, 1, 1}, {"a.biases", NS_RAW, 1, 1, 1},
            {"b.weight", NS_RAW, 1, 1, 1}, {"b.scales", NS_RAW, 1, 1, 1}, {"b.biases", NS_RAW, 1, 1, 1},
            {"c.weight", NS_RAW, 1, 1, 1}, {"c.scales", NS_RAW, 1, 1, 1}, {"c.biases", NS_RAW, 1, 1, 1}};
        const NsRaw raw[9] = {
            {"U32", 3, {2, 8, 16}, 2 * 8 * 16 * 4}, {"BF16", 3, {2, 8, 4}, 2 * 8 * 4 * 2}, {"BF16", 3, {2, 8, 4}, 2 * 8 * 4 * 2},
            {"U32", 2, {4, 32}, 4 * 32 * 4}, {"BF16", 2, {4, 2}, 4 * 2 * 2}, {"BF16", 2, {4, 2}, 4 * 2 * 2},
            {"U32", 2, {4, 16}, 4 * 16 * 4}, {"BF16", 2, {4, 2}, 4 * 2 * 2}, {"BF16", 2, {4, 2}, 4 * 2 * 2}};
        order = -1;
        snprintf(path, sizeof path, "%s_groups", dir);
        mkdir(path, 0755);
        CHECK(ns_write_raw(path, sp, raw, 9, 1u << 20, NULL, fill, &order, err, sizeof err) == 0, "groups write: %s", err);
        CHECK(ns_open(&m, path, err, sizeof err) == 0, "groups open: %s", err);
        const NsTensor* a = ns_find(&m, "a.weight");
        const NsTensor* b = ns_find(&m, "b.weight");
        const NsTensor* c = ns_find(&m, "c.weight");
        CHECK(a && a->enc == NS_Q4 && a->group == 32 && a->slices == 2 && a->rows == 8 && a->cols == 128, "Q4 group 32");
        CHECK(b && b->enc == NS_Q8 && b->group == 64 && b->rows == 4 && b->cols == 128, "Q8 group 64");
        CHECK(c && c->enc == NS_Q4 && c->group == 64 && c->rows == 4 && c->cols == 128, "Q4 group 64");
        if (a) CHECK(a->s[1].len == 2 * 8 * 4 * 2 && a->s[1].p[3] == byte_of(1, 0, 3), "Q4 group 32 scales");
        ns_close(&m);
        // c's shapes are Q4 group 64 or Q8 group 32 (an 8-bit PLE table of 160 columns): config.json's entry decides
        char cfg[600];
        snprintf(cfg, sizeof cfg, "%s/config.json", path);
        static const char* q8 = "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"c\": {\"group_size\": 32, \"bits\": 8}}}";
        static const char* wrong = "{\"quantization\": {\"c\": {\"group_size\": 64, \"bits\": 8}}}";
        CHECK(spit_file(cfg, (const uint8_t*) q8, (long) strlen(q8)) == 0, "config write");
        CHECK(ns_open(&m, path, err, sizeof err) == 0, "Q8 group 32 open: %s", err);
        c = ns_find(&m, "c.weight");
        CHECK(c && c->enc == NS_Q8 && c->group == 32 && c->rows == 4 && c->cols == 64, "Q8 group 32 by the config");
        ns_close(&m);
        CHECK(spit_file(cfg, (const uint8_t*) wrong, (long) strlen(wrong)) == 0, "config write");
        CHECK(ns_open(&m, path, err, sizeof err) != 0 && strstr(err, "c:"), "a config that the shapes contradict: %s", err);
        remove_dir(path);
    }
    {   // crafted files: each is refused, and none reads past the file (ASan) or overflows (UBSan)
        static const struct { const char* name; const char* head; long hn; } bad[] = {
            {"shorter than the length", "", -1},
            {"a header longer than the file", "{}", 1 << 20},
            {"a header without its end", "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8", 0},
            {"a negative shape", "{\"a\":{\"dtype\":\"BF16\",\"shape\":[-2,2],\"data_offsets\":[0,8]}}", 0},
            {"an offset past the file", "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,800]}}", 0},
            {"an end near 2^64", "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[1,18446744073709551615]}}", 0},
            {"a shape past 2^40 values", "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1048577,1048577],\"data_offsets\":[0,8]}}", 0},
        };
        char path[512], file[600];
        snprintf(path, sizeof path, "%s_crafted", dir);
        snprintf(file, sizeof file, "%s/model.safetensors", path);
        for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
            mkdir(path, 0755);
            uint8_t buf[256] = {0};
            const size_t h = strlen(bad[i].head);
            const uint64_t hn = bad[i].hn > 0 ? (uint64_t) bad[i].hn : h;
            memcpy(buf, &hn, 8);
            memcpy(buf + 8, bad[i].head, h);
            CHECK(spit_file(file, buf, bad[i].hn < 0 ? 4 : (long) (8 + h + (bad[i].hn ? 0 : 8))) == 0, "write %s", bad[i].name);
            NsModel cm;
            err[0] = 0;
            CHECK(ns_open(&cm, path, err, sizeof err) != 0 && err[0], "%s: opened", bad[i].name);
            remove_dir(path);
        }
    }

    {   // the index and config parser on truncated literals, each in a heap buffer of its exact size (ASan sees a past read)
        static const char* cut[] = {"[1,t", "[1,tr", "{\"a\":fals", "[nul", "tru"};
        for (size_t i = 0; i < sizeof cut / sizeof cut[0]; ++i) {
            const size_t n = strlen(cut[i]);
            char* b = (char*) malloc(n);
            memcpy(b, cut[i], n);
            Json* j = json_parse(b, n, NULL, 0);
            CHECK(!j, "truncated JSON %s parsed", cut[i]);
            json_free(j);
            free(b);
        }
    }

    printf("test_model_st: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
