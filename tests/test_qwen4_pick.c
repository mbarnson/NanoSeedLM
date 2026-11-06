// tests/test_qwen4_pick.c - eng_generate_pick (engine/qwen4_ext.h) with MTP drafts on tests/data/qwen4_tiny, through
// engine/engine.c (CUDA or Metal): a greedy pick gives eng_generate's tokens without drafts, a stop ends the call before
// its token (the history holds what was committed), and a seeded sampler (harness/sample.h) repeats. Skipped without a GPU.
#define _DEFAULT_SOURCE   // glibc: setenv / unsetenv under -std=c99 (Windows: _putenv_s)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../harness/sample.h"
#include "engine_api.h"
#include "format.h"
#include "qwen4_ext.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)
#define NT 40

static Eng* open_k(int k, char* err, int errlen) {   // the tiny model with Q4E_DRAFTS = k (0: unset)
    char v[16];
    snprintf(v, sizeof v, "%d", k);
#ifdef _WIN32
    _putenv_s("Q4E_DRAFTS", k ? v : "");
#else
    if (k) setenv("Q4E_DRAFTS", v, 1); else unsetenv("Q4E_DRAFTS");
#endif
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = "tests/data/qwen4_tiny";
    o.max_seqs = 1;
    o.kv_tokens = 256;
    return eng_open(&o, err, errlen);
}
static int32_t pick_max(void* ctx, const float* l, int V, int32_t draft) {
    (void) ctx; (void) draft;
    int b = 0;
    for (int v = 1; v < V; ++v) if (l[v] > l[b]) b = v;
    return b;
}
typedef struct { int left; } Stop;   // the pick after `left` tokens is a stop
static int32_t pick_stop(void* ctx, const float* l, int V, int32_t draft) {
    Stop* s = (Stop*) ctx;
    return s->left-- > 0 ? pick_max(NULL, l, V, draft) : -1;
}

int main(void) {
    char err[512] = "";
    StFile st;
    if (st_open_file(&st, "tests/data/qwen4_tiny_golden.safetensors", err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
    const int32_t* ids = (const int32_t*) st_data(&st, st_find(&st, "ids"));
    Eng* e0 = open_k(0, err, sizeof err);   // the engine's own open fails without a GPU
    if (!e0 && strstr(err, "device")) { printf("SKIP: %s\n", err); return 77; }
    Eng* e3 = e0 ? open_k(3, err, sizeof err) : NULL;
    CHECK(e0 && e3, "open: %s", err);
    if (!e0 || !e3) { printf("test_qwen4_pick: FAIL\n"); return 1; }
    const int seq = 0;
    int32_t plain[NT], got[NT], a[NT], b[NT];

    // Greedy: eng_generate without drafts, against eng_generate_pick with 3 drafts in calls of 4
    CHECK(eng_prefill(e0, 0, ids, 20) == 0 && eng_generate(e0, &seq, 1, NT, ENG_MODE_AR, plain, NULL) == 0, "greedy without drafts");
    CHECK(eng_prefill(e3, 0, ids, 20) == 0, "prefill");
    int n = 0;
    while (n < NT) {
        const int c = eng_generate_pick(e3, 0, NT - n < 4 ? NT - n : 4, pick_max, NULL, got + n, NULL);
        if (c <= 0) break;
        n += c;
    }
    CHECK(n == NT && !memcmp(got, plain, sizeof plain), "greedy picks with drafts: %d tokens, or other tokens", n);

    // A stop after 13 tokens: 13 committed, and the history ends there
    Stop s = {13};
    CHECK(eng_prefill(e3, 0, ids, 20) == 0 && eng_generate_pick(e3, 0, NT, pick_stop, &s, got, NULL) == 13 && eng_len(e3, 0) == 33 &&
          !memcmp(got, plain, 13 * sizeof(int32_t)), "a stop after 13 tokens");

    // A seeded sampler with drafts repeats; at temperature 0 it is greedy decode
    Sampler sa = {1.0, 0.95, 0, 20, 0, 7}, sb = sa, sg = {0, 1, 0, 0, 0, 7};
    CHECK(eng_prefill(e3, 0, ids, 20) == 0 && eng_generate_pick(e3, 0, NT, sample_engpick, &sa, a, NULL) == NT &&
          eng_prefill(e3, 0, ids, 20) == 0 && eng_generate_pick(e3, 0, NT, sample_engpick, &sb, b, NULL) == NT && !memcmp(a, b, sizeof a),
          "a seeded sampler with drafts differs between runs");
    CHECK(eng_prefill(e3, 0, ids, 20) == 0 && eng_generate_pick(e3, 0, NT, sample_engpick, &sg, got, NULL) == NT && !memcmp(got, plain, sizeof plain),
          "the sampler at temperature 0 is not greedy decode");
    CHECK(eng_generate_pick(e3, 0, 1, NULL, NULL, got, NULL) == -1, "no pick");

    eng_close(e3);
    eng_close(e0);
    st_close_file(&st);
    printf("test_qwen4_pick: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
