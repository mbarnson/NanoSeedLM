// tests/test_qwen4_api.c - engine_api.h over the qwen4 engine (engine/qwen4_api.c) on tests/data/qwen4_tiny, against
// direct qwen4_eng.h calls: steps, extended prompts, rewinds, greedy decode and scoring. Skipped without a GPU.
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "engine_qwen4.h"
#include "engine_api.h"
#include "format.h"
#include "qwen4_eng.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)
#define DIR_ "tests/data/qwen4_tiny"

static int same(const float* a, const float* b, int n) { return !memcmp(a, b, sizeof(float) * (size_t) n); }
static int argmax(const float* l, int n) { int b = 0; for (int v = 1; v < n; ++v) if (l[v] > l[b]) b = v; return b; }

// The logits after the prompt ids[0 .. n) through q4e directly: n - 1 rows, then the last
static void direct(Q4Eng* q, const int32_t* ids, int n, float* lg) {
    q4e_reset(q);
    if (n > 1) q4e_forward(q, ids, n - 1, NULL);
    q4e_forward(q, ids + n - 1, 1, lg);
}

int main(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { printf("SKIP: no CUDA device\n"); return 77; }
    char err[512] = "";
    StFile st;
    if (st_open_file(&st, "tests/data/qwen4_tiny_golden.safetensors", err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
    const StEntry* ei = st_find(&st, "ids");
    const int n = (int) ei->shape[0];
    const int32_t* ids = (const int32_t*) st_data(&st, ei);
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = DIR_;
    o.max_seqs = 2;
    o.kv_tokens = 256;
    CHECK(eng_open(&o, err, sizeof err) == NULL && strstr(err, "one sequence slot"), "two slots open: %s", err);
    o.max_seqs = 1;
    Eng* e = eng_open(&o, err, sizeof err);
    const Q4EOpts qo = {256, 0, NULL, -1, 0, 0};
    Q4Eng* q = e ? q4e_open(DIR_, &qo, err, sizeof err) : NULL;
    CHECK(e && q, "open: %s", err);
    if (!e || !q) { printf("test_qwen4_api: FAIL\n"); return 1; }
    const int V = eng_vocab(e);
    float* a = (float*) malloc(sizeof(float) * (size_t) V * (n + 1));
    float* b = (float*) malloc(sizeof(float) * (size_t) V * (n + 1));
    printf("%s\n", eng_describe(e));

    CHECK(eng_prefill(e, 0, ids, 30) == 0 && eng_step(e, 0, a) == 0, "prefill 30");
    direct(q, ids, 30, b);
    CHECK(same(a, b, V), "prefill and step differ from the engine");
    CHECK(eng_step(e, 0, b) == 0 && same(a, b, V), "a second step without a push changes the logits");
    CHECK(eng_push(e, 0, ids[30]) == 0 && eng_step(e, 0, a) == 0 && q4e_forward(q, ids + 30, 1, b) == 0 && same(a, b, V), "push, step");

    int reused = -1;   // an extended prompt continues the engine (more than 8 new rows: the same GEMM kernels, bit for bit)
    CHECK(eng_prefill(e, 0, ids, 20) == 0 && eng_prefill_cached(e, 0, ids, n, &reused) == 0 && eng_step(e, 0, a) == 0, "extend");
    direct(q, ids, n, b);
    CHECK(reused == 19 && same(a, b, V), "the extended prompt (%d reused) differs from one prefill", reused);
    int32_t other[64];
    memcpy(other, ids, sizeof(int32_t) * (size_t) n);
    other[3] ^= 1;
    CHECK(eng_prefill_cached(e, 0, other, n, &reused) == 0 && reused == 0, "a changed prompt kept %d positions", reused);

    CHECK(eng_prefill(e, 0, ids, n) == 0 && eng_rewind(e, 0, 40) == 0 && eng_len(e, 0) == 40 && eng_step(e, 0, a) == 0, "rewind");
    direct(q, ids, 40, b);
    CHECK(same(a, b, V), "the rewound slot differs from a prefill of 40");

    eng_free(e, 0);   // the prompt in steps of 16 rows from an empty slot
    CHECK(eng_len(e, 0) == 0 && eng_prefill_begin(e, 0, ids, n, NULL) == 0, "begin");
    int left = 1, rounds = 0;
    while (left > 0 && rounds < 99) { left = eng_prefill_next(e, 0, 16); ++rounds; }
    CHECK(left == 0 && eng_step(e, 0, a) == 0, "prefill in steps (%d)", left);
    direct(q, ids, n, b);
    CHECK(rounds == 3 && same(a, b, V), "the stepped prefill (%d rounds) differs from one prefill", rounds);
    CHECK(eng_prefill(e, 0, ids, 20) == 0 && eng_prefill_begin(e, 0, ids, n, &reused) == 0 && reused == 19 &&   // nslm-serve's
          eng_prefill_next(e, 0, INT_MAX) == 0 && eng_step(e, 0, a) == 0 && same(a, b, V), "a kept prefix, then all rows (INT_MAX)");

    int32_t out[6], seq = 0;   // greedy decode against the engine's own argmax loop
    CHECK(eng_prefill(e, 0, ids, 20) == 0 && eng_generate(e, &seq, 1, 6, ENG_MODE_AR, out, NULL) == 0, "generate");
    direct(q, ids, 20, b);
    int ok = 1;
    for (int j = 0; j < 6; ++j) {
        const int32_t t = argmax(b, V);
        ok &= out[j] == t;
        q4e_forward(q, &t, 1, b);
    }
    CHECK(ok && eng_len(e, 0) == 26, "generate differs from greedy decode");

    CHECK(eng_score(e, 0, ids, 10, 20, a) == 0, "score");   // logits predicting ids[10 .. 30)
    q4e_reset(q);
    q4e_forward(q, ids, 9, NULL);
    q4e_forward(q, ids + 9, 20, b);
    CHECK(same(a, b, 20 * V), "score differs from the engine's rows");

    // MTP verify (plan gate MB): 4 decode rows in one forward give each row the logits of 4 one-row forwards, bit for bit
    q4e_reset(q);
    q4e_forward(q, ids, 20, NULL);
    for (int j = 0; j < 4; ++j) q4e_forward(q, ids + 20 + j, 1, a + (size_t) j * V);
    q4e_reset(q);
    q4e_forward(q, ids, 20, NULL);
    CHECK(q4e_forward(q, ids + 20, 4, b) == 0 && same(a, b, 4 * V), "4 decode rows in one forward differ from 4 one-row forwards");
    CHECK(eng_fork(e, 1, 0) == -1 && eng_kv_bytes(e) == 0 && eng_kv_read(e, 0, 0, 1, a) == -1, "one slot, no KV files");
    eng_close(e);
    q4e_close(q);
    free(a); free(b);
    st_close_file(&st);
    printf("test_qwen4_api: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
