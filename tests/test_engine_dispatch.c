// tests/test_engine_dispatch.c - engine/engine.c: eng_open takes the qwen4 engine for tests/data/qwen4_tiny, its calls
// match the qwen4 adapter's bit for bit, and the MoVA extensions refuse it. Skipped without a GPU.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "engine_api.h"
#include "mova_ext.h"
#include "engine_qwen4.h"
#include "engine_api_fns.h"
#include "engine_unname.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)

int main(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { printf("SKIP: no CUDA device\n"); return 77; }
    char err[512] = "";
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = "tests/data/no_such_model";
    o.max_seqs = 1;
    o.kv_tokens = 256;
    CHECK(!eng_open(&o, err, sizeof err) && err[0], "a missing model opens");
    o.model_dir = "tests/data/qwen4_tiny";
    Eng* e = eng_open(&o, err, sizeof err);
    struct Q4Api* q = e ? q4_eng_open(&o, err, sizeof err) : NULL;
    CHECK(e && q, "open: %s", err);
    if (!e || !q) { printf("test_engine_dispatch: FAIL\n"); return 1; }
    CHECK(!strcmp(eng_describe(e), q4_eng_describe(q)), "describe: %s", eng_describe(e));
    const int V = eng_vocab(e);
    float* a = (float*) malloc(sizeof(float) * (size_t) V * 4);
    float* b = (float*) malloc(sizeof(float) * (size_t) V * 4);
    const int32_t ids[12] = {1, 5, 9, 2, 7, 3, 3, 8, 4, 6, 2, 11};
    CHECK(eng_prefill(e, 0, ids, 12) == 0 && eng_step(e, 0, a) == 0, "prefill, step");
    CHECK(q4_eng_prefill(q, 0, ids, 12) == 0 && q4_eng_step(q, 0, b) == 0 && !memcmp(a, b, sizeof(float) * (size_t) V),
          "the step differs from the adapter's");
    CHECK(eng_score(e, 0, ids, 6, 4, a) == 0 && q4_eng_score(q, 0, ids, 6, 4, b) == 0 &&
          !memcmp(a, b, sizeof(float) * (size_t) V * 4), "score differs from the adapter's");
    CHECK(eng_len(e, 0) == q4_eng_len(q, 0) && eng_mova_timing(e, 1) == -1 && eng_mla_capture(e, 1) == -1,
          "length or the MoVA extensions");
    eng_close(e);
    q4_eng_close(q);
    free(a); free(b);
    printf("test_engine_dispatch: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
