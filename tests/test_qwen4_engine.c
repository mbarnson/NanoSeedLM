// tests/test_qwen4_engine.c - the qwen4 CUDA engine (engine/qwen4_eng.h) on tests/data/qwen4_tiny against mlx-vlm's
// float32 logits in uneven rows; rows of 8 and the smallest expert cache must give the same logits.  Skipped without a GPU.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "format.h"
#include "qwen4_eng.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { printf("SKIP: no CUDA device\n"); return 77; }
    char err[512] = "";
    StFile st;
    if (st_open_file(&st, "tests/data/qwen4_tiny_golden.safetensors", err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
    const StEntry* ei = st_find(&st, "ids");
    const StEntry* el = st_find(&st, "logits");
    const int n = (int) ei->shape[0], V = (int) el->shape[1];
    const int32_t* ids = (const int32_t*) st_data(&st, ei);
    const float* want = (const float*) st_data(&st, el);
    const Q4EOpts o = {256, 0, NULL, 0, 0};
    Q4Eng* e = q4e_open("tests/data/qwen4_tiny", &o, err, sizeof err);
    if (!e) { printf("FAIL: open: %s\n", err); return 1; }
    float* got = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* again = (float*) malloc(sizeof(float) * (size_t) n * V);
    const int rows[8] = {7, 1, 1, 8, 8, 8, 8, 7};
    for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(e, ids + at, rows[i], got + (size_t) at * V) == 0, "forward: %s", q4e_error(e));
    CHECK(q4e_pos(e) == n, "position %d", q4e_pos(e));
    q4e_reset(e);
    CHECK(q4e_forward(e, ids, n, again) == 0, "forward after reset: %s", q4e_error(e));
    double worst = 0, scale = 0, same = 0;
    int argmax_diff = 0;
    for (int t = 0; t < n; ++t) {
        int a = 0, b = 0;
        for (int v = 0; v < V; ++v) {
            const size_t i = (size_t) t * V + v;
            worst = fmax(worst, fabs((double) got[i] - want[i]));
            scale = fmax(scale, fabs((double) want[i]));
            same = fmax(same, fabs((double) got[i] - again[i]));
            if (got[i] > got[(size_t) t * V + a]) a = v;
            if (want[i] > want[(size_t) t * V + b]) b = v;
        }
        argmax_diff += a != b;
    }
    printf("engine logits: max |CUDA - mlx-vlm| %.2e of max |logit| %.3f, argmax differs in %d of %d rows; rows of 8 vs uneven %.1e\n",
           worst, scale, argmax_diff, n, same);
    CHECK(worst < 3e-2 * scale && argmax_diff <= n / 16, "logits differ from mlx-vlm beyond the BF16 matvec rounding");
    CHECK(same < 1e-5 * scale, "the row split changes the logits");
    q4e_close(e);
    for (int stream = 0; stream < 2; ++stream) {   // the smallest cache (one layer's experts for all), pinned or from the files
    const Q4EOpts oc = {256, 0, NULL, 1e-9, stream};
    e = q4e_open("tests/data/qwen4_tiny", &oc, err, sizeof err);
    CHECK(e != NULL, "open with a cache: %s", err);
    if (e) {
        CHECK(q4e_forward(e, ids, n, again) == 0, "forward with a cache: %s", q4e_error(e));
        double dc = 0;
        for (size_t i = 0; i < (size_t) n * V; ++i) dc = fmax(dc, fabs((double) got[i] - again[i]));
        uint64_t hits, misses;
        int slots;
        q4e_cache_stats(e, &hits, &misses, &slots);
        printf("expert cache of %d slots%s: %llu hits, %llu misses; logits against no cache %.1e\n", slots, stream ? ", streamed" : "",
               (unsigned long long) hits, (unsigned long long) misses, dc);
        CHECK(dc == 0 && misses > 0 && slots == q4e_cfg(e)->n_exp, "the expert cache changes the logits, or never misses");
        q4e_close(e);
    }
    }
    free(got);
    free(again);
    st_close_file(&st);
    printf("test_qwen4_engine: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
