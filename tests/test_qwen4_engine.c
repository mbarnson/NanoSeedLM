// tests/test_qwen4_engine.c - the qwen4 CUDA engine on tests/data/qwen4_tiny against mlx-vlm's float32 logits: matvec
// rows (any split, the same logits) and one GEMM row of 48, with and without the smallest expert cache.  Skipped without a GPU.
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
    float* big = (float*) malloc(sizeof(float) * (size_t) n * V);
    const int rows[8] = {7, 1, 1, 8, 8, 8, 8, 7};
    for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(e, ids + at, rows[i], got + (size_t) at * V) == 0, "forward: %s", q4e_error(e));
    CHECK(q4e_pos(e) == n, "position %d", q4e_pos(e));
    q4e_reset(e);
    for (int at = 0; at < n; at += 8) CHECK(q4e_forward(e, ids + at, n - at < 8 ? n - at : 8, again + (size_t) at * V) == 0, "rows of 8: %s", q4e_error(e));
    q4e_reset(e);
    CHECK(q4e_forward(e, ids, n, big) == 0, "one row of %d: %s", n, q4e_error(e));
    double worst = 0, scale = 0, same = 0, wbig = 0;
    int argmax_diff = 0, argmax_big = 0;
    for (int t = 0; t < n; ++t) {
        int a = 0, b = 0;
        for (int v = 0; v < V; ++v) {
            const size_t i = (size_t) t * V + v;
            worst = fmax(worst, fabs((double) got[i] - want[i]));
            scale = fmax(scale, fabs((double) want[i]));
            same = fmax(same, fabs((double) got[i] - again[i]));
            wbig = fmax(wbig, fabs((double) big[i] - want[i]));
            if (got[i] > got[(size_t) t * V + a]) a = v;
            if (want[i] > want[(size_t) t * V + b]) b = v;
        }
        int c = 0;
        for (int v = 1; v < V; ++v) if (big[(size_t) t * V + v] > big[(size_t) t * V + c]) c = v;
        argmax_diff += a != b;
        argmax_big += c != b;
    }
    printf("engine logits: max |CUDA - mlx-vlm| %.2e of max |logit| %.3f, argmax differs in %d of %d rows; rows of 8 vs uneven %.1e\n",
           worst, scale, argmax_diff, n, same);
    printf("one GEMM row of %d: max |CUDA - mlx-vlm| %.2e, argmax differs in %d rows\n", n, wbig, argmax_big);
    CHECK(worst < 3e-2 * scale && argmax_diff <= n / 16, "logits differ from mlx-vlm beyond the BF16 matvec rounding");
    CHECK(wbig < 3e-2 * scale && argmax_big <= n / 16, "GEMM-row logits differ from mlx-vlm beyond the BF16 rounding");
    CHECK(same < 1e-5 * scale, "the row split changes the logits");
    q4e_close(e);
    for (int stream = 0; stream < 2; ++stream) {   // the smallest cache (one layer's experts for all), pinned or from the files
    const Q4EOpts oc = {256, 0, NULL, 1e-9, stream};
    e = q4e_open("tests/data/qwen4_tiny", &oc, err, sizeof err);
    CHECK(e != NULL, "open with a cache: %s", err);
    if (e) {
        CHECK(q4e_forward(e, ids, n, again) == 0, "forward with a cache: %s", q4e_error(e));
        double dc = 0;
        for (size_t i = 0; i < (size_t) n * V; ++i) dc = fmax(dc, fabs((double) big[i] - again[i]));
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
    free(big);
    st_close_file(&st);
    printf("test_qwen4_engine: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
