// tests/test_qwen4_engine.c - the qwen4 CUDA engine on tests/data/qwen4_tiny against mlx-vlm's float32 logits: matvec
// rows (any split, the same logits, CUDA graphs or not) and one GEMM row of 48, with and without the smallest expert
// cache.  Also kc_bucket for qwen4's 512 experts (the tiny model has 8).  Skipped without a GPU.
#define _DEFAULT_SOURCE   // glibc: setenv / unsetenv under -std=c99 (Windows: _putenv_s)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "format.h"
#include "kernels_cuda.h"
#include "qwen4_eng.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void set_env(const char* k, const char* v) {   // v = NULL: unset
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

// kc_bucket of rows x k skewed selections over n experts against the host grouping: pairs by expert, tiles of <= MMT_BN
static void bucket_check(int n, int rows, int k) {
    const int P = rows * k, mt = P / MMT_BN + n;
    int32_t* inds = (int32_t*) malloc(4 * (size_t) P);
    uint32_t x = 12345;
    for (int i = 0; i < P; ++i) { x = x * 1664525u + 1013904223u; inds[i] = (int) ((x >> 8) % (uint32_t) n * ((x >> 4) & 1)); }   // half on expert 0
    int32_t *di, *dperm, *dnt;
    MmTile* dtiles;
    cudaMalloc((void**) &di, 4 * (size_t) P); cudaMalloc((void**) &dperm, 4 * (size_t) P);
    cudaMalloc((void**) &dnt, 4 * (size_t) (2 * n + 3)); cudaMalloc((void**) &dtiles, sizeof(MmTile) * (size_t) mt);
    cudaMemcpy(di, inds, 4 * (size_t) P, cudaMemcpyHostToDevice);
    kc_bucket(0, di, P, n, dperm, dtiles, dnt);
    int32_t* perm = (int32_t*) malloc(4 * (size_t) P);
    MmTile* tiles = (MmTile*) malloc(sizeof(MmTile) * (size_t) mt);
    int nt = -1;
    cudaMemcpy(perm, dperm, 4 * (size_t) P, cudaMemcpyDeviceToHost);
    cudaMemcpy(tiles, dtiles, sizeof(MmTile) * (size_t) mt, cudaMemcpyDeviceToHost);
    cudaMemcpy(&nt, dnt, 4, cudaMemcpyDeviceToHost);
    int bad = 0, at = 0, t = 0;
    for (int ex = 0; ex < n; ++ex) {
        const int start = at;
        for (int i = 0; i < P; ++i) if (inds[i] == ex) bad += perm[at++] != i;
        for (int s = start; s < at; s += MMT_BN, ++t)
            bad += t >= nt || tiles[t].slice != ex || tiles[t].start != s || tiles[t].count != (at - s < MMT_BN ? at - s : MMT_BN);
    }
    CHECK(cudaGetLastError() == cudaSuccess && nt == t && bad == 0, "kc_bucket of %d pairs over %d experts: %d tiles (want %d), %d wrong", P, n, nt, t, bad);
    cudaFree(di); cudaFree(dperm); cudaFree(dnt); cudaFree(dtiles);
    free(inds); free(perm); free(tiles);
}

int main(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { printf("SKIP: no CUDA device\n"); return 77; }
    bucket_check(512, 128, 10);   // qwen4: 512 experts, top 10, a prompt chunk of 128 rows
    bucket_check(128, 64, 8);     // MoVA
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
    {   // the decode rows above ran as CUDA graphs: without them, the same logits
        set_env("NSLM_NO_GRAPH", "1");
        Q4Eng* ng = q4e_open("tests/data/qwen4_tiny", &o, err, sizeof err);
        set_env("NSLM_NO_GRAPH", NULL);
        CHECK(ng != NULL, "open without graphs: %s", err);
        int diff = 0;
        for (int i = 0, at = 0; ng && i < 8; at += rows[i++]) {
            CHECK(q4e_forward(ng, ids + at, rows[i], again + (size_t) at * V) == 0, "forward without graphs: %s", q4e_error(ng));
            for (size_t j = (size_t) at * V; j < (size_t) (at + rows[i]) * V; ++j) diff += got[j] != again[j];
        }
        printf("decode rows with and without CUDA graphs: %d logits differ\n", diff);
        CHECK(diff == 0, "CUDA graphs change the logits of decode rows");
        q4e_close(ng);
    }
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
    // the GEMM rounds every projection's inputs to BF16 too, on top of the BF16 outputs: twice the matvec bound
    CHECK(wbig < 6e-2 * scale && argmax_big <= n / 16, "GEMM-row logits differ from mlx-vlm beyond the BF16 rounding");
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
        q4e_reset(e);   // decode rows (matvecs) through the cache
        for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(e, ids + at, rows[i], again + (size_t) at * V) == 0, "rows with a cache: %s", q4e_error(e));
        double dr = 0;
        for (size_t i = 0; i < (size_t) n * V; ++i) dr = fmax(dr, fabs((double) got[i] - again[i]));
        printf("decode rows with the cache%s: logits against no cache %.1e\n", stream ? ", streamed" : "", dr);
        CHECK(dr == 0, "the expert cache changes the logits of decode rows");
        q4e_close(e);
    }
    }
    {   // a prompt of more than Q4E_LOGIT_ROWS rows in one forward, or in three: the next token's logits bit for bit
        const int L = 199;
        int32_t* lid = (int32_t*) malloc(sizeof(int32_t) * (L + 1));
        for (int i = 0; i <= L; ++i) lid[i] = ids[i % n];
        float* one = (float*) malloc(sizeof(float) * (size_t) V);
        float* three = (float*) malloc(sizeof(float) * (size_t) V);
        e = q4e_open("tests/data/qwen4_tiny", &o, err, sizeof err);
        CHECK(e != NULL, "open: %s", err);
        if (e) {
            CHECK(q4e_forward(e, lid, L, NULL) == 0 && q4e_forward(e, lid + L, 1, one) == 0, "one forward of %d rows: %s", L, q4e_error(e));
            q4e_reset(e);
            CHECK(q4e_forward(e, lid, 64, NULL) == 0 && q4e_forward(e, lid + 64, 64, NULL) == 0 && q4e_forward(e, lid + 128, L - 128, NULL) == 0 &&
                  q4e_forward(e, lid + L, 1, three) == 0, "three forwards: %s", q4e_error(e));
            int diff = 0;
            for (int v = 0; v < V; ++v) diff += one[v] != three[v];
            printf("a prompt of %d rows in one forward or three: %d logits differ\n", L, diff);
            CHECK(diff == 0, "the prompt's row split changes the next token's logits");
            q4e_close(e);
        }
        free(lid); free(one); free(three);
    }
    {   // more than Q4E_MAX_ROWS prompt rows: one layer-major pass of three chunks, or the three chunks as row-major
        // forwards, bit for bit (a chunk of 8 rows or fewer would take the matvecs, so the last one has 40)
        const int L = 2 * Q4E_MAX_ROWS + 40;
        const Q4EOpts ol = {L + 8, 0, NULL, 0, 0};
        int32_t* lid = (int32_t*) malloc(sizeof(int32_t) * (L + 1));
        for (int i = 0; i <= L; ++i) lid[i] = ids[(i * 7) % n];
        float* pass = (float*) malloc(sizeof(float) * (size_t) V);
        float* rows3 = (float*) malloc(sizeof(float) * (size_t) V);
        e = q4e_open("tests/data/qwen4_tiny", &ol, err, sizeof err);
        CHECK(e != NULL, "open: %s", err);
        if (e) {
            CHECK(q4e_forward(e, lid, L, NULL) == 0 && q4e_forward(e, lid + L, 1, pass) == 0, "a pass of %d rows: %s", L, q4e_error(e));
            q4e_reset(e);
            const int a = Q4E_MAX_ROWS, b = Q4E_MAX_ROWS;
            CHECK(q4e_forward(e, lid, a, NULL) == 0 && q4e_forward(e, lid + a, b, NULL) == 0 && q4e_forward(e, lid + a + b, L - a - b, NULL) == 0 &&
                  q4e_forward(e, lid + L, 1, rows3) == 0, "three forwards: %s", q4e_error(e));
            int diff = 0;
            for (int v = 0; v < V; ++v) diff += pass[v] != rows3[v];
            printf("a prompt of %d rows in one layer-major pass or three forwards: %d logits differ\n", L, diff);
            CHECK(diff == 0, "the layer-major pass changes the next token's logits");
            q4e_close(e);
        }
        free(lid); free(pass); free(rows3);
    }
    free(got);
    free(again);
    free(big);
    st_close_file(&st);
    printf("test_qwen4_engine: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
