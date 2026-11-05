// tests/test_qwen4_metal.c - the qwen4 Metal engine (engine/qwen4_gpu.m) on tests/data/qwen4_tiny against mlx-vlm's
// float32 logits, with test_qwen4_engine.c's limits: matvec rows (any split, within BF16 rounding), one GEMM row of 48, the
// same logits twice (the QSA selection in block order), and a prompt of 199 rows in one forward or three, bit for bit.
// Then tests/data/qwen4_tiny_attn (strong attention) against nslm/qwen4_ref.h, with test_qwen4_engine.c's bounds.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "qwen4_eng.h"
#include "qwen4_ref.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int argmax(const float* v, int n) {
    int a = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[a]) a = i;
    return a;
}

int main(void) {
    char err[512] = "";
    StFile st;
    if (st_open_file(&st, "tests/data/qwen4_tiny_golden.safetensors", err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
    const StEntry* ei = st_find(&st, "ids");
    const StEntry* el = st_find(&st, "logits");
    const int n = (int) ei->shape[0], V = (int) el->shape[1];
    const int32_t* ids = (const int32_t*) st_data(&st, ei);
    const float* want = (const float*) st_data(&st, el);
    const Q4EOpts o = {256, 0, NULL, 0, 0, 0};
    Q4Eng* e = q4e_open("tests/data/qwen4_tiny", &o, err, sizeof err);
    if (!e) { printf("FAIL: open: %s\n", err); return 1; }
    float* got = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* twice = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* again = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* big = (float*) malloc(sizeof(float) * (size_t) n * V);
    const int rows[8] = {7, 1, 1, 8, 8, 8, 8, 7};
    for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(e, ids + at, rows[i], got + (size_t) at * V) == 0, "forward: %s", q4e_error(e));
    CHECK(q4e_pos(e) == n, "position %d", q4e_pos(e));
    q4e_reset(e);
    for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(e, ids + at, rows[i], twice + (size_t) at * V) == 0, "again: %s", q4e_error(e));
    int ndiff = 0;
    for (size_t j = 0; j < (size_t) n * V; ++j) ndiff += got[j] != twice[j];
    printf("the same rows twice: %d logits differ\n", ndiff);
    CHECK(ndiff == 0, "the logits depend on the run (QSA selection order?)");
    q4e_reset(e);
    for (int at = 0; at < n; at += 8) CHECK(q4e_forward(e, ids + at, n - at < 8 ? n - at : 8, again + (size_t) at * V) == 0, "rows of 8: %s", q4e_error(e));
    q4e_reset(e);
    CHECK(q4e_forward(e, ids, n, big) == 0, "one row of %d: %s", n, q4e_error(e));
    double worst = 0, scale = 0, same = 0, wbig = 0;
    int argmax_diff = 0, argmax_big = 0;
    for (int t = 0; t < n; ++t) {
        for (int v = 0; v < V; ++v) {
            const size_t i = (size_t) t * V + v;
            worst = fmax(worst, fabs((double) got[i] - want[i]));
            scale = fmax(scale, fabs((double) want[i]));
            same = fmax(same, fabs((double) got[i] - again[i]));
            wbig = fmax(wbig, fabs((double) big[i] - want[i]));
        }
        const int b = argmax(want + (size_t) t * V, V);
        argmax_diff += argmax(got + (size_t) t * V, V) != b;
        argmax_big += argmax(big + (size_t) t * V, V) != b;
    }
    int first = -1;   // the first row whose logits depend on the split
    for (int t = 0; t < n && first < 0; ++t)
        for (int v = 0; v < V; ++v) if (got[(size_t) t * V + v] != again[(size_t) t * V + v]) { first = t; break; }
    printf("engine logits: max |Metal - mlx-vlm| %.2e of max |logit| %.3f, argmax differs in %d of %d rows; rows of 8 vs uneven %.1e"
           " (from row %d)\n", worst, scale, argmax_diff, n, same, first);
    printf("one GEMM row of %d: max |Metal - mlx-vlm| %.2e, argmax differs in %d rows\n", n, wbig, argmax_big);
    CHECK(worst < 3e-2 * scale && argmax_diff <= n / 16, "logits differ from mlx-vlm beyond the BF16 matvec rounding");
    // the GEMM rounds every projection's inputs to BF16 too, on top of the BF16 outputs: twice the matvec bound
    CHECK(wbig < 6e-2 * scale && argmax_big <= n / 16, "GEMM-row logits differ from mlx-vlm beyond the BF16 rounding");
    // Unlike CUDA's, Metal's k_mv sums one token's row (its per-lane path) in another order than several tokens' rows, so
    // the BF16 rounding of a projection output can flip with the row count: the split moves logits by BF16 steps, no more
    CHECK(same < 1e-2 * scale, "the row split changes the logits beyond BF16 rounding");
    q4e_close(e);
    {   // qwen4_tiny_attn (tools/qwen4_tiny_attn.py): attention strong enough that zero keys move the C reference's
        // logits by about 1.7 of 1.7. The engine against the reference, in decode rows and in one prompt forward.
        Qwen4Ref* r = qwen4_ref_open("tests/data/qwen4_tiny_attn", err, sizeof err);
        CHECK(r != NULL, "reference: %s", err);
        Q4Eng* ea = r ? q4e_open("tests/data/qwen4_tiny_attn", &o, err, sizeof err) : NULL;
        CHECK(!r || ea != NULL, "open qwen4_tiny_attn: %s", err);
        float* ref = (float*) malloc(sizeof(float) * (size_t) n * V);
        float* sr8 = (float*) malloc(sizeof(float) * (size_t) n * V);   // in decode rows
        float* s1 = (float*) malloc(sizeof(float) * (size_t) n * V);    // in one forward
        if (r && ea && qwen4_ref_forward(r, ids, n, ref) == 0) {
            for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(ea, ids + at, rows[i], sr8 + (size_t) at * V) == 0, "attn rows: %s", q4e_error(ea));
            q4e_reset(ea);
            CHECK(q4e_forward(ea, ids, n, s1) == 0, "attn, one row of %d: %s", n, q4e_error(ea));
            // per row: a near-tie of two QSA blocks can flip under BF16 rounding, and strong attention then moves that
            // row a lot. So at most n / 16 rows may pass the bound, as for the argmax.
            double sr = 0, wr = 0, wg = 0;
            for (size_t i = 0; i < (size_t) n * V; ++i) sr = fmax(sr, fabs((double) ref[i]));
            int ar = 0, ag = 0, br = 0, bg = 0;
            for (int t = 0; t < n; ++t) {
                double er = 0, eg = 0;
                for (int v = 0; v < V; ++v) {
                    const size_t i = (size_t) t * V + v;
                    er = fmax(er, fabs((double) sr8[i] - ref[i]));
                    eg = fmax(eg, fabs((double) s1[i] - ref[i]));
                }
                const int a0 = argmax(ref + (size_t) t * V, V);
                ar += a0 != argmax(sr8 + (size_t) t * V, V);
                ag += a0 != argmax(s1 + (size_t) t * V, V);
                br += er > 3e-2 * sr;
                bg += eg > 6e-2 * sr;
                wr = fmax(wr, er);
                wg = fmax(wg, eg);
            }
            printf("strong attention: max |Metal - C reference| %.2e (rows), %.2e (one GEMM row) of max |logit| %.3f; past the bound in %d, %d of"
                   " %d rows; argmax differs in %d, %d\n", wr, wg, sr, br, bg, n, ar, ag);
            CHECK(br <= n / 16 && ar <= n / 16, "strong-attention logits differ from the C reference beyond the BF16 matvec rounding");
            CHECK(bg <= n / 16 && ag <= n / 16, "strong-attention GEMM-row logits differ from the C reference beyond the BF16 rounding");
        } else CHECK(!r || !ea, "reference forward: %s", r ? qwen4_ref_error(r) : "");
        free(ref); free(sr8); free(s1);
        q4e_close(ea);
        qwen4_ref_close(r);
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
    free(got); free(twice); free(again); free(big);
    st_close_file(&st);
    printf("test_qwen4_metal: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
