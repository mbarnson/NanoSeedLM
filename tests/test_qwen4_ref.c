// tests/test_qwen4_ref.c - the qwen4_exp CPU reference (nslm/qwen4_ref.h) against mlx-vlm on a tiny random model
// (tests/data/qwen4_tiny, from tools/qwen4_golden.py): the configuration, the PLE n-gram tables and rows, the logits.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "qwen4_ops.h"
#include "qwen4_ref.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char* DIR_ = "tests/data/qwen4_tiny";

static const int64_t* i64(const StFile* f, const char* name, int64_t n) {
    const StEntry* e = st_find(f, name);
    return e && !strcmp(e->dtype, "I64") && e->end - e->begin == (uint64_t) n * 8 ? (const int64_t*) st_data(f, e) : NULL;
}

int main(void) {
    char err[512] = "";
    Qwen4Cfg c;
    CHECK(qwen4_cfg_load(&c, DIR_, err, sizeof err) == 0, "config: %s", err);
    CHECK(c.n_layer == 4 && c.d == 64 && c.hc == 4 && c.rot_dim == 8 && c.eos == 7 && c.attn[3] && !c.attn[0], "geometry");
    CHECK(c.ple[1] == 0 && c.ple[0] == -1 && c.ple[2] == -1, "PLE layer (ple_layer_ids [2] is layer 1)");
    CHECK(qwen4_cfg_parse(&c, "{\"text_config\": {\"model_type\": \"qwen3_5_moe\"}}", err, sizeof err) != 0, "a non-qwen4 config parsed");
    qwen4_cfg_load(&c, DIR_, err, sizeof err);

    // the n-gram tables against the folder's own (mlx-vlm computes and stores them)
    StFile st;
    char path[512];
    snprintf(path, sizeof path, "%s/model.safetensors", DIR_);
    CHECK(st_open_file(&st, path, err, sizeof err) == 0, "%s", err);
    Qwen4Ngram g;
    qwen4_ngram_init(&c, 0, &g);
    const char* pre = "language_model.model.layers.1.ple.ple_embedding.";
    char nm[256];
    snprintf(nm, sizeof nm, "%slayer_multipliers", pre);
    const int64_t* mult = i64(&st, nm, c.ngram);
    snprintf(nm, sizeof nm, "%sngram_heads_vocab_sizes", pre);
    const int64_t* size = i64(&st, nm, g.heads);
    snprintf(nm, sizeof nm, "%sngram_heads_offsets", pre);
    const int64_t* off = i64(&st, nm, g.heads);
    CHECK(mult && size && off, "n-gram tables missing");
    for (int p = 0; mult && p < c.ngram; ++p) CHECK(g.mult[p] == mult[p], "multiplier %d: %lld, want %lld", p, (long long) g.mult[p], (long long) mult[p]);
    for (int h = 0; size && off && h < g.heads; ++h)
        CHECK(g.size[h] == size[h] && g.off[h] == off[h], "head %d: size %lld off %lld", h, (long long) g.size[h], (long long) g.off[h]);
    CHECK(g.rows == 4096, "PLE rows %lld", (long long) g.rows);
    st_close_file(&st);

    snprintf(path, sizeof path, "%s_golden.safetensors", DIR_);
    CHECK(st_open_file(&st, path, err, sizeof err) == 0, "%s", err);
    const StEntry* ei = st_find(&st, "ids");
    const StEntry* el = st_find(&st, "logits");
    const StEntry* er = st_find(&st, "ple_rows");
    if (!ei || !el || !er) { printf("FAIL: golden tensors missing\n"); return 1; }
    const int n = (int) ei->shape[0], V = (int) el->shape[1];
    const int32_t* ids = (const int32_t*) st_data(&st, ei);
    const float* want = (const float*) st_data(&st, el);
    const int64_t* rows = (const int64_t*) st_data(&st, er);
    int bad_rows = 0;
    for (int t = 0; t < n; ++t) {
        int64_t r[QWEN4_MAX_PLE_HEADS];
        qwen4_ngram_rows(&c, &g, ids, t, r);
        for (int h = 0; h < g.heads; ++h) bad_rows += r[h] != rows[t * g.heads + h];
    }
    CHECK(bad_rows == 0, "%d PLE n-gram rows differ from mlx-vlm's", bad_rows);
    int pl = 0;   // the layer of PLE 0
    while (pl + 1 < c.n_layer && c.ple[pl] != 0) ++pl;
    NsModel m;
    CHECK(ns_open(&m, DIR_, err, sizeof err) == 0, "%s", err);
    for (int t = 0; t < n; ++t)
        CHECK(qwen4_ple_prefetch(&m, NULL, &c, &g, pl, ids, t) == g.heads, "token %d: PLE rows to prefetch not found", t);
    ns_close(&m);

    Qwen4Ref* ref = qwen4_ref_open(DIR_, err, sizeof err);
    CHECK(ref != NULL, "open: %s", err);
    if (ref) {
        CHECK(V == qwen4_ref_cfg(ref)->vocab, "vocab");
        float* got = (float*) malloc(sizeof(float) * (size_t) n * (size_t) V);
        CHECK(qwen4_ref_forward(ref, ids, n, got) == 0, "forward: %s", qwen4_ref_error(ref));
        double worst = 0, scale = 0;
        int argmax_diff = 0, worst_t = 0;
        for (int t = 0; t < n; ++t) {
            int a = 0, b = 0;
            for (int v = 0; v < V; ++v) {
                const double d = fabs((double) got[t * V + v] - want[t * V + v]);
                if (d > worst) { worst = d; worst_t = t; }
                if (fabs(want[t * V + v]) > scale) scale = fabs(want[t * V + v]);
                if (got[t * V + v] > got[t * V + a]) a = v;
                if (want[t * V + v] > want[t * V + b]) b = v;
            }
            argmax_diff += a != b;
        }
        printf("logits: max |C - mlx-vlm| %.2e (row %d) of max |logit| %.3f, argmax differs in %d of %d rows\n", worst, worst_t,
               scale, argmax_diff, n);
        CHECK(worst < 1e-5 * scale && argmax_diff == 0, "logits differ from mlx-vlm");
        free(got);
        qwen4_ref_close(ref);
    }
    st_close_file(&st);
    printf("test_qwen4_ref: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
