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

static int argmax(const float* l, int V) { int b = 0; for (int v = 1; v < V; ++v) if (l[v] > l[b]) b = v; return b; }
// The MTP head (plan M1, gate MA) against mlx-vlm: its logits over the prompt, then 3 greedy drafts. Skipped without a golden.
static void mtp_case(Qwen4Ref* ref, const int32_t* ids, int n, int V) {
    StFile g;
    char err[256];
    if (st_open_file(&g, "tests/data/qwen4_tiny_mtp_golden.safetensors", err, sizeof err)) { printf("MTP: no golden yet\n"); return; }
    const StEntry *el = st_find(&g, "logits"), *ed = st_find(&g, "drafts");
    const float* want = (const float*) st_data(&g, el);
    const int32_t* wd = (const int32_t*) st_data(&g, ed);
    const int HD = qwen4_ref_cfg(ref)->hc * qwen4_ref_cfg(ref)->d, nd = (int) ed->shape[0];
    float *lg = (float*) malloc(sizeof(float) * (size_t) (n + nd) * V), *H = (float*) malloc(sizeof(float) * (size_t) (n + nd) * HD);
    float* Hm = (float*) malloc(sizeof(float) * (size_t) (n + nd) * HD);
    int32_t* tok = (int32_t*) malloc(sizeof(int32_t) * (size_t) (n + nd));
    CHECK(qwen4_ref_forward_h(ref, ids, n, lg, H) == 0, "forward_h: %s", qwen4_ref_error(ref));
    for (int t = 0; t + 1 < n; ++t) tok[t] = ids[t + 1];
    tok[n - 1] = argmax(lg + (size_t) (n - 1) * V, V);   // the bonus: the main model's next token
    double worst = 0, scale = 0;
    int same = 1;
    for (int j = 0; j < nd; ++j) {   // each draft: the head over the prompt and the drafts so far, its streams fed forward
        CHECK(qwen4_ref_mtp(ref, tok, H, n + j, lg, Hm) == 0, "mtp: %s", qwen4_ref_error(ref));
        if (!j)
            for (size_t x = 0; x < (size_t) n * V; ++x) { worst = fmax(worst, fabs((double) lg[x] - want[x])); scale = fmax(scale, fabs(want[x])); }
        tok[n + j] = argmax(lg + (size_t) (n + j - 1) * V, V);
        memcpy(H + (size_t) (n + j) * HD, Hm + (size_t) (n + j - 1) * HD, sizeof(float) * (size_t) HD);
        same &= tok[n + j] == wd[j];
    }
    printf("MTP: logits max |C - mlx-vlm| %.2e of %.3f; drafts %s\n", worst, scale, same ? "equal" : "differ");
    CHECK(worst < 1e-5 * scale && same, "MTP head differs from mlx-vlm");
    free(lg); free(H); free(Hm); free(tok);
    st_close_file(&g);
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
        qwen4_ref_kv_bf16(ref, 1);   // keys and values rounded to BF16 (plan K): near the f32 cache's logits, not equal
        CHECK(qwen4_ref_forward(ref, ids, n, got) == 0, "forward, BF16 KV: %s", qwen4_ref_error(ref));
        double kw = 0;
        int kad = 0;
        for (int t = 0; t < n; ++t) {
            int a = 0, b = 0;
            for (int v = 0; v < V; ++v) {
                const double d = fabs((double) got[t * V + v] - want[t * V + v]);
                if (d > kw) kw = d;
                if (got[t * V + v] > got[t * V + a]) a = v;
                if (want[t * V + v] > want[t * V + b]) b = v;
            }
            kad += a != b;
        }
        printf("logits with the BF16 KV cache: max |C - mlx-vlm| %.2e of max |logit| %.3f, argmax differs in %d of %d rows\n", kw, scale, kad, n);
        CHECK(kw > 0 && kw < 2e-2 * scale && kad == 0, "the BF16 KV cache: logits %.2e from mlx-vlm's", kw);
        qwen4_ref_kv_bf16(ref, 0);
        free(got);
        mtp_case(ref, ids, n, V);
        qwen4_ref_close(ref);
    }
    st_close_file(&st);
    printf("test_qwen4_ref: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
