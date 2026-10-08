// tests/test_engine.c - the GPU engine (Metal or CUDA, whichever is linked) against the CPU reference forward
// (nslm/mova_ref.h), through engine_api.h only.
//
// A small MoVA model folder with random weights in every encoding the engine reads (SEED4P4 routed experts, Q4 value
// experts, Q8 projections / embedding / head, BF16 norms and routers) is written to out/test/engine_model.  Then:
//   prompt scoring (eng_score: the GEMM and prefill-attention kernels) against the reference logits,
//   token-by-token decode (eng_prefill + eng_step: the matvec and split-key attention kernels) against them,
//   the engine's router choices (eng_mova_routes) against the reference's,
//   prompt-lookup decode (ENG_MODE_PL) committing exactly the tokens of plain greedy decode,
//   batched decode of several sequence slots (eng_step_batch) against decoding each alone.
// Logits agree to the engines' f32 accumulation (relative error well under a BF16 step of the logit scale) wherever
// the router choices agree; a choice can flip on a near tie, which the test counts and bounds.
#define _DEFAULT_SOURCE   // glibc: setenv / unsetenv under -std=c11 (macOS declares them anyway; Windows: _putenv_s)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "engine_api.h"
#include "lfsr.h"
#include "model_st.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"
#include "mova_ext.h"
#include "mova_ref.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char* CONFIG =
    "{\n  \"architectures\": [\"K2HorizonForCausalLM\"],\n  \"attention_gate_func\": \"softplus\",\n"
    "  \"decoder_sparse_step\": 1,\n  \"head_dim\": 128,\n  \"hidden_act\": \"silu\",\n  \"hidden_size\": 512,\n"
    "  \"intermediate_size\": 1024,\n  \"layernorm_num_groups\": 2,\n  \"mlp_only_layers\": [0],\n"
    "  \"model_type\": \"k2_horizon\",\n  \"moe_gate_bias\": true,\n  \"moe_intermediate_size\": 256,\n"
    "  \"mova_num_experts\": 8,\n  \"mova_num_experts_per_tok\": 2,\n  \"norm_topk_prob\": true,\n"
    "  \"num_attention_heads\": 8,\n  \"num_experts\": 16,\n  \"num_experts_per_tok\": 4,\n  \"num_hidden_layers\": 5,\n"
    "  \"num_key_value_heads\": 2,\n  \"num_shared_experts\": 1,\n  \"query_key_norm\": false,\n  \"rms_norm_eps\": 1e-06,\n"
    "  \"rope_head_dim\": 128,\n  \"rope_parameters\": {\"rope_theta\": 10000000.0, \"rope_type\": \"default\"},\n"
    "  \"router_scaling_factor\": 2.5,\n  \"router_score_func\": \"sigmoid\",\n  \"tie_word_embeddings\": false,\n"
    "  \"use_sliding_window\": false,\n  \"vocab_size\": 2048\n}\n";

static uint32_t rng_u32(uint64_t* s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t) (*s >> 33);
}
static double rng_unit(uint64_t* s) { return (rng_u32(s) & 0xFFFFFF) / 16777216.0; }   // [0, 1)
static uint16_t f2bf(float f) { return nslm_f2bf(f); }

typedef struct {
    const NsSpec* t;
    const MovaTensor* mt;
} Gen;

// Deterministic contents per (tensor, stream): weights of moderate size so activations stay O(1) through the layers.
static int fill(void* ctx, int ti, int s, uint8_t* dst, uint64_t len) {
    const Gen* g = (const Gen*) ctx;
    const NsSpec* t = &g->t[ti];
    const MovaTensor* mt = &g->mt[ti];
    uint64_t st = 0x9E3779B97F4A7C15ull * (uint64_t) (ti + 1) + 0x632BE59BD9B4E019ull * (uint64_t) (s + 1);
    if (t->enc == NS_BF16) {
        uint16_t* o = (uint16_t*) dst;
        for (uint64_t i = 0; i < len / 2; ++i) {
            const double u = rng_unit(&st) * 2 - 1;
            const float v = mt->kind == MOVA_K_NORM ? (float) (1.0 + 0.1 * u)
                          : mt->kind == MOVA_K_ROUTER_BIAS ? (float) (0.02 * u)
                          : mt->kind == MOVA_K_HEADS ? (float) (0.08 * u)
                          : strstr(mt->name, ".mla.") ? (float) (0.04 * u)      // MLA projections: BF16, Q8's magnitude
                          : (float) (1.0 * u);                                   // routers: well-separated scores
            o[i] = f2bf(v);
        }
        return 0;
    }
    if (t->enc == NS_Q8 || t->enc == NS_Q4) {
        if (s == 0) { for (uint64_t i = 0; i < len; ++i) dst[i] = (uint8_t) rng_u32(&st); return 0; }
        const float sc = t->enc == NS_Q8 ? 0.0003f : 0.005f, mid = t->enc == NS_Q8 ? 128.0f : 8.0f;
        uint16_t* o = (uint16_t*) dst;
        uint64_t s1 = 0x1234567ull * (uint64_t) (ti + 1);   // scales and biases from one stream, so they pair up
        for (uint64_t i = 0; i < len / 2; ++i) {
            const float sv = sc * (float) (0.75 + 0.5 * rng_unit(&s1));
            o[i] = f2bf(s == 1 ? sv : -mid * sv * (float) (0.9 + 0.2 * rng_unit(&st)));
        }
        return 0;
    }
    // SEED4P4: seeds in [1, 65535], any coefficients, exponents in [bias, bias + 2]
    if (s == 0) { uint16_t* o = (uint16_t*) dst; for (uint64_t i = 0; i < len / 2; ++i) o[i] = (uint16_t) (1 + rng_u32(&st) % 65535); }
    else if (s == 1) { uint16_t* o = (uint16_t*) dst; for (uint64_t i = 0; i < len / 2; ++i) o[i] = (uint16_t) rng_u32(&st); }
    else if (s == 2) { int32_t* o = (int32_t*) dst; for (uint64_t i = 0; i < len / 4; ++i) o[i] = -10; }
    else for (uint64_t i = 0; i < len; ++i) dst[i] = (uint8_t) ((rng_u32(&st) % 3) | ((rng_u32(&st) % 3) << 4));
    return 0;
}

// skip: leave out the tensors whose names contain it (NULL: none)
static int write_model(const char* dir, const char* config, const char* skip, char* err, int errlen) {
    mkdir("out", 0755);
    mkdir("out/test", 0755);
    mkdir(dir, 0755);
    char path[1024];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) { snprintf(err, (size_t) errlen, "cannot write %.400s", path); return -1; }
    fputs(config, f);
    fclose(f);
    MovaCfg c;
    if (mova_cfg_parse(&c, config, err, errlen)) return -1;
    MovaTensor* mt = NULL;
    const int n = mova_tensors(&c, &mt);
    NsSpec* sp = (NsSpec*) calloc((size_t) n, sizeof(NsSpec));
    int m = 0;
    for (int i = 0; i < n; ++i) {
        if (skip && strstr(mt[i].name, skip)) continue;
        mt[m] = mt[i];
        const int k = mt[m].kind;
        sp[m].name = mt[m].name;
        sp[m].enc = k == MOVA_K_EXPERTS ? NS_SEED4P4 : k == MOVA_K_VEXPERTS ? NS_Q4
                  : (k == MOVA_K_NORM || k == MOVA_K_ROUTER || k == MOVA_K_ROUTER_BIAS || k == MOVA_K_HEADS) ? NS_BF16
                  : strstr(mt[m].name, ".mla.") ? NS_BF16 : NS_Q8;   // MLA tensors: BF16, as exported (no *.weight name)
        sp[m].slices = mt[m].slices;
        sp[m].rows = mt[m].rows;
        sp[m].cols = mt[m].cols;
        ++m;
    }
    Gen g = {sp, mt};
    const int rc = ns_write(dir, sp, m, 64ull << 20, NULL, fill, &g, err, errlen);
    free(sp);
    free(mt);
    return rc;
}

// Relative logit error of one row: max |a - b| over the row, divided by max |b|.
static double row_err(const float* a, const float* b, int V) {
    double m = 0, s = 0;
    for (int i = 0; i < V; ++i) { m = fmax(m, fabs((double) a[i] - b[i])); s = fmax(s, fabs((double) b[i])); }
    return s > 0 ? m / s : m;
}
// The router choices of one row (every sparse layer), as sets: selection order among near-equal scores does not matter.
static int same_choice(const int32_t* a, const int32_t* b, int n, int k) {
    for (int i = 0; i < n; ++i) {
        int x[16], y[16];
        memcpy(x, a + i * k, sizeof(int32_t) * (size_t) k);
        memcpy(y, b + i * k, sizeof(int32_t) * (size_t) k);
        for (int p = 1; p < k; ++p) for (int q = p; q > 0 && x[q] < x[q - 1]; --q) { const int t = x[q]; x[q] = x[q - 1]; x[q - 1] = t; }
        for (int p = 1; p < k; ++p) for (int q = p; q > 0 && y[q] < y[q - 1]; --q) { const int t = y[q]; y[q] = y[q - 1]; y[q - 1] = t; }
        if (memcmp(x, y, sizeof(int) * (size_t) k)) return 0;
    }
    return 1;
}
static int argmax(const float* l, int V) {
    int b = 0;
    for (int i = 1; i < V; ++i) if (l[i] > l[b]) b = i;
    return b;
}

static void set_env(const char* k, const char* v) {   // v = NULL: unset
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

// A second engine opened with one setting changed (env k = v, and the KV format) runs the same scoring and decoding as
// e; returns the number of differing logits and tokens and the largest relative logit difference.  Twice: the second
// round starts from caches the first one left.
static int compare_engines(Eng* e, const EngOpts* o0, int kv_format, const char* k, const char* v, uint64_t* st, int V,
                           double* rel, const char* what) {
    EngOpts o = *o0;
    o.kv_format = kv_format;
    char err[512] = "";
    if (k) set_env(k, v);
    Eng* e2 = eng_open(&o, err, sizeof err);
    if (k) set_env(k, NULL);
    if (!e2) { printf("FAIL: eng_open (%s): %s\n", what, err); ++fails; return -1; }
    printf("%s: %s\n", what, eng_describe(e2));
    const int NP = 300, cnt = 40, seq = 0;
    int32_t* pid = (int32_t*) malloc(sizeof(int32_t) * NP);
    for (int i = 0; i < NP; ++i) pid[i] = (int32_t) (rng_u32(st) % (uint32_t) V);
    float* la = (float*) malloc(sizeof(float) * (size_t) cnt * V);
    float* lb = (float*) malloc(sizeof(float) * (size_t) cnt * V);
    int32_t ga[32], gb[32];
    int diff = 0;
    *rel = 0;
    for (int round = 0; round < 2; ++round) {
        CHECK(eng_score(e, 0, pid, NP - cnt, cnt, la) == 0 && eng_score(e2, 0, pid, NP - cnt, cnt, lb) == 0, "scoring");
        for (size_t i = 0; i < (size_t) cnt * V; ++i) diff += la[i] != lb[i];
        for (int t = 0; t < cnt; ++t) *rel += row_err(lb + (size_t) t * V, la + (size_t) t * V, V) / (2.0 * cnt);   // mean over both rounds
        CHECK(eng_prefill(e, 0, pid, NP / 2) == 0 && eng_generate(e, &seq, 1, 32, ENG_MODE_AR, ga, NULL) == 0, "decode");
        CHECK(eng_prefill(e2, 0, pid, NP / 2) == 0 && eng_generate(e2, &seq, 1, 32, ENG_MODE_AR, gb, NULL) == 0, "decode");
        diff += memcmp(ga, gb, sizeof ga) != 0;
    }
    free(pid); free(la); free(lb);
    eng_close(e2);
    return diff;
}

// 9. batched decode (eng_step_batch): NB sequences of mixed lengths in their own slots (more than one forward's MV_MAXT
// rows), filled by chunked prefill (eng_prefill_begin / eng_prefill_next, interleaved), then stepped together with
// teacher-forced tokens.  Each row against the same sequence run alone (same chunks, same steps) in a one-slot engine:
// every row must agree bit for bit (a dense matvec adds each row's products in the same order for any row count), and
// prompt caches must not depend on the prefill chunks either (a server's chunks start wherever a slot's cache reuse
// ends).  Route flips are still counted and bounded, and argmax checked, so a failure says how far apart the rows are.
static void test_batch(const char* dir, const char* what) {
    enum { NB = 11, STEPS = 4, CAP = 256, CH = 50, NSP = 4, TK = 4, TKV = 2 };
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = dir;
    o.resource_dir = getenv("NSLM_RES") ? getenv("NSLM_RES") : "out/res";
    o.max_seqs = 1;
    o.kv_tokens = CAP;
    char err[512] = "";
    Eng* e1 = eng_open(&o, err, sizeof err);
    o.max_seqs = NB + 1;
    o.kv_tokens = (int64_t) CAP * (NB + 1);
    Eng* eb = e1 ? eng_open(&o, err, sizeof err) : NULL;
    if (!e1 || !eb) { ++fails; printf("FAIL: %s batch: eng_open: %s\n", what, err); eng_close(e1); return; }
    const int V = eng_vocab(eb);
    uint64_t st = 4242;
    int len[NB], seqs[NB];
    int32_t ids[NB][CAP], tf[NB][STEPS];
    for (int i = 0; i < NB; ++i) {
        len[i] = 2 + (i * 37) % 190;   // 2 .. 189 tokens: chunks of CH rows and short tails (the decode kernels)
        seqs[i] = NB - i;              // slots 11 .. 1, slot 0 unused: nonzero cache bases
        for (int k = 0; k < len[i]; ++k) ids[i][k] = (int32_t) (rng_u32(&st) % (uint32_t) V);
        for (int j = 0; j < STEPS; ++j) tf[i][j] = (int32_t) (rng_u32(&st) % (uint32_t) V);
    }
    // alone: each sequence in the one-slot engine
    float* ref = (float*) malloc(sizeof(float) * (size_t) NB * STEPS * V), *lb = (float*) malloc(sizeof(float) * (size_t) NB * V);
    int32_t rm[NB][STEPS][NSP * TK], rv[NB][STEPS][NSP * TKV], bm[NB][NSP * TK], bv[NB][NSP * TKV];
    for (int i = 0; i < NB; ++i) {
        int reused = -1, left;
        CHECK(eng_prefill_begin(e1, 0, ids[i], len[i], &reused) == 0 && reused == 0, "%s batch: prefill_begin alone", what);
        while ((left = eng_prefill_next(e1, 0, CH)) > 0) {}
        CHECK(left == 0, "%s batch: prefill_next alone", what);
        for (int j = 0; j < STEPS; ++j) {
            eng_mova_routes(e1, 1, 1);
            CHECK(eng_step(e1, 0, ref + ((size_t) i * STEPS + j) * V) == 0, "%s batch: step alone", what);
            CHECK(eng_mova_routes_read(e1, 1, rm[i][j], rv[i][j], NULL, NULL) == 0, "%s batch: routes alone", what);
            eng_push(e1, 0, tf[i][j]);
        }
    }
    eng_mova_routes(e1, 0, 0);
    for (int k = 0; k < 2; ++k) {   // sequence 5 (187 tokens) prefilled in chunks of 7, and in one call
        const int i = 5;
        int left = 0;
        if (k == 0) {
            CHECK(eng_prefill_begin(e1, 0, ids[i], len[i], NULL) == 0, "%s batch: prefill_begin", what);
            while ((left = eng_prefill_next(e1, 0, 7)) > 0) {}
        } else left = eng_prefill(e1, 0, ids[i], len[i]);
        CHECK(left == 0 && eng_step(e1, 0, lb) == 0, "%s batch: prefill in chunks", what);
        const int same = !memcmp(lb, ref + (size_t) i * STEPS * V, sizeof(float) * (size_t) V);
        const double re = row_err(lb, ref + (size_t) i * STEPS * V, V);
        printf("%s prefill %s: %s (logit error %.2e)\n", what, k ? "in one call" : "in chunks of 7", same ? "bit-equal to chunks of 50" : "differs", re);
        CHECK(same, "%s batch: prefill chunking changed the logits (error %.3e)", what, re);
    }
    // together: chunked prefills interleaved, then batched steps
    for (int i = 0; i < NB; ++i) CHECK(eng_prefill_begin(eb, seqs[i], ids[i], len[i], NULL) == 0, "%s batch: prefill_begin", what);
    CHECK(eng_step_batch(eb, seqs, NB, lb) != 0, "%s batch: a step over slots still prefilling must fail", what);
    for (int busy = 1; busy;) {
        busy = 0;
        for (int i = 0; i < NB; ++i) {
            const int left = eng_prefill_next(eb, seqs[i], CH);
            CHECK(left >= 0, "%s batch: prefill_next", what);
            busy |= left > 0;
        }
    }
    const int dup[3] = {seqs[0], seqs[1], seqs[0]};
    CHECK(eng_step_batch(eb, dup, 3, lb) != 0, "%s batch: a sequence listed twice must fail", what);
    int rows = 0, exact = 0, flips = 0, am_same = 0;
    double worst = 0;
    for (int j = 0; j < STEPS; ++j) {
        eng_mova_routes(eb, 1, NB);
        CHECK(eng_step_batch(eb, seqs, NB, lb) == 0, "%s batch: eng_step_batch", what);
        CHECK(eng_mova_routes_read(eb, NB, &bm[0][0], &bv[0][0], NULL, NULL) == 0, "%s batch: routes", what);
        for (int i = 0; i < NB; ++i) {
            const float* r = ref + ((size_t) i * STEPS + j) * V, *b = lb + (size_t) i * V;
            ++rows;
            exact += !memcmp(r, b, sizeof(float) * (size_t) V);
            if (!same_choice(bm[i], rm[i][j], NSP, TK) || !same_choice(bv[i], rv[i][j], NSP, TKV)) { ++flips; continue; }
            worst = fmax(worst, row_err(b, r, V));
            am_same += argmax(b, V) == argmax(r, V);
        }
        for (int i = 0; i < NB; ++i) eng_push(eb, seqs[i], tf[i][j]);
    }
    eng_mova_routes(eb, 0, 0);
    printf("%s batched decode: %d sequences x %d steps, %d of %d rows bit-equal to decoding alone, %d route flips, logit "
           "error max %.2e (other rows), argmax equal %d\n", what, NB, STEPS, exact, rows, flips, worst, am_same);
    CHECK(flips <= rows / 8, "%s batch: %d rows with different router choices", what, flips);
    CHECK(worst < 2e-2, "%s batch: logit error %.3e against decoding alone", what, worst);
    CHECK(am_same == rows - flips, "%s batch: argmax differs in %d rows", what, rows - flips - am_same);
    CHECK(exact == rows, "%s batch: %d of %d rows differ from decoding alone", what, rows - exact, rows);
    CHECK(eng_prefill(eb, NB + 1, ids[0], 4) != 0 && eng_prefill(eb, 1, ids[0], CAP + 1) != 0, "%s batch: slot bounds", what);
    {   // a slot's cache copied out (eng_kv_read) and into another slot (eng_kv_write): that slot then reuses it and
        // decodes as the original sequence did alone
        const int i = 5, P = 128, src = seqs[i];
        void* kv = malloc((size_t) eng_kv_bytes(eb) * P);
        int reused = -1, left;
        void* hi = (char*) kv + (size_t) eng_kv_bytes(eb) * (P / 2);   // positions [P / 2, P) (the layout is per layer)
        CHECK(eng_kv_bytes(eb) > 0 && eng_kv_read(eb, src, 0, P / 2, kv) == 0 && eng_kv_read(eb, src, P / 2, P, hi) == 0,
              "%s batch: eng_kv_read", what);
        CHECK(eng_kv_read(eb, src, 0, CAP, kv) != 0, "%s batch: eng_kv_read past the cache", what);
        CHECK(eng_kv_write(eb, 0, ids[i], 0, P / 2, kv) == 0 && eng_len(eb, 0) == P / 2, "%s batch: eng_kv_write", what);
        CHECK(eng_kv_write(eb, 0, ids[i], P / 2 + 1, P, hi) != 0, "%s batch: eng_kv_write past the cache", what);
        CHECK(eng_kv_write(eb, 0, ids[i], P / 2, P, hi) == 0, "%s batch: eng_kv_write", what);
        CHECK(eng_prefill_begin(eb, 0, ids[i], len[i], &reused) == 0 && reused == P, "%s batch: reuse of a written cache (%d)", what, reused);
        while ((left = eng_prefill_next(eb, 0, CH)) > 0) {}
        CHECK(left == 0 && eng_step(eb, 0, lb) == 0, "%s batch: step after eng_kv_write", what);
        const double re = row_err(lb, ref + (size_t) i * STEPS * V, V);
        const int same = !memcmp(lb, ref + (size_t) i * STEPS * V, sizeof(float) * (size_t) V);
        printf("%s cache copied between slots: %s (logit error %.2e)\n", what, same ? "bit-equal" : "differs", re);
        CHECK(same, "%s batch: a copied cache decodes differently (error %.3e)", what, re);
        free(kv);
    }
    eng_close(e1);
    eng_close(eb);
    free(ref);
    free(lb);
}

// 8. MLA (a TransMLA conversion): the same random model with latent attention (ranks 64..128, one 128-dim RoPE key),
// against the C reference: prompt scoring (one-pass latent attention) and decode (split-key + reduce), the cache size.
static void test_mla(void) {
    const char* dir = "out/test/engine_model_mla";
    static const int ranks[5] = {64, 96, 64, 128, 96};
    char* config = (char*) malloc(strlen(CONFIG) + 128);
    sprintf(config, "{\"mla_ranks\": [%d, %d, %d, %d, %d], \"mla_rope_dim\": 128,%s", ranks[0], ranks[1], ranks[2], ranks[3],
            ranks[4], CONFIG + 1);
    char err[512] = "";
    if (write_model(dir, config, NULL, err, sizeof err)) { ++fails; printf("FAIL: MLA model folder: %s\n", err); free(config); return; }
    // A folder without one layer's per-head query map: the reference (and below, the engine) must refuse it, naming
    // the tensor
    const char* bad = "out/test/engine_model_mla_missing";
    const int bad_ok = write_model(bad, config, "layers.2.self_attn.mla.q_lat", err, sizeof err) == 0;
    CHECK(bad_ok, "MLA folder without q_lat: %s", err);
    if (bad_ok) {
        MovaRef* rb = mova_ref_open(bad, 1, err, sizeof err);
        CHECK(!rb && strstr(err, "layers.2.self_attn.mla.q_lat"), "MLA reference without q_lat: %s", rb ? "opened" : err);
        if (rb) mova_ref_close(rb);
    }
    free(config);
    // The per-head maps are stacked 3-D BF16 tensors [n_head][rows][cols], as the conversion exports them: the engine
    // maps them from the folder (as from the real converted model's model-mla-delta.safetensors), and the checkpoint
    // reader (the packer's, and the engine's fallback) reads them whole
    {
        MovaCkpt* ck = mova_ckpt_open(dir, err, sizeof err);
        NsModel nm;
        CHECK(ck && ns_open(&nm, dir, err, sizeof err) == 0, "MLA folder as a checkpoint: %s", err);
        if (ck) {
            const char* nm1 = "model.layers.1.self_attn.mla.q_lat";
            const uint16_t* w3 = mova_ckpt_bf16_3d(ck, nm1, 8, ranks[1], 128, err, sizeof err);
            const NsTensor* t = ns_find(&nm, nm1);
            CHECK(w3 && t && t->slices == 8 && !memcmp(w3, t->s[0].p, (size_t) 8 * ranks[1] * 128 * 2), "3-D read of %s: %s", nm1, err);
            CHECK(!mova_ckpt_bf16(ck, nm1, ranks[1], 128, err, sizeof err), "a 3-D tensor read as 2-D");
            ns_close(&nm);
            mova_ckpt_close(ck);
        }
    }
    MovaRef* ref = mova_ref_open(dir, 8, err, sizeof err);
    if (!ref) { ++fails; printf("FAIL: MLA reference: %s\n", err); return; }
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = dir;
    o.resource_dir = getenv("NSLM_RES") ? getenv("NSLM_RES") : "out/res";
    o.max_seqs = 1;
    o.kv_tokens = 1024;
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { ++fails; printf("FAIL: MLA eng_open: %s\n", err); mova_ref_close(ref); return; }
    printf("%s\n", eng_describe(e));
    if (bad_ok) {
        EngOpts ob = o;
        ob.model_dir = bad;
        ob.kv_tokens = 64;
        Eng* eb = eng_open(&ob, err, sizeof err);
        CHECK(!eb && strstr(err, "layers.2.self_attn.mla.q_lat"), "MLA engine without q_lat: %s", eb ? "opened" : err);
        if (eb) eng_close(eb);
    }
    int64_t want_kv = 0;
    for (int l = 0; l < 5; ++l) want_kv += (int64_t) o.kv_tokens * (128 + ranks[l]) * 2;
    EngMem mem;
    eng_mem(e, &mem);
    CHECK(mem.kv == want_kv, "MLA KV cache %lld bytes, want %lld", (long long) mem.kv, (long long) want_kv);
    o.kv_format = ENG_KV_Q8;
    Eng* e8 = eng_open(&o, err, sizeof err);
    CHECK(!e8 && strstr(err, "MLA"), "MLA with the 8-bit cache must be refused (not ignored): %s", e8 ? "opened" : err);
    if (e8) eng_close(e8);
    const int V = eng_vocab(e), N = 48, NS = 4, TK = 4, TKV = 2;
    uint64_t st = 11;
    int32_t ids[64];
    for (int i = 0; i < N; ++i) ids[i] = (int32_t) (rng_u32(&st) % (uint32_t) V);
    float* rl = (float*) malloc(sizeof(float) * (size_t) N * V), *el = (float*) malloc(sizeof(float) * (size_t) N * V);
    int32_t* rm = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TK), *rv = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TKV);
    int32_t* em = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TK), *ev = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TKV);
    CHECK(mova_ref_forward(ref, ids, N - 1, 0, rl, rm, rv) == 0, "MLA reference forward");
    // Prompt rows: the CUDA engine decompresses the keys per head (kc_mla_prefill), and with NSLM_MLA_DEC_KEYS=0 attends
    // absorbed as the Metal engine does; both against the reference (absorbed, in double).
    const int cuda = strstr(eng_describe(e), "CUDA") != NULL;
    for (int path = 0; path < 1 + cuda; ++path) {
        Eng* es = e;
        if (path) {
            EngOpts oa = o;
            oa.kv_format = ENG_KV_BF16;
            set_env("NSLM_MLA_DEC_KEYS", "0");
            es = eng_open(&oa, err, sizeof err);
            set_env("NSLM_MLA_DEC_KEYS", NULL);
            if (!es) { ++fails; printf("FAIL: MLA eng_open (absorbed prompts): %s\n", err); break; }
        }
        const char* pn = !cuda ? "" : path ? " (absorbed prompts)" : " (prompt keys decompressed)";
        eng_mova_routes(es, 1, N);
        CHECK(eng_score(es, 0, ids, 1, N - 1, el) == 0, "MLA eng_score%s", pn);
        CHECK(eng_mova_routes_read(es, N - 1, em, ev, NULL, NULL) == 0, "MLA routes%s", pn);
        eng_mova_routes(es, 0, 0);
        int flips = 0, clean = 0, am_same = 0;
        double worst = 0, mean = 0;
        for (int t = 0; t < N - 1; ++t) {
            const int same = same_choice(em + t * NS * TK, rm + t * NS * TK, NS, TK) && same_choice(ev + t * NS * TKV, rv + t * NS * TKV, NS, TKV);
            flips += !same;
            const double re = row_err(el + (size_t) t * V, rl + (size_t) t * V, V);
            am_same += argmax(el + (size_t) t * V, V) == argmax(rl + (size_t) t * V, V);
            mean += re;
            if (!flips) { ++clean; worst = fmax(worst, re); }
        }
        mean /= N - 1;
        printf("MLA scoring%s: %d rows, %d before the first router flip (%d flipped); logit error max %.2e (clean rows), mean "
               "%.2e; argmax equal %d / %d\n", pn, N - 1, clean, flips, worst, mean, am_same, N - 1);
        // The latent attention computes in f32 against the reference's double (GQA's prefill kernel instead reproduces
        // MLX's rounding points, which the reference mirrors), and the decompressed path rounds its keys and values to
        // BF16 where the reference rounds q_lat q and the latent output, so 1-ulp differences start at row 0 and tip a
        // near-tie route early.  Over five token sequences on an RTX 4080 the first flip came at rows 3-47 (absorbed) and
        // 2-12 (decompressed), with up to 8 flips, clean-row errors up to 1.3e-2 and argmax equal in 44 or more of 47
        // rows on either path.  The kernels themselves are pinned to BF16 ulps by tests/test_mova_kernels.c; this test
        // checks the wiring, which these bounds already rule out.
        CHECK(flips <= (N - 1) / 4 && clean >= 2, "MLA scoring%s: %d flips, %d clean rows", pn, flips, clean);
        CHECK(worst < 0.02 && mean < 0.05, "MLA scoring%s: logit error max %.3e, mean %.3e", pn, worst, mean);
        CHECK(am_same >= (N - 1) * 9 / 10, "MLA scoring%s: argmax equal in %d of %d rows", pn, am_same, N - 1);
        if (path) eng_close(es);
    }
    const int P0 = 8;   // decode: prefill 8 rows (the split-key path), then one row at a time
    CHECK(eng_prefill(e, 0, ids, P0) == 0, "MLA prefill");
    float* lg = (float*) malloc(sizeof(float) * (size_t) V);
    double dworst = 0, dmean = 0;
    int dsame = 0;
    for (int t = P0 - 1; t < N - 1; ++t) {
        CHECK(eng_step(e, 0, lg) == 0, "MLA step");
        const double re = row_err(lg, rl + (size_t) t * V, V);
        dworst = fmax(dworst, re);
        dmean += re / (N - P0);
        dsame += argmax(lg, V) == argmax(rl + (size_t) t * V, V);
        eng_push(e, 0, ids[t + 1]);
    }
    printf("MLA decode: %d steps, logit error mean %.2e max %.2e, argmax equal %d\n", N - P0, dmean, dworst, dsame);
    CHECK(dmean < 0.05 && dworst < 0.25, "MLA decode: logit error mean %.3e, max %.3e", dmean, dworst);
    // prompt lookup (multi-row verify forwards: split-key latent attention over several rows) must commit the tokens of
    // plain greedy decode
    {
        int32_t rep[40], ar[24], pl[24];
        for (int i = 0; i < 40; ++i) rep[i] = ids[i % 10];
        const int seq = 0;
        EngStats sp;
        memset(&sp, 0, sizeof sp);
        CHECK(eng_prefill(e, 0, rep, 40) == 0 && eng_generate(e, &seq, 1, 24, ENG_MODE_AR, ar, NULL) == 0, "MLA AR generate");
        CHECK(eng_prefill(e, 0, rep, 40) == 0 && eng_generate(e, &seq, 1, 24, ENG_MODE_PL, pl, &sp) == 0, "MLA PL generate");
        // The CUDA engine's one-row and multi-row forwards differ by an ulp here and there (its matvec sums one row in
        // another order than several), so a verify forward may break a near-tie the other way: tokens up to the first
        // difference must match, and there greedy decode's choice may lead the verify's by at most that noise (the
        // logits of one and several rows differ by < 2e-2 on this model).
        int k = 0;
        while (k < 24 && ar[k] == pl[k]) ++k;
        if (k < 24) {
            float* lk = (float*) malloc(sizeof(float) * (size_t) V);
            CHECK(eng_prefill(e, 0, rep, 40) == 0, "MLA AR replay");
            for (int j = 0; j <= k; ++j) {
                CHECK(eng_step(e, 0, lk) == 0, "MLA AR replay step");
                eng_push(e, 0, ar[j]);
            }
            const double gap = (double) lk[ar[k]] - lk[pl[k]];
            printf("MLA prompt lookup: differs from greedy decode at token %d, a near-tie (logit gap %.4f)\n", k, gap);
            CHECK(gap >= 0 && gap < 0.04, "MLA: prompt lookup committed different tokens than greedy decode (token %d, logit gap %.4f)", k, gap);
            free(lk);
        }
        printf("MLA prompt lookup: %lld forwards for 24 tokens, %lld of %lld proposals accepted\n", (long long) sp.forwards,
               (long long) sp.accepted, (long long) sp.proposals);
        CHECK(sp.accepted > 0, "MLA prompt lookup: no proposal accepted, so no multi-row verify was tested");
    }
    {   // placement: with the cache mostly in host memory (the latents of every width staged per layer for prompts) the
        // engine scores and decodes exactly as with all of it in fast memory (engines with unified memory ignore this)
        double rel;
        const int diff = compare_engines(e, &o, ENG_KV_BF16, "NSLM_KV_VRAM_MB", "0.2", &st, V, &rel, "MLA KV mostly in host memory");
        printf("MLA KV mostly in host memory: %d differences\n", diff);
        CHECK(diff == 0, "MLA: KV placement changed the results");
    }
    eng_close(e);
    mova_ref_close(ref);
    free(rl); free(el); free(rm); free(rv); free(em); free(ev); free(lg);
    test_batch(dir, "MLA");
}

int main(void) {
    const char* dir = "out/test/engine_model";
    char err[512] = "";
    if (write_model(dir, CONFIG, NULL, err, sizeof err)) { printf("FAIL: model folder: %s\n", err); return 1; }
    MovaRef* ref = mova_ref_open(dir, 8, err, sizeof err);
    if (!ref) { printf("FAIL: reference: %s\n", err); return 1; }
    {   // a folder without one layer's stacked experts (as an original checkpoint, which stores them per expert): the
        // reference refuses it, naming the tensor (was a NULL dereference in forward)
        const char* nx = "out/test/engine_model_no_experts";
        char e2[512] = "";
        const int ok = write_model(nx, CONFIG, "layers.2.mlp.experts.up_proj", e2, sizeof e2) == 0;
        CHECK(ok, "folder without experts: %s", e2);
        MovaRef* rx = ok ? mova_ref_open(nx, 1, e2, sizeof e2) : NULL;
        CHECK(ok && !rx && strstr(e2, "layers.2.mlp.experts.up_proj") && strstr(e2, "packed"), "reference without experts: %s",
              rx ? "opened" : e2);
        if (rx) mova_ref_close(rx);
    }
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = dir;
    o.resource_dir = getenv("NSLM_RES") ? getenv("NSLM_RES") : "out/res";
    o.max_seqs = 1;
    o.kv_tokens = 1024;
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e && (strstr(err, "no CUDA device") || strstr(err, "no Metal device"))) { printf("SKIP: %s\n", err); return 77; }   // a machine without a GPU
    if (!e) { printf("FAIL: eng_open: %s\n", err); return 1; }
    printf("%s\n", eng_describe(e));
    const int V = eng_vocab(e), N = 48, NS = 4, TK = 4, TKV = 2;
    uint64_t st = 7;
    int32_t ids[64];
    for (int i = 0; i < N; ++i) ids[i] = (int32_t) (rng_u32(&st) % (uint32_t) V);

    // reference: logits for rows 0 .. N-2 (predicting ids[1 ..]) and its routes
    float* rl = (float*) malloc(sizeof(float) * (size_t) N * V);
    int32_t* rm = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TK);
    int32_t* rv = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TKV);
    // two references: the prefill kernels' attention rounding (scoring forwards of more than 8 rows) and the decode
    // kernels' (single rows)
    float* rd = (float*) malloc(sizeof(float) * (size_t) N * V);
    mova_ref_attn_rounding(ref, 1);
    CHECK(mova_ref_forward(ref, ids, N - 1, 0, rl, rm, rv) == 0, "reference forward");
    mova_ref_attn_rounding(ref, 0);
    CHECK(mova_ref_forward(ref, ids, N - 1, 0, rd, NULL, NULL) == 0, "reference forward");

    // 1. prompt scoring: one forward of N - 1 rows
    float* el = (float*) malloc(sizeof(float) * (size_t) N * V);
    int32_t* em = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TK);
    int32_t* ev = (int32_t*) malloc(sizeof(int32_t) * (size_t) N * NS * TKV);
    eng_mova_routes(e, 1, N);
    CHECK(eng_score(e, 0, ids, 1, N - 1, el) == 0, "eng_score");
    CHECK(eng_mova_routes_read(e, N - 1, em, ev, NULL, NULL) == 0, "routes");
    eng_mova_routes(e, 0, 0);
    // A row whose own router choices differ is not comparable; nor is any row after one: a flip changes that row's
    // value-expert output, which later rows read through the KV cache.  Rows before the first flip must agree to the
    // engines' rounding; all rows together, to a BF16 network's chaotic drift (a stray ulp moves later roundings).
    int flips = 0, clean = 0, am_same = 0;
    double worst = 0, mean = 0;
    for (int t = 0; t < N - 1; ++t) {
        const int same = same_choice(em + t * NS * TK, rm + t * NS * TK, NS, TK) && same_choice(ev + t * NS * TKV, rv + t * NS * TKV, NS, TKV);
        flips += !same;
        const double re = row_err(el + (size_t) t * V, rl + (size_t) t * V, V);
        am_same += argmax(el + (size_t) t * V, V) == argmax(rl + (size_t) t * V, V);
        mean += re;
        if (!flips) { ++clean; worst = fmax(worst, re); }
    }
    mean /= N - 1;
    printf("scoring: %d rows, %d before the first router flip (%d rows flipped); logit error: those rows max %.2e, all rows "
           "mean %.2e; argmax equal %d / %d\n", N - 1, clean, flips, worst, mean, am_same, N - 1);
    CHECK(flips <= (N - 1) / 4, "scoring: %d rows with a different router choice", flips);
    CHECK(clean >= (N - 1) / 4, "scoring: only %d rows before the first router flip", clean);   // else worst says nothing
    CHECK(worst < 0.02, "scoring: logit error %.3e before the first flip", worst);
    CHECK(mean < 0.05, "scoring: mean logit error %.3e", mean);
    CHECK(am_same >= (N - 1) * 9 / 10, "scoring: argmax equal in %d of %d rows", am_same, N - 1);

    // 2. decode: prefill 8 tokens, then step through the rest one row at a time
    const int P0 = 8;
    CHECK(eng_prefill(e, 0, ids, P0) == 0, "prefill");
    float* lg = (float*) malloc(sizeof(float) * (size_t) V);
    double dworst = 0, dmean = 0;
    int dsame = 0;
    for (int t = P0 - 1; t < N - 1; ++t) {
        CHECK(eng_step(e, 0, lg) == 0, "step");
        const double re = row_err(lg, rd + (size_t) t * V, V);
        dworst = fmax(dworst, re);
        dmean += re / (N - P0);
        dsame += argmax(lg, V) == argmax(rd + (size_t) t * V, V);
        eng_push(e, 0, ids[t + 1]);
    }
    printf("decode: %d steps, logit error max %.2e, mean %.2e (all rows), argmax equal %d / %d\n", N - P0, dworst, dmean,
           dsame, N - P0);
    // the same bounds as scoring's all-row ones (rows after a router flip may differ more: a gross bound for the max)
    CHECK(dmean < 0.05, "decode: mean logit error %.3e", dmean);
    CHECK(dworst < 0.25, "decode: logit error %.3e", dworst);
    CHECK(dsame >= (N - P0) * 9 / 10, "decode: argmax equal in %d of %d steps", dsame, N - P0);

    // 3. prompt lookup must commit the tokens of plain greedy decode (a repetitive prompt so drafts are offered)
    int32_t rep[64];
    for (int i = 0; i < 40; ++i) rep[i] = ids[i % 10];
    int32_t ar[24], pl[24];
    const int seq = 0;
    EngStats sp;
    memset(&sp, 0, sizeof sp);
    CHECK(eng_prefill(e, 0, rep, 40) == 0 && eng_generate(e, &seq, 1, 24, ENG_MODE_AR, ar, NULL) == 0, "AR generate");
    CHECK(eng_prefill(e, 0, rep, 40) == 0 && eng_generate(e, &seq, 1, 24, ENG_MODE_PL, pl, &sp) == 0, "PL generate");
    CHECK(!memcmp(ar, pl, sizeof ar), "prompt lookup committed different tokens than greedy decode");
    printf("prompt lookup: %lld forwards for 24 tokens, %lld of %lld proposals accepted\n", (long long) sp.forwards,
           (long long) sp.accepted, (long long) sp.proposals);

    // 4. long prompts: scoring with route capture (forward by forward) and without (an engine may then run the prompt
    // layer by layer) must give the same logits bit for bit: same kernels, same chunks, same positions.  600 rows: two
    // chunks; 517: a last chunk of 5 rows, which takes the decode kernels (and must not disturb the layer prefetch);
    // also with an expert cache too small for one layer's experts, so that every layer's experts are copied in
    {
        EngOpts os = o;
        set_env("NSLM_EXPERT_VRAM_MB", "13");
        Eng* es = eng_open(&os, err, sizeof err);
        set_env("NSLM_EXPERT_VRAM_MB", NULL);
        CHECK(es != NULL, "eng_open (small expert cache): %s", err);
        const int lens[2] = {600, 517};
        for (int k = 0; k < 4; ++k) {
            Eng* ek = k < 2 ? e : es;
            const int NL = lens[k & 1];
            if (!ek) continue;
            int32_t* lid = (int32_t*) malloc(sizeof(int32_t) * NL);
            for (int i = 0; i < NL; ++i) lid[i] = (int32_t) (rng_u32(&st) % (uint32_t) V);
            const int from = NL - 70, cnt = 70;
            float* la = (float*) malloc(sizeof(float) * (size_t) cnt * V);
            float* lb = (float*) malloc(sizeof(float) * (size_t) cnt * V);
            CHECK(eng_score(ek, 0, lid, from, cnt, la) == 0, "long scoring");
            eng_mova_routes(ek, 1, NL);
            CHECK(eng_score(ek, 0, lid, from, cnt, lb) == 0, "long scoring with routes");
            eng_mova_routes(ek, 0, 0);
            int diff = 0;
            for (size_t i = 0; i < (size_t) cnt * V; ++i) diff += la[i] != lb[i];
            printf("long prompt (%d tokens, %s expert cache): %d of %d logits differ between the two forwards\n", NL,
                   k < 2 ? "full" : "small", diff, cnt * V);
            CHECK(diff == 0, "long prompt (%d tokens): layer-major and chunked forwards differ", NL);
            free(lid); free(la); free(lb);
        }
        if (es) eng_close(es);
    }

    // 5. cached prefill: a shared prefix is reused and the result equals a full prefill's next step
    int reused = 0;
    CHECK(eng_prefill(e, 0, ids, 30) == 0, "prefill 30");
    CHECK(eng_prefill_cached(e, 0, ids, 36, &reused) == 0 && reused == 29, "cached prefill reused %d", reused);
    CHECK(eng_len(e, 0) == 36, "length after cached prefill");

    // 5b. a server's sequence: a short request, then a long one sharing a prefix (a cached prefill of many rows), then
    // decoding; the long request's tokens must equal a fresh prefill's
    {
        const int NL = 300;
        int32_t* lp = (int32_t*) malloc(sizeof(int32_t) * NL);
        for (int i = 0; i < NL; ++i) lp[i] = i < 4 ? ids[i] : (int32_t) (rng_u32(&st) % (uint32_t) V);
        int32_t ga[16], gb[16];
        int reused2 = 0;
        CHECK(eng_prefill(e, 0, ids, 20) == 0 && eng_generate(e, &seq, 1, 8, ENG_MODE_AR, ga, NULL) == 0, "short request");
        CHECK(eng_prefill_cached(e, 0, lp, NL, &reused2) == 0 && reused2 == 4, "long cached prefill (reused %d)", reused2);
        CHECK(eng_generate(e, &seq, 1, 16, ENG_MODE_AR, ga, NULL) == 0, "decode after the long cached prefill");
        CHECK(eng_prefill(e, 0, lp, NL) == 0 && eng_generate(e, &seq, 1, 16, ENG_MODE_AR, gb, NULL) == 0, "fresh prefill");
        CHECK(!memcmp(ga, gb, sizeof ga), "a cached prefill decodes differently from a fresh one");
        printf("long cached prefill: %s\n", memcmp(ga, gb, sizeof ga) ? "DIFFERS" : "same tokens as a fresh prefill");
        free(lp);
    }

    // 6. placement must not change results: an engine allowed to keep only part of the experts (NSLM_EXPERT_VRAM_MB)
    // or of the KV cache (NSLM_KV_VRAM_MB) in fast memory scores and decodes exactly as this one does (engines with
    // unified memory ignore both)
    {
        double rel;
        int diff = compare_engines(e, &o, ENG_KV_BF16, "NSLM_EXPERT_VRAM_MB", "13", &st, V, &rel, "small expert cache");
        printf("small expert cache: %d differences\n", diff);
        CHECK(diff == 0, "a smaller expert cache changed the results");
        diff = compare_engines(e, &o, ENG_KV_BF16, "NSLM_KV_VRAM_MB", "1", &st, V, &rel, "KV mostly in host memory");
        printf("KV mostly in host memory: %d differences\n", diff);
        CHECK(diff == 0, "KV placement changed the results");
        // the prefill GEMM with exact (f32) seed weights: a different rounding point, so close, not equal
        diff = compare_engines(e, &o, ENG_KV_BF16, "NSLM_SEED_GEMM_F32", "1", &st, V, &rel, "f32 seed weights in the GEMM");
        printf("f32 seed weights in the GEMM: %d values differ, relative logit difference mean %.2e\n", diff, rel);
        CHECK(rel < 0.05, "f32 seed weights in the GEMM: mean relative logit difference %.3e", rel);
    }

    // 7. the 8-bit KV cache: close to BF16 (its rounding is of BF16's size), and placement-invariant itself
    {
        EngOpts o8 = o;
        o8.kv_format = ENG_KV_Q8;
        Eng* e8 = eng_open(&o8, err, sizeof err);
        if (!e8 && strstr(err, "not supported")) printf("Q8 KV: skipped (%s)\n", err);   // a BF16-only engine (Metal)
        else CHECK(e8 != NULL, "eng_open (Q8 KV): %s", err);
        if (e8) {
            double rel;
            const int diff = compare_engines(e, &o, ENG_KV_Q8, NULL, NULL, &st, V, &rel, "Q8 KV against BF16");
            printf("Q8 KV against BF16 KV: %d values differ, relative logit difference mean %.2e\n", diff, rel);
            // a gross-error bound only: the random model amplifies 8-bit rounding chaotically; the 8-bit cache's quality is
            // measured on the real model (nslm-mova-refcheck --kv q8: KL 1e-2 against the CPU reference, 5e-3 for BF16)
            CHECK(rel < 0.25, "Q8 KV: mean relative logit difference %.3e", rel);
            const int d2 = compare_engines(e8, &o8, ENG_KV_Q8, "NSLM_KV_VRAM_MB", "0.5", &st, V, &rel, "Q8 KV mostly in host memory");
            printf("Q8 KV mostly in host memory: %d differences\n", d2);
            CHECK(d2 == 0, "Q8 KV placement changed the results");
            eng_close(e8);
        }
    }

    eng_close(e);
    mova_ref_close(ref);
    free(rl); free(rd); free(rm); free(rv); free(el); free(em); free(ev); free(lg);
    test_batch(dir, "GQA");
    test_mla();
    printf("test_engine: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
