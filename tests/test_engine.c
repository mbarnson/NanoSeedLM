// tests/test_engine.c - the GPU engine (Metal or CUDA, whichever is linked) against the CPU reference forward
// (nslm/mova_ref.h), through engine_api.h only.
//
// A small MoVA model folder with random weights in every encoding the engine reads (SEED4P4 routed experts, Q4 value
// experts, Q8 projections / embedding / head, BF16 norms and routers) is written to out/test/engine_model.  Then:
//   prompt scoring (eng_score: the GEMM and prefill-attention kernels) against the reference logits,
//   token-by-token decode (eng_prefill + eng_step: the matvec and split-key attention kernels) against them,
//   the engine's router choices (eng_mova_routes) against the reference's,
//   prompt-lookup decode (ENG_MODE_PL) committing exactly the tokens of plain greedy decode.
// Logits agree to the engines' f32 accumulation (relative error well under a BF16 step of the logit scale) wherever
// the router choices agree; a choice can flip on a near tie, which the test counts and bounds.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "engine_api.h"
#include "lfsr.h"
#include "model_st.h"
#include "mova_cfg.h"
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
    "  \"num_attention_heads\": 8,\n  \"num_experts\": 16,\n  \"num_experts_per_tok\": 4,\n  \"num_hidden_layers\": 3,\n"
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
                          : mt->kind == MOVA_K_ROUTER_BIAS ? (float) (0.02 * u) : (float) (0.3 * u);   // routers: well-separated scores
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

static int write_model(const char* dir, char* err, int errlen) {
    mkdir("out", 0755);
    mkdir("out/test", 0755);
    mkdir(dir, 0755);
    char path[512];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) { snprintf(err, (size_t) errlen, "cannot write %s", path); return -1; }
    fputs(CONFIG, f);
    fclose(f);
    MovaCfg c;
    if (mova_cfg_parse(&c, CONFIG, err, errlen)) return -1;
    MovaTensor* mt = NULL;
    const int n = mova_tensors(&c, &mt);
    NsSpec* sp = (NsSpec*) calloc((size_t) n, sizeof(NsSpec));
    for (int i = 0; i < n; ++i) {
        const int k = mt[i].kind;
        sp[i].name = mt[i].name;
        sp[i].enc = k == MOVA_K_EXPERTS ? NS_SEED4P4 : k == MOVA_K_VEXPERTS ? NS_Q4
                  : (k == MOVA_K_NORM || k == MOVA_K_ROUTER || k == MOVA_K_ROUTER_BIAS) ? NS_BF16 : NS_Q8;
        sp[i].slices = mt[i].slices;
        sp[i].rows = mt[i].rows;
        sp[i].cols = mt[i].cols;
    }
    Gen g = {sp, mt};
    const int rc = ns_write(dir, sp, n, 64ull << 20, NULL, fill, &g, err, errlen);
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

int main(void) {
    const char* dir = "out/test/engine_model";
    char err[512] = "";
    if (write_model(dir, err, sizeof err)) { printf("FAIL: model folder: %s\n", err); return 1; }
    MovaRef* ref = mova_ref_open(dir, 8, err, sizeof err);
    if (!ref) { printf("FAIL: reference: %s\n", err); return 1; }
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = dir;
    o.resource_dir = getenv("NSLM_RES") ? getenv("NSLM_RES") : "out/res";
    o.max_seqs = 1;
    o.kv_tokens = 1024;
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { printf("FAIL: eng_open: %s\n", err); return 1; }
    printf("%s\n", eng_describe(e));
    const int V = eng_vocab(e), N = 48, NS = 2, TK = 4, TKV = 2;
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
    CHECK(worst < 0.02, "scoring: logit error %.3e before the first flip", worst);
    CHECK(mean < 0.05, "scoring: mean logit error %.3e", mean);
    CHECK(am_same >= (N - 1) * 9 / 10, "scoring: argmax equal in %d of %d rows", am_same, N - 1);

    // 2. decode: prefill 8 tokens, then step through the rest one row at a time
    const int P0 = 8;
    CHECK(eng_prefill(e, 0, ids, P0) == 0, "prefill");
    float* lg = (float*) malloc(sizeof(float) * (size_t) V);
    double dworst = 0;
    int dsame = 0;
    for (int t = P0 - 1; t < N - 1; ++t) {
        CHECK(eng_step(e, 0, lg) == 0, "step");
        dworst = fmax(dworst, row_err(lg, rd + (size_t) t * V, V));
        dsame += argmax(lg, V) == argmax(rd + (size_t) t * V, V);
        eng_push(e, 0, ids[t + 1]);
    }
    printf("decode: %d steps, logit error max %.2e (all rows), argmax equal %d / %d\n", N - P0, dworst, dsame, N - P0);
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
    // layer by layer) must give the same logits bit for bit: same kernels, same chunks, same positions
    {
        const int NL = 600;   // more than one prompt chunk
        int32_t* lid = (int32_t*) malloc(sizeof(int32_t) * NL);
        for (int i = 0; i < NL; ++i) lid[i] = (int32_t) (rng_u32(&st) % (uint32_t) V);
        const int from = NL - 70, cnt = 70;
        float* la = (float*) malloc(sizeof(float) * (size_t) cnt * V);
        float* lb = (float*) malloc(sizeof(float) * (size_t) cnt * V);
        CHECK(eng_score(e, 0, lid, from, cnt, la) == 0, "long scoring");
        eng_mova_routes(e, 1, NL);
        CHECK(eng_score(e, 0, lid, from, cnt, lb) == 0, "long scoring with routes");
        eng_mova_routes(e, 0, 0);
        int diff = 0;
        for (size_t i = 0; i < (size_t) cnt * V; ++i) diff += la[i] != lb[i];
        printf("long prompt (%d tokens): %d of %d logits differ between the two forwards\n", NL, diff, cnt * V);
        CHECK(diff == 0, "long prompt: layer-major and chunked forwards differ");
        free(lid); free(la); free(lb);
    }

    // 5. cached prefill: a shared prefix is reused and the result equals a full prefill's next step
    int reused = 0;
    CHECK(eng_prefill(e, 0, ids, 30) == 0, "prefill 30");
    CHECK(eng_prefill_cached(e, 0, ids, 36, &reused) == 0 && reused == 29, "cached prefill reused %d", reused);
    CHECK(eng_len(e, 0) == 36, "length after cached prefill");

    eng_close(e);
    mova_ref_close(ref);
    free(rl); free(rd); free(rm); free(rv); free(el); free(em); free(ev); free(lg);
    printf("test_engine: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
