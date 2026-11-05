// tests/test_qwen4_api_mtp.c - eng_generate with MTP drafts (Q4E_DRAFTS, engine/qwen4_api.c) on tests/data/qwen4_tiny,
// through engine/engine.c (CUDA or Metal). Plan gate MB: at 1 to 3 drafts, the tokens, the history and the next logits are
// those of greedy decode without drafts, bit for bit, in chunks of any size, after a pushed token, after a rewind and up
// to the last position. The tiny model's head rarely drafts its next token: after 8 times token 216, the first step's draft
// 1 is accepted (the C reference's bonus, draft and next token lead the runner-up by 5.6% of the top logit or more).
// Skipped without a GPU.
#define _DEFAULT_SOURCE   // glibc: setenv / unsetenv under -std=c99 (Windows: _putenv_s)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_api.h"
#include "format.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)
#define DIR_ "tests/data/qwen4_tiny"
#define NT 40   // tokens decoded after a prompt of 20

static void set_env(const char* k, const char* v) {   // v = NULL: unset
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

// The tiny model (or dir) with Q4E_DRAFTS = k (0: unset) and max_pos positions
static Eng* open_k(const char* dir, int k, int max_pos, char* err, int errlen) {
    char v[16];
    snprintf(v, sizeof v, "%d", k);
    set_env("Q4E_DRAFTS", k ? v : NULL);
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = dir;
    o.max_seqs = 1;
    o.kv_tokens = max_pos;
    Eng* e = eng_open(&o, err, errlen);
    set_env("Q4E_DRAFTS", NULL);
    return e;
}

// The prompt ids[0 .. n), then nt tokens of greedy decode in eng_generate calls of chunk[0], chunk[1], ... tokens (cycled);
// the logits after them in lg (eng_step). 0, or -1
static int run(Eng* e, const int32_t* ids, int n, int nt, const int* chunk, int nc, int32_t* out, float* lg, EngStats* st) {
    const int seq = 0;
    if (eng_prefill(e, 0, ids, n)) return -1;
    for (int j = 0, c = 0; j < nt; ++c) {
        const int g = chunk[c % nc] < nt - j ? chunk[c % nc] : nt - j;
        if (eng_generate(e, &seq, 1, g, ENG_MODE_AR, out + j, st)) return -1;
        j += g;
    }
    return lg ? eng_step(e, 0, lg) : 0;
}

int main(void) {
    char err[512] = "";
    StFile st;
    if (st_open_file(&st, "tests/data/qwen4_tiny_golden.safetensors", err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
    const int32_t* ids = (const int32_t*) st_data(&st, st_find(&st, "ids"));
    Eng* e0 = open_k(DIR_, 0, 256, err, sizeof err);   // the engine's own open fails without a GPU
    if (!e0 && strstr(err, "device")) { printf("SKIP: %s\n", err); return 77; }
    CHECK(e0 != NULL, "open: %s", err);
    if (!e0) { printf("test_qwen4_api_mtp: FAIL\n"); return 1; }
    const int V = eng_vocab(e0), one[1] = {NT}, odd[4] = {1, 2, 3, 5};
    float* want = (float*) malloc(sizeof(float) * (size_t) V);
    float* lg = (float*) malloc(sizeof(float) * (size_t) V);
    int32_t plain[NT], spec[NT], seq[20 + NT];

    // Without drafts: one call, or calls of 1, 2, 3 and 5 tokens: the same decode
    CHECK(strstr(eng_describe(e0), "MTP") == NULL, "no Q4E_DRAFTS: %s", eng_describe(e0));
    CHECK(run(e0, ids, 20, NT, one, 1, plain, want, NULL) == 0 && eng_len(e0, 0) == 20 + NT, "decode without drafts");
    CHECK(run(e0, ids, 20, NT, odd, 4, spec, lg, NULL) == 0 && !memcmp(spec, plain, sizeof plain) && !memcmp(lg, want, sizeof(float) * (size_t) V),
          "decode in calls of 1, 2, 3 and 5 tokens differs");

    // A pushed token that is not greedy decode's, then more decode; a rewind into the decoded tokens
    int32_t pa[7 + 10], other, rep[8], rp[16];
    float* want_p = (float*) malloc(sizeof(float) * (size_t) V);
    float* want_r = (float*) malloc(sizeof(float) * (size_t) V);
    const int s0 = 0;
    CHECK(run(e0, ids, 20, 7, one, 1, pa, NULL, NULL) == 0 && eng_push(e0, 0, other = (pa[6] + 1) % V) == 0 &&
          eng_generate(e0, &s0, 1, 10, ENG_MODE_AR, pa + 7, NULL) == 0 && eng_step(e0, 0, want_p) == 0, "push without drafts");
    memcpy(seq, ids, sizeof(int32_t) * 20);
    memcpy(seq + 20, plain, sizeof(int32_t) * 13);
    CHECK(eng_prefill(e0, 0, seq, 33) == 0 && eng_step(e0, 0, want_r) == 0, "the rewind's reference");
    float* want_a = (float*) malloc(sizeof(float) * (size_t) V);
    for (int i = 0; i < 8; ++i) rep[i] = 216;
    CHECK(run(e0, rep, 8, 16, one, 1, rp, want_a, NULL) == 0, "decode after 8 times token 216");

    for (int k = 1; k <= 3; ++k) {
        Eng* e = open_k(DIR_, k, 256, err, sizeof err);
        CHECK(e != NULL && strstr(eng_describe(e), "MTP draft"), "open with %d drafts: %s", k, e ? eng_describe(e) : err);
        if (!e) continue;
        EngStats gs;
        memset(&gs, 0, sizeof gs);
        CHECK(run(e, ids, 20, NT, one, 1, spec, lg, &gs) == 0, "decode with %d drafts", k);
        int same = 0;
        while (same < NT && spec[same] == plain[same]) ++same;
        printf("%d drafts a step: %lld forwards, %lld of %lld drafts accepted, the first %d tokens as without drafts\n", k,
               (long long) gs.forwards, (long long) gs.accepted, (long long) gs.proposals, same);
        CHECK(same == NT, "%d drafts a step change token %d", k, same);
        CHECK(!memcmp(lg, want, sizeof(float) * (size_t) V) && eng_len(e, 0) == 20 + NT, "%d drafts: the next logits or the length (%d) differ", k, eng_len(e, 0));
        CHECK(gs.tokens == NT && gs.cycles + gs.accepted == NT && gs.proposals > 0 && gs.accepted <= gs.proposals &&
              gs.forwards <= gs.cycles + 1, "%d drafts: stats tokens %lld steps %lld forwards %lld drafts %lld accepted %lld", k,
              (long long) gs.tokens, (long long) gs.cycles, (long long) gs.forwards, (long long) gs.proposals, (long long) gs.accepted);

        memset(&gs, 0, sizeof gs);
        CHECK(run(e, rep, 8, 16, one, 1, spec, lg, &gs) == 0 && !memcmp(spec, rp, sizeof rp) && !memcmp(lg, want_a, sizeof(float) * (size_t) V),
              "%d drafts after 8 times token 216 differ", k);
        printf("%d drafts a step after 8 times token 216: %lld of %lld drafts accepted\n", k, (long long) gs.accepted, (long long) gs.proposals);
        CHECK(gs.accepted >= 1 && gs.cycles + gs.accepted == 16, "%d drafts after 8 times token 216: %lld accepted", k, (long long) gs.accepted);

        CHECK(run(e, ids, 20, NT, odd, 4, spec, lg, NULL) == 0 && !memcmp(spec, plain, sizeof plain) && !memcmp(lg, want, sizeof(float) * (size_t) V),
              "%d drafts in calls of 1, 2, 3 and 5 tokens differ", k);

        int32_t q[7 + 10];
        CHECK(run(e, ids, 20, 7, one, 1, q, NULL, NULL) == 0 && eng_push(e, 0, other) == 0 && eng_generate(e, &s0, 1, 10, ENG_MODE_AR, q + 7, NULL) == 0 &&
              eng_step(e, 0, lg) == 0 && !memcmp(q, pa, sizeof q) && !memcmp(lg, want_p, sizeof(float) * (size_t) V), "%d drafts after a pushed token differ", k);

        CHECK(run(e, ids, 20, NT, one, 1, spec, NULL, NULL) == 0 && eng_rewind(e, 0, 33) == 0 && eng_len(e, 0) == 33 && eng_step(e, 0, lg) == 0 &&
              !memcmp(lg, want_r, sizeof(float) * (size_t) V), "%d drafts, then a rewind to 33: the logits differ", k);
        CHECK(eng_push(e, 0, plain[13]) == 0 && eng_generate(e, &s0, 1, NT - 14, ENG_MODE_AR, spec, NULL) == 0 && !memcmp(spec, plain + 14, sizeof(int32_t) * (NT - 14)),
              "%d drafts after the rewind differ", k);
        eng_close(e);
    }

    // Up to the last of 64 positions: the drafts stop at the end of the context, and a 65th token fails as without drafts
    {
        Eng* a = open_k(DIR_, 0, 64, err, sizeof err), *b = open_k(DIR_, 3, 64, err, sizeof err);
        int32_t ta[44], tb[44], x;
        CHECK(a && b && run(a, ids, 20, 44, one, 1, ta, NULL, NULL) == 0 && run(b, ids, 20, 44, one, 1, tb, NULL, NULL) == 0 &&
              !memcmp(ta, tb, sizeof ta) && eng_len(b, 0) == 64, "64 positions with 3 drafts: %s", err);
        CHECK(a && b && eng_generate(a, &s0, 1, 1, ENG_MODE_AR, &x, NULL) == -1 && eng_generate(b, &s0, 1, 1, ENG_MODE_AR, &x, NULL) == -1,
              "a 65th position");
        eng_close(a);
        eng_close(b);
    }

    Eng* bad = open_k(DIR_, 4, 256, err, sizeof err);
    CHECK(!bad && strstr(err, "Q4E_DRAFTS"), "4 drafts open: %s", err);
    bad = open_k("tests/data/qwen4_tiny_attn", 1, 256, err, sizeof err);
    CHECK(!bad && strstr(err, "no MTP head"), "drafts without mtp.* open: %s", err);
    eng_close(e0);
    free(want); free(lg); free(want_p); free(want_r); free(want_a);
    st_close_file(&st);
    printf("test_qwen4_api_mtp: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
