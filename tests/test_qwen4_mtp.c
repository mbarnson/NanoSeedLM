// tests/test_qwen4_mtp.c - the qwen4 engine's MTP head (Q4EOpts.mtp, CUDA or Metal) on tests/data/qwen4_tiny: drafts
// against the C reference, verify forwards that keep each row's state, and greedy decode with drafts. Skipped without a GPU.
#define _DEFAULT_SOURCE   // glibc: setenv / unsetenv under -std=c99 (Windows: _putenv_s)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "qwen4_eng.h"
#include "qwen4_ref.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)
#define DIR_ "tests/data/qwen4_tiny"

static void set_env(const char* k, const char* v) {   // v = NULL: unset
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}
static int argmax(const float* l, int n) { int b = 0; for (int v = 1; v < n; ++v) if (l[v] > l[b]) b = v; return b; }

// The C reference's bonus and 3 drafts after ids[0..n) (test_qwen4_ref checks the head against mlx-vlm's): the bonus
// given, or the arg max when bonus < 0.  1 when each draft leads the next logit by over 10% of the largest (BF16-safe)
static int ref_drafts(Qwen4Ref* r, const int32_t* ids, int n, int bonus, int32_t* want) {
    const int V = qwen4_ref_cfg(r)->vocab, HD = qwen4_ref_cfg(r)->hc * qwen4_ref_cfg(r)->d;
    float* lg = (float*) malloc(sizeof(float) * (size_t) (n + 3) * V);
    float* H = (float*) malloc(sizeof(float) * (size_t) (n + 3) * HD), *Hm = (float*) malloc(sizeof(float) * (size_t) (n + 3) * HD);
    int32_t* tok = (int32_t*) malloc(sizeof(int32_t) * (size_t) (n + 3));
    int clear = 1;
    qwen4_ref_forward_h(r, ids, n, lg, H);
    for (int t = 0; t + 1 < n; ++t) tok[t] = ids[t + 1];
    want[0] = tok[n - 1] = bonus >= 0 ? bonus : argmax(lg + (size_t) (n - 1) * V, V);
    for (int j = 0; j < 3; ++j) {
        qwen4_ref_mtp(r, tok, H, n + j, lg, Hm);
        const float* l = lg + (size_t) (n + j - 1) * V;
        const int b = argmax(l, V);
        float second = -1e30f;
        for (int v = 0; v < V; ++v) if (v != b && l[v] > second) second = l[v];
        clear &= l[b] - second > 0.1f * (l[b] > 0 ? l[b] : -l[b]);
        want[1 + j] = tok[n + j] = b;
        memcpy(H + (size_t) (n + j) * HD, Hm + (size_t) (n + j - 1) * HD, sizeof(float) * (size_t) HD);
    }
    free(lg); free(H); free(Hm); free(tok);
    return clear;
}

// The prompt ids[0..n) in rows of at most `rows` (0: n - 1 rows without logits, then one), the last row's logits in lg
static int prompt(Q4Eng* e, const int32_t* ids, int n, int rows, float* lg, float* buf) {
    q4e_reset(e);
    const int V = q4e_cfg(e)->vocab;
    if (!rows) return (n > 1 && q4e_forward(e, ids, n - 1, NULL)) || q4e_forward(e, ids + n - 1, 1, lg);
    for (int at = 0; at < n; at += rows) {
        const int T = n - at < rows ? n - at : rows;
        if (q4e_forward(e, ids + at, T, buf)) return -1;
        if (at + T == n) memcpy(lg, buf + (size_t) (T - 1) * V, sizeof(float) * (size_t) V);
    }
    return 0;
}

// Greedy decode of nt tokens after the prompt ids[0..n): one row a step (k = 0), or k drafts verified a step
static int decode(Q4Eng* e, const int32_t* ids, int n, int k, int nt, int32_t* out, int* accepted) {
    const int V = q4e_cfg(e)->vocab;
    float* lg = (float*) malloc(sizeof(float) * (size_t) V * (Q4E_MAX_DRAFTS + 1));
    int rc = prompt(e, ids, n, 0, lg, NULL), got = 0;
    *accepted = 0;
    while (!rc && got < nt) {
        int32_t row[Q4E_MAX_DRAFTS + 1];
        row[0] = argmax(lg, V);   // the bonus
        if (!k) { out[got++] = row[0]; rc = q4e_forward(e, row, 1, lg); continue; }
        int m = 1;
        rc = q4e_draft(e, k, row + 1) || q4e_forward_keep(e, row, k + 1, lg);
        while (!rc && m <= k && argmax(lg + (size_t) (m - 1) * V, V) == row[m]) ++m;   // rows 0 .. m - 1 hold
        if (!rc) rc = q4e_accept(e, m);
        for (int i = 0; i < m && got < nt; ++i) out[got++] = row[i];
        *accepted += m - 1;
        memmove(lg, lg + (size_t) (m - 1) * V, sizeof(float) * (size_t) V);
    }
    free(lg);
    return rc;
}

int main(void) {
    char err[512] = "";
    StFile st;
    if (st_open_file(&st, "tests/data/qwen4_tiny_golden.safetensors", err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
    const StEntry* ei = st_find(&st, "ids");
    const int N = (int) ei->shape[0];
    const int32_t* ids = (const int32_t*) st_data(&st, ei);
    Q4EOpts o;
    memset(&o, 0, sizeof o);
    o.max_pos = 256;
    Q4Eng* probe = q4e_open(DIR_, &o, err, sizeof err);   // the engine's own open fails without a GPU
    if (!probe && strstr(err, "device")) { printf("SKIP: %s\n", err); return 77; }
    q4e_close(probe);
    o.mtp = Q4E_MAX_DRAFTS;
    CHECK(!q4e_open("tests/data/qwen4_tiny_attn", &o, err, sizeof err) && strstr(err, "no MTP head"), "a folder without mtp.*: %s", err);
    Qwen4Ref* r = qwen4_ref_open(DIR_, err, sizeof err);
    Q4Eng* e = r ? q4e_open(DIR_, &o, err, sizeof err) : NULL;
    CHECK(r && e, "open: %s", err);
    if (!r || !e) { printf("test_qwen4_mtp: FAIL\n"); return 1; }
    const int V = q4e_cfg(e)->vocab;
    float* lg = (float*) malloc(sizeof(float) * (size_t) V * N);
    float* buf = (float*) malloc(sizeof(float) * (size_t) V * N);
    int32_t d[3];
    CHECK(q4e_draft(e, 1, d) == -1, "a draft before any logits");

    // Lengths where the reference's bonus and drafts lead the next logit by over 10% of the largest: BF16 rounding keeps them
    const int lens[2] = {16, 45}, rows[3] = {0, 8, 1};
    for (int i = 0; i < 2; ++i) {
        int32_t want[4];
        ref_drafts(r, ids, lens[i], -1, want);
        for (int w = 0; w < 3; ++w) {
            CHECK(prompt(e, ids, lens[i], rows[w], lg, buf) == 0 && q4e_draft(e, 3, d) == 0, "prompt of %d: %s", lens[i], q4e_error(e));
            CHECK(argmax(lg, V) == want[0] && !memcmp(d, want + 1, sizeof d), "%d tokens in rows of %d: bonus %d drafts %d %d %d, want %d %d %d %d",
                  lens[i], rows[w], argmax(lg, V), d[0], d[1], d[2], want[0], want[1], want[2], want[3]);
        }
        printf("prompt of %d: bonus %d, drafts %d %d %d (the C reference's)\n", lens[i], want[0], want[1], want[2], want[3]);
    }
    {   // a bonus that is not the arg max (a sampled token): the head drafts on it, as the reference does
        int32_t want[4], ref[4];
        ref_drafts(r, ids, 45, -1, ref);
        int b = -1;
        for (int t = 1; t < 64 && b < 0; ++t) if ((ref[0] + t) % V != ref[0] && ref_drafts(r, ids, 45, (ref[0] + t) % V, want)) b = (ref[0] + t) % V;
        CHECK(b >= 0, "no other bonus with clear drafts in the reference");
        if (b >= 0) {
            CHECK(prompt(e, ids, 45, 0, lg, buf) == 0 && q4e_draft_from(e, b, 3, d) == 0 && !memcmp(d, want + 1, sizeof d),
                  "bonus %d: drafts %d %d %d, want %d %d %d (%s)", b, d[0], d[1], d[2], want[1], want[2], want[3], q4e_error(e));
            CHECK(prompt(e, ids, 45, 0, lg, buf) == 0 && q4e_draft_from(e, -1, 3, d) == 0 && !memcmp(d, ref + 1, sizeof d),
                  "q4e_draft_from(-1) is q4e_draft");
            CHECK(q4e_draft_from(e, V, 1, d) == -1, "a bonus past the vocabulary");
        }
    }
    {   // the drafts run as a CUDA graph for each k: without graphs, the same drafts
        set_env("NSLM_NO_GRAPH", "1");
        Q4Eng* ng = q4e_open(DIR_, &o, err, sizeof err);
        set_env("NSLM_NO_GRAPH", NULL);
        int32_t dn[3];
        CHECK(ng && prompt(ng, ids, 45, 8, lg, buf) == 0 && q4e_draft(ng, 3, dn) == 0 && prompt(e, ids, 45, 8, lg, buf) == 0 &&
              q4e_draft(e, 3, d) == 0 && !memcmp(d, dn, sizeof d), "drafts without CUDA graphs differ: %s", ng ? q4e_error(ng) : err);
        q4e_close(ng);
    }
    {   // the head's experts through a small expert cache (its sparse layer n_layer): the same drafts
        Q4EOpts oc = o;
        oc.cache_gb = 1e-5;
        Q4Eng* ec = q4e_open(DIR_, &oc, err, sizeof err);
        int32_t dc[3];
        uint64_t h = 0, m = 0;
        int slots = 0;
        CHECK(ec && prompt(ec, ids, 45, 8, lg, buf) == 0 && q4e_draft(ec, 3, dc) == 0 && prompt(e, ids, 45, 8, lg, buf) == 0 &&
              q4e_draft(e, 3, d) == 0 && !memcmp(d, dc, sizeof d), "drafts through the expert cache differ: %s", ec ? q4e_error(ec) : err);
        if (ec) q4e_cache_stats(ec, &h, &m, &slots);
        printf("drafts through a cache of %d slots: %llu misses\n", slots, (unsigned long long) m);
        CHECK(slots == 0 || m > 0, "the cache never missed");   // Metal: no expert cache
        q4e_close(ec);
    }
    {   // mtp = 1: one draft, one state copy
        Q4EOpts o1 = o;
        o1.mtp = 1;
        Q4Eng* e1 = q4e_open(DIR_, &o1, err, sizeof err);
        CHECK(e1 && prompt(e1, ids, 16, 0, lg, NULL) == 0 && q4e_draft(e1, 2, d) == -1 && q4e_draft(e1, 1, d) == 0 &&
              q4e_forward_keep(e1, ids, 3, NULL) == -1, "an engine of one draft: %s", e1 ? q4e_error(e1) : err);
        q4e_close(e1);
    }
    CHECK(q4e_draft(e, 4, d) == -1 && q4e_forward_keep(e, ids, 5, NULL) == -1 && q4e_accept(e, 1) == -1, "out-of-range calls");

    // A verify forward of 4 rows, then rows 0 .. m - 1 kept: the next row's logits as without it, bit for bit
    for (int m = 1; m <= 4; ++m) {
        float* a = lg, *b = buf;
        CHECK(prompt(e, ids, 20, 0, a, NULL) == 0 && q4e_forward(e, ids + 20, m, NULL) == 0 && q4e_forward(e, ids + 20 + m, 1, a) == 0, "plain");
        CHECK(prompt(e, ids, 20, 0, b, NULL) == 0 && q4e_forward_keep(e, ids + 20, 4, NULL) == 0 && q4e_accept(e, m) == 0 &&
              q4e_pos(e) == 20 + m && q4e_forward(e, ids + 20 + m, 1, b) == 0, "keep %d: %s", m, q4e_error(e));
        CHECK(!memcmp(a, b, sizeof(float) * (size_t) V), "keeping %d of 4 rows changes the next logits", m);
    }

    // A prompt past one chunk (rows 128: layer-major passes, the head per chunk) drafts as in one forward
    {
        int32_t lid[200], dp[3];
        for (int i = 0; i < 200; ++i) lid[i] = ids[i % N];
        Q4EOpts oc = o;
        oc.rows = 128;
        Q4Eng* ec = q4e_open(DIR_, &oc, err, sizeof err);
        CHECK(ec && prompt(ec, lid, 200, 0, lg, NULL) == 0 && q4e_draft(ec, 3, dp) == 0, "chunks: %s", ec ? q4e_error(ec) : err);
        CHECK(prompt(e, lid, 200, 0, buf, NULL) == 0 && q4e_draft(e, 3, d) == 0 && !memcmp(d, dp, sizeof d) && !memcmp(lg, buf, sizeof(float) * (size_t) V),
              "drafts after 200 rows in passes of 128 (%d %d %d) and in one forward (%d %d %d)", dp[0], dp[1], dp[2], d[0], d[1], d[2]);
        q4e_close(ec);
    }

    // Plan gate MB: greedy decode with drafts gives the tokens of greedy decode without
    int32_t plain[40], spec[40];
    int acc = 0;
    CHECK(decode(e, ids, 20, 0, 40, plain, &acc) == 0, "decode: %s", q4e_error(e));
    for (int k = 1; k <= Q4E_MAX_DRAFTS; ++k) {
        CHECK(decode(e, ids, 20, k, 40, spec, &acc) == 0, "decode with %d drafts: %s", k, q4e_error(e));
        int same = 0;
        while (same < 40 && spec[same] == plain[same]) ++same;
        printf("greedy decode of 40 tokens with %d drafts a step: %d accepted, the first %d tokens as without\n", k, acc, same);
        CHECK(same == 40, "%d drafts a step change token %d", k, same);
    }
    q4e_close(e);
    qwen4_ref_close(r);
    free(lg); free(buf);
    st_close_file(&st);
    printf("test_qwen4_mtp: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
