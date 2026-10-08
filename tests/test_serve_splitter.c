// tests/test_serve_splitter.c - parts of the server (harness/nslm-serve.c) without a model or a socket: the thinking /
// answer splitter (every byte split of each case, and one-byte feeds, must give the same reasoning and content), JSON
// output of invalid UTF-8, and the clamping of request numbers.
#define main nslm_server_main
#include "../harness/nslm-serve.c"
#undef main

static int check_split(const char* raw, const char* want_r, const char* want_c, bool chat, size_t cut) {
    const size_t n = strlen(raw);
    Gen g;
    memset(&g, 0, sizeof g);
    g.close_tag = chat ? "</ifm|think>" : NULL;
    Splitter sp = {0};
    if (cut <= n) emit_upto(NULL, &g, &sp, raw, utf8_complete(raw, cut), false);
    else for (size_t k = 1; k < n; ++k) emit_upto(NULL, &g, &sp, raw, utf8_complete(raw, k), false);
    emit_upto(NULL, &g, &sp, raw, n, true);
    buf_put(&g.reasoning, "", 0);
    buf_put(&g.content, "", 0);
    const int ok = !strcmp(g.reasoning.p, want_r) && !strcmp(g.content.p, want_c);
    if (!ok) fprintf(stderr, "split at %zu: %s\n  reasoning: %s\n  content: %s\n", cut, raw, g.reasoning.p, g.content.p);
    free(g.reasoning.p);
    free(g.content.p);
    return !ok;
}
static int check(const char* raw, const char* reasoning, const char* content, bool chat) {
    const size_t n = strlen(raw);
    for (size_t cut = 0; cut <= n + 1; ++cut)
        if (check_split(raw, reasoning, content, chat, cut)) return 1;
    return 0;
}

int main(void) {
    int failed = 0;
    const char* answer = "{\"say\":\"Hello\",\"commands\":[],\"stop\":true}";
    const char* tags[3] = {"ifm|think", "ifm|think_fast", "ifm|think_faster"};
    char raw[512];
    for (int i = 0; i < 3; ++i) {
        snprintf(raw, sizeof raw, "plan</%s>\n%s", tags[i], answer);
        failed += check(raw, "plan", answer, true);
        snprintf(raw, sizeof raw, "<%s>plan</%s>%s", tags[i], tags[i], answer);
        failed += check(raw, "plan", answer, true);
        snprintf(raw, sizeof raw, "<%s>\nplan\n</%s>\n%s", tags[i], tags[i], answer);
        failed += check(raw, "\nplan", answer, true);
        snprintf(raw, sizeof raw, "<%s>plan", tags[i]);
        failed += check(raw, "plan", "", true);
    }
    failed += check("<ifm|think><ifm|think_fast>plan</ifm|think_faster>answer", "plan", "answer", true);
    failed += check("caf\xC3\xA9 \xE2\x98\x95</ifm|think>r\xC3\xA9ponse", "caf\xC3\xA9 \xE2\x98\x95", "r\xC3\xA9ponse", true);
    failed += check("plan</ifm|thi", "plan</ifm|thi", "", true);
    failed += check("plan<unknown>tag</ifm|think>answer", "plan<unknown>tag", "answer", true);
    // Missing closes never promote reasoning (including JSON) into executable answer content.
    failed += check(answer, answer, "", true);
    failed += check("</ifm|think><ifm|think>quoted", "", "<ifm|think>quoted", true);
    failed += check("<ifm|think>raw</ifm|think>", "", "<ifm|think>raw</ifm|think>", false);
    printf("thinking splitter: %s\n", failed ? "FAILED" : "passed (all byte splits and one-byte feeds)");

    // JSON text stays valid UTF-8 whatever bytes the model samples: invalid sequences become U+FFFD, one per byte
    int jf = 0;
    static const struct { const char *in, *out; } U[] = {
        {"ok \xC3\xA9 \xE2\x98\x95 \xF0\x9F\x98\x80", "\"ok \xC3\xA9 \xE2\x98\x95 \xF0\x9F\x98\x80\""},   // valid: unchanged
        {"a\xE2" "b", "\"a\xEF\xBF\xBD" "b\""},                   // a lead byte followed by ASCII
        {"\x80", "\"\xEF\xBF\xBD\""},                             // a lone continuation byte
        {"\xC0\xAF", "\"\xEF\xBF\xBD\xEF\xBF\xBD\""},             // an overlong '/'
        {"\xED\xA0\x80", "\"\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\""},   // a surrogate
        {"\xF4\x90\x80\x80", "\"\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\""},   // past U+10FFFF
        {"end\xF0\x9F", "\"end\xEF\xBF\xBD\xEF\xBF\xBD\""},        // truncated at the end
    };
    for (size_t i = 0; i < sizeof U / sizeof *U; ++i) {
        Json* j = jstrn(U[i].in, strlen(U[i].in));
        char* d = json_dumps(j, 1);
        if (strcmp(d, U[i].out)) { ++jf; fprintf(stderr, "UTF-8 case %zu: %s\n", i, d); }
        free(d);
        json_free(j);
    }
    // request numbers: integers exact, everything else clamped (no undefined casts)
    const char* nums = "[18446744073709551615, 9007199254740993, 1e10, -5, 1e300, -1e300, 3.9, 1]";
    Json* a = json_parse(nums, strlen(nums), NULL, 0);
    jf += !a || a->n != 8;
    if (a && a->n == 8) {
        jf += jclamp(a->v[0], 0, INT32_MAX) != INT32_MAX;                  // past int64: a double, clamped
        jf += a->v[1]->t != J_INT || (uint64_t) a->v[1]->i != 9007199254740993ull;   // 2^53 + 1 stays exact
        jf += jclamp(a->v[2], 0, INT32_MAX) != INT32_MAX;                  // max_tokens 1e10
        jf += jclamp(a->v[3], 0, INT32_MAX) != 0;
        jf += jclamp(a->v[4], INT64_MIN, INT64_MAX) != INT64_MAX;
        jf += jclamp(a->v[5], INT64_MIN, INT64_MAX) != INT64_MIN;
        jf += jclamp(a->v[6], 0, 100) != 3;
        jf += jclamp(a->v[7], 0, 100) != 1;
    }
    json_free(a);
    printf("JSON output and request numbers: %s\n", jf ? "FAILED" : "passed");
    failed += jf;

    // sampling: min_p keeps the tokens of at least min_p x the top probability (with temperature, top_p and top_k), and
    // a request's draws depend only on its own seed
    int sf = 0;
    const float lg[8] = {5, 4.5f, 3, 2, 1, 0, -1, -2};   // p / p_max = 1, .61, .14, .05, ...
    int hits[8] = {0};
    for (uint64_t k = 0; k < 4000; ++k) {
        uint64_t r = k;
        ++hits[sample_row(lg, 8, 1.0, 1.0, 0, 0.1, &r)];
    }
    sf += !hits[0] || !hits[1] || !hits[2];
    for (int i = 3; i < 8; ++i) sf += hits[i] != 0;
    memset(hits, 0, sizeof hits);
    for (uint64_t k = 0; k < 4000; ++k) {   // min_p 0: the whole distribution (here every token has p > 1e-8)
        uint64_t r = k;
        ++hits[sample_row(lg, 8, 1.0, 1.0, 0, 0, &r)];
    }
    sf += !hits[3] || !hits[4];
    {
        uint64_t a = 77, b = 77, x = 5;
        int32_t ta[64], tb[64];
        for (int i = 0; i < 64; ++i) ta[i] = sample_row(lg, 8, 1.3, 0.95, 6, 0.02, &a);
        for (int i = 0; i < 64; ++i) {   // interleaved with another request's draws
            sample_row(lg, 8, 0.7, 1.0, 0, 0, &x);
            tb[i] = sample_row(lg, 8, 1.3, 0.95, 6, 0.02, &b);
        }
        sf += memcmp(ta, tb, sizeof ta) != 0;
    }
    printf("sampling (min_p, per-request seeds): %s\n", sf ? "FAILED" : "passed");
    failed += sf;
    return failed ? 1 : 0;
}
