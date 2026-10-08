// tests/test_serve_splitter.c - the server's thinking / answer splitter (harness/nslm-serve.c), without a model or a
// socket: every byte split of each case, and one-byte feeds, must give the same reasoning and content.
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
    return failed ? 1 : 0;
}
