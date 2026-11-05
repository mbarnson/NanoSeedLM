// tests/test_qwen4_template.c - harness/chat_qwen.c against transformers' rendering of Qwen3.8-Flash-Next's template
// (tests/data/qwen4_template_golden.json, from tools/qwen4_template_golden.py), byte for byte, errors included.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_template.h"

int main(void) {
    FILE* f = fopen("tests/data/qwen4_template_golden.json", "rb");
    if (!f) { printf("FAIL: tests/data/qwen4_template_golden.json missing\n"); return 1; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* s = (char*) malloc((size_t) n + 1);
    const size_t got = fread(s, 1, (size_t) n, f);
    fclose(f);
    char err[512];
    Json* cases = json_parse(s, got, err, sizeof err);
    free(s);
    if (!cases) { printf("FAIL: golden: %s\n", err); return 1; }
    int fails = 0;
    for (int i = 0; i < cases->n; ++i) {
        const Json* c = cases->v[i];
        const Json* kw = json_get(c, "kwargs");
        const Json* th = json_get(kw, "enable_thinking"), *pt = json_get(kw, "preserve_thinking");
        CtOpts o;
        memset(&o, 0, sizeof o);
        o.effort = json_gets(kw, "reasoning_effort");
        o.add_generation_prompt = json_truthy(json_get(kw, "add_generation_prompt"));
        o.no_thinking = th && th->t == J_FALSE;
        o.no_preserve_thinking = pt && pt->t == J_FALSE;
        err[0] = 0;
        char* text = ct_render_qwen(json_get(c, "messages"), json_get(c, "tools"), &o, err, sizeof err);
        const char* want = json_gets(c, "text"), *werr = json_gets(c, "error");
        const int ok = want ? text && !strcmp(text, want) : !text && werr && !strcmp(err, werr);
        if (!ok) {
            ++fails;
            printf("FAIL %s:\n  want %s\n  got  %s\n", json_gets(c, "name"), want ? want : werr, text ? text : err);
        }
        free(text);
    }
    printf("test_qwen4_template: %d cases, %s\n", cases->n, fails ? "FAIL" : "PASS");
    json_free(cases);
    return fails != 0;
}
