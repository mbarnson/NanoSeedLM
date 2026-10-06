// tests/test_chat_template.c - the C chat template against transformers' rendering (tests/data/template_golden.json,
// from tools/template_golden.py), plus the OpenAI normalisation and Python number formatting.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_template.h"

static char* slurp(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t) ftell(f);
    fseek(f, 0, SEEK_SET);
    char* p = (char*) malloc(*n + 1);
    *n = fread(p, 1, *n, f);
    p[*n] = 0;
    fclose(f);
    return p;
}
static Json* load(const char* path) {
    size_t n;
    char* s = slurp(path, &n);
    char err[256];
    Json* j = s ? json_parse(s, n, err, sizeof err) : NULL;
    if (!j) fprintf(stderr, "cannot load %s: %s\n", path, s ? err : "missing");
    free(s);
    return j;
}
static void show_diff(const char* want, const char* got) {
    size_t i = 0;
    while (want[i] && want[i] == got[i]) ++i;
    const size_t from = i > 60 ? i - 60 : 0;
    fprintf(stderr, "  first difference at byte %zu\n  want: ...%.120s\n  got:  ...%.120s\n", i, want + from, got + from);
}
static int check_num(const char* text, const char* want) {
    Json* j = json_parse(text, strlen(text), NULL, 0);
    Buf b = {0};
    py_num(&b, j);
    const int ok = !strcmp(b.p, want);
    if (!ok) fprintf(stderr, "py_num(%s) = %s, want %s\n", text, b.p, want);
    free(b.p);
    json_free(j);
    return ok;
}

int main(void) {
    int fails = 0;
    Json* cases = load("tests/data/template_cases.json");
    Json* gold = load("tests/data/template_golden.json");
    if (!cases || !gold || cases->n != gold->n) return 1;
    for (int i = 0; i < cases->n; ++i) {
        const Json* c = cases->v[i];
        const Json* kw = json_get(c, "kwargs");
        CtOpts o = {json_gets(kw, "tool_presentation_format"), json_gets(kw, "tool_call_format"),
                    json_gets(kw, "reasoning_effort"), 1};
        char err[512];
        char* got = ct_render(json_get(c, "messages"), json_get(c, "tools"), &o, err, sizeof err);
        const char* want = json_gets(gold->v[i], "text");
        const char* want_err = json_gets(gold->v[i], "error");
        const char* name = json_gets(c, "name");
        if (want && (!got || strcmp(got, want))) {
            fprintf(stderr, "FAIL %s\n", name);
            if (got) show_diff(want, got); else fprintf(stderr, "  error: %s\n", err);
            ++fails;
        } else if (want_err && (got || strcmp(err, want_err))) {
            fprintf(stderr, "FAIL %s: want error \"%s\", got %s\n", name, want_err, got ? "text" : err);
            ++fails;
        }
        free(got);
    }
    printf("template: %d cases, %d fail\n", cases->n, fails);

    // OpenAI-style input: string arguments, no thinking field, text parts, developer role
    const char* req = "[{\"role\":\"developer\",\"content\":[{\"type\":\"text\",\"text\":\"Be brief.\"}]},"
                      "{\"role\":\"user\",\"content\":\"Hi\"},"
                      "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\","
                      "\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\": 1, \\\"b\\\": \\\"x\\\"}\"}}]},"
                      "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"ok\"},"
                      "{\"role\":\"assistant\",\"content\":\"<ifm|think>\\nmull</ifm|think>Done.\"},"
                      "{\"role\":\"user\",\"content\":\"Again\"}]";
    Json* m = json_parse(req, strlen(req), NULL, 0);
    char err[256];
    if (ct_normalize(m, err, sizeof err)) { fprintf(stderr, "normalize: %s\n", err); ++fails; }
    CtOpts o = {0};
    o.add_generation_prompt = 1;
    char* got = ct_render(m, NULL, &o, err, sizeof err);
    const char* want = "<|ifm|begin_of_text|><|ifm|im_start|>system\nBe brief.<|ifm|im_end|><|ifm|im_start|>user\nHi"
                       "<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think><ifm|tool_calls>\n"
                       "<ifm|tool_call>f\n<ifm|arg_key>a</ifm|arg_key>\n<ifm|arg_value>1</ifm|arg_value>\n"
                       "<ifm|arg_key>b</ifm|arg_key>\n<ifm|arg_value>x</ifm|arg_value>\n</ifm|tool_call>\n"
                       "</ifm|tool_calls><|ifm|im_end|><|ifm|im_start|>tool\nok<|ifm|im_end|><|ifm|im_start|>assistant\n"
                       "<ifm|think>\nmull</ifm|think>Done.<|ifm|im_end|><|ifm|im_start|>user\nAgain<|ifm|im_end|>"
                       "<|ifm|im_start|>assistant\n<ifm|think>\n";
    if (!got || strcmp(got, want)) {
        fprintf(stderr, "FAIL normalize\n");
        if (got) show_diff(want, got); else fprintf(stderr, "  %s\n", err);
        ++fails;
    }
    free(got);
    json_free(m);

    const char* nums[][2] = {{"0.1", "0.1"}, {"2.0", "2.0"}, {"1e-05", "1e-05"}, {"1e16", "1e+16"}, {"1e15", "1000000000000000.0"},
                             {"12.5", "12.5"}, {"-0.00012", "-0.00012"}, {"123456789.125", "123456789.125"},
                             {"3.14159265358979", "3.14159265358979"}, {"1.5e300", "1.5e+300"}, {"42", "42"}, {"-7", "-7"}};
    for (size_t k = 0; k < sizeof nums / sizeof nums[0]; ++k) fails += !check_num(nums[k][0], nums[k][1]);

    json_free(cases);
    json_free(gold);
    printf("test_chat_template: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
