// tests/test_qwen4_tools.c - Qwen3.8-Flash-Next tool calls (harness/tool_calls.c tc_parse_qwen): the template's
// <tool_call><function=..><parameter=..> format to OpenAI calls, typed by the tools' schemas as vLLM's qwen3_coder parser.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tool_calls.h"

static const char* TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"location\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"},\"hours\":{\"type\":\"array\"},"
    "\"metric\":{\"type\":\"boolean\"},\"code\":{\"type\":\"string\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"search\",\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"query\":{\"type\":\"string\"}}}}}]";

static int fails = 0;

static void expect(const char* name, const char* text, const char* want_visible, const char* want_calls) {
    Json* tools = json_parse(TOOLS, strlen(TOOLS), NULL, 0);
    char* vis = NULL;
    Json* calls = NULL;
    char err[256];
    const int rc = tc_parse_qwen(text, tools, &vis, &calls, err, sizeof err);
    if (!want_calls) {
        if (rc == 0) { printf("FAIL %s: parsed, want an error\n", name); ++fails; }
    } else if (rc) {
        printf("FAIL %s: %s\n", name, err);
        ++fails;
    } else {
        char* got = json_dumps(calls, 1);
        if (strcmp(got, want_calls) || strcmp(vis, want_visible)) {
            printf("FAIL %s\n  calls %s\n  want  %s\n  visible \"%s\" want \"%s\"\n", name, got, want_calls, vis, want_visible);
            ++fails;
        }
        free(got);
    }
    free(vis);
    json_free(calls);
    json_free(tools);
}

int main(void) {
    expect("two calls after text (the template's rendering)",
           "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis, FR\n</parameter>\n"
           "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=search>\n"
           "<parameter=query>\ncaf\xc3\xa9 \"quoted\"\n</parameter>\n</function>\n</tool_call>",
           "Let me check.", "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris, FR\",\"days\":3}},"
                            "{\"name\":\"search\",\"arguments\":{\"query\":\"caf\xc3\xa9 \\\"quoted\\\"\"}}]");
    expect("types by schema",
           "<tool_call>\n<function=get_weather>\n<parameter=code>\n007\n</parameter>\n<parameter=hours>\n[9, 12.5]\n"
           "</parameter>\n<parameter=metric>\ntrue\n</parameter>\n<parameter=note>\n5\n</parameter>\n</function>\n</tool_call>",
           "", "[{\"name\":\"get_weather\",\"arguments\":{\"code\":\"007\",\"hours\":[9,12.5],\"metric\":true,\"note\":\"5\"}}]");
    expect("a multi-line value keeps its inner newlines",
           "<tool_call>\n<function=search>\n<parameter=query>\n\nline 1\nline 2\n\n</parameter>\n</function>\n</tool_call>",
           "", "[{\"name\":\"search\",\"arguments\":{\"query\":\"\\nline 1\\nline 2\\n\"}}]");
    expect("no parameters", "<tool_call>\n<function=search>\n</function>\n</tool_call>", "",
           "[{\"name\":\"search\",\"arguments\":{}}]");
    expect("no calls", "Just text.\n", "Just text.\n", "[]");
    expect("an unclosed call", "<tool_call>\n<function=search>\n<parameter=query>\nParis", NULL, NULL);
    expect("an unclosed parameter", "<tool_call>\n<function=search>\n<parameter=query>\nParis\n</function>\n</tool_call>",
           NULL, NULL);
    expect("no function", "<tool_call>\n</tool_call>", NULL, NULL);
    expect("text after the function", "<tool_call>\n<function=search>\n</function>\nmore\n</tool_call>", NULL, NULL);
    printf("test_qwen4_tools: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
