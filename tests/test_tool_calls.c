// tests/test_tool_calls.c - K2-Horizon tool-call parsing (the cases of oMLX's K2 parser).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tool_calls.h"

static const char* TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"location\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"},\"hours\":{\"type\":\"array\"},"
    "\"metric\":{\"type\":\"boolean\"},\"code\":{\"type\":\"string\"}}}}}]";

static int fails = 0;

static void expect(const char* name, const char* text, const char* want_visible, const char* want_calls) {
    Json* tools = json_parse(TOOLS, strlen(TOOLS), NULL, 0);
    char* vis = NULL;
    Json* calls = NULL;
    char err[256];
    const int rc = tc_parse(text, tools, &vis, &calls, err, sizeof err);
    if (!want_calls) {
        if (rc == 0) { fprintf(stderr, "FAIL %s: parsed, want an error\n", name); ++fails; }
    } else if (rc) {
        fprintf(stderr, "FAIL %s: %s\n", name, err);
        ++fails;
    } else {
        char* got = json_dumps(calls, 1);
        if (strcmp(got, want_calls) || strcmp(vis, want_visible)) {
            fprintf(stderr, "FAIL %s\n  calls %s\n  want  %s\n  visible \"%s\" want \"%s\"\n", name, got, want_calls, vis, want_visible);
            ++fails;
        }
        free(got);
    }
    free(vis);
    json_free(calls);
    json_free(tools);
}

int main(void) {
    expect("xml",
           "<ifm|tool_calls>\n<ifm|tool_call>get_weather\n<ifm|arg_key>location</ifm|arg_key>\n<ifm|arg_value>Paris, FR"
           "</ifm|arg_value>\n<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_value>2</ifm|arg_value>\n<ifm|arg_key>hours"
           "</ifm|arg_key>\n<ifm|arg_value>[9, 12.5]</ifm|arg_value>\n<ifm|arg_key>metric</ifm|arg_key>\n<ifm|arg_value>"
           "True</ifm|arg_value>\n<ifm|arg_key>code</ifm|arg_key>\n<ifm|arg_value>007</ifm|arg_value>\n</ifm|tool_call>\n"
           "</ifm|tool_calls>",
           "", "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris, FR\",\"days\":2,\"hours\":[9,12.5],"
               "\"metric\":true,\"code\":\"007\"}}]");
    expect("xml_typed",
           "<ifm|tool_calls><ifm|tool_call>get_weather\n<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_type>string"
           "</ifm|arg_type>\n<ifm|arg_value>3</ifm|arg_value>\n<ifm|arg_key>location</ifm|arg_key><ifm|arg_type>integer"
           "</ifm|arg_type><ifm|arg_value>42</ifm|arg_value></ifm|tool_call></ifm|tool_calls>",
           "", "[{\"name\":\"get_weather\",\"arguments\":{\"days\":\"3\",\"location\":42}}]");
    expect("json_and_prose",
           "Checking.\n<ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Oslo\", "
           "\"days\": \"4\"}}</ifm|tool_call>\n<ifm|tool_call>other\n</ifm|tool_call>\n</ifm|tool_calls>\nDone.",
           "Checking.\n\nDone.",
           "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Oslo\",\"days\":4}},{\"name\":\"other\",\"arguments\":{}}]");
    expect("unknown_tool_text_value", "<ifm|tool_calls><ifm|tool_call>f\n<ifm|arg_key>q</ifm|arg_key><ifm|arg_value>hello world"
           "</ifm|arg_value></ifm|tool_call></ifm|tool_calls>", "", "[{\"name\":\"f\",\"arguments\":{\"q\":\"hello world\"}}]");
    expect("no_calls", "Just text.", "Just text.", "[]");
    expect("unclosed_group", "<ifm|tool_calls><ifm|tool_call>f\n</ifm|tool_call>", NULL, NULL);
    expect("unclosed_call", "<ifm|tool_calls><ifm|tool_call>f\n<ifm|arg_key>a</ifm|arg_key>", NULL, NULL);
    expect("no_name", "<ifm|tool_calls><ifm|tool_call>\n<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>1</ifm|arg_value>"
           "</ifm|tool_call></ifm|tool_calls>", NULL, NULL);
    expect("garbage_between_args", "<ifm|tool_calls><ifm|tool_call>f\n<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>1"
           "</ifm|arg_value>junk<ifm|arg_key>b</ifm|arg_key><ifm|arg_value>2</ifm|arg_value></ifm|tool_call></ifm|tool_calls>",
           NULL, NULL);
    expect("empty_group", "<ifm|tool_calls>\n</ifm|tool_calls>", NULL, NULL);
    printf("test_tool_calls: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
