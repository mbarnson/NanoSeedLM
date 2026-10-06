// harness/tool_calls.c - see tool_calls.h.  Follows oMLX's K2 parser (omlx/patches/k2_horizon/tool_parser.py).
#include "tool_calls.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CALL_OPEN "<ifm|tool_call>"
#define CALL_CLOSE "</ifm|tool_call>"

static const char* skip_ws(const char* s) { while (*s && isspace((unsigned char) *s)) ++s; return s; }
static int starts(const char* s, const char* p) { return !strncmp(s, p, strlen(p)); }
static char* dup_trim(const char* a, const char* z) {
    while (a < z && isspace((unsigned char) *a)) ++a;
    while (z > a && isspace((unsigned char) z[-1])) --z;
    char* s = (char*) malloc((size_t) (z - a) + 1);
    memcpy(s, a, (size_t) (z - a));
    s[z - a] = 0;
    return s;
}
static int fail(char* err, int errlen, const char* msg) { snprintf(err, (size_t) errlen, "%s", msg); return -1; }

// Names of the tool's parameters whose schema type is "string".
static int is_string_arg(const Json* tools, const char* tool, const char* arg) {
    for (int t = 0; tools && tools->t == J_ARR && t < tools->n; ++t) {
        const Json* fn = json_get(tools->v[t], "function");
        if (!fn || !json_gets(fn, "name") || strcmp(json_gets(fn, "name"), tool)) continue;
        const Json* spec = json_get(json_get(json_get(fn, "parameters"), "properties"), arg);
        const char* type = json_gets(spec, "type");
        return type && !strcmp(type, "string");
    }
    return 0;
}
// json.loads, then the Python literals True / False / None, else the raw string.
static Json* deserialize(const char* s) {
    const char* a = skip_ws(s);
    size_t n = strlen(a);
    while (n && isspace((unsigned char) a[n - 1])) --n;
    Json* j = n ? json_parse(a, n, NULL, 0) : NULL;
    if (j) return j;
    if (n == 4 && !strncmp(a, "True", 4)) return json_new(J_TRUE);
    if (n == 5 && !strncmp(a, "False", 5)) return json_new(J_FALSE);
    if (n == 4 && !strncmp(a, "None", 4)) return json_new(J_NULL);
    return json_str(s);
}
// Python str(value) for a declared string argument that came as another type.
static Json* as_string(const Json* v) {
    if (v->t == J_STR) return json_copy(v);
    if (v->t == J_TRUE) return json_str("True");
    if (v->t == J_FALSE) return json_str("False");
    if (v->t == J_NULL) return json_str("None");
    char* s = json_dumps(v, 0);
    Json* j = json_str(s);
    free(s);
    return j;
}
static Json* make_call(const char* name, Json* args) {
    Json* c = json_new(J_OBJ);
    json_set(c, "name", json_str(name));
    json_set(c, "arguments", args);
    return c;
}

// One XML call body: "name\n<ifm|arg_key>k</ifm|arg_key>[<ifm|arg_type>t</ifm|arg_type>]<ifm|arg_value>v</ifm|arg_value>..."
static Json* xml_call(const char* body, const Json* tools, char* err, int errlen) {
    static const char *K0 = "<ifm|arg_key>", *K1 = "</ifm|arg_key>", *T0 = "<ifm|arg_type>", *T1 = "</ifm|arg_type>",
                      *V0 = "<ifm|arg_value>", *V1 = "</ifm|arg_value>";
    Json* args = json_new(J_OBJ);
    const char* first = NULL;
    const char* end_prev = NULL;
    int leftover = 0;
    char* name = NULL;
    for (const char* p = strstr(body, K0); p; p = strstr(p + 1, K0)) {
        const char* k1 = strstr(p + strlen(K0), K1);
        if (!k1) continue;
        const char* q = skip_ws(k1 + strlen(K1));
        const char *t0 = NULL, *t1 = NULL;
        if (starts(q, T0) && (t1 = strstr(q + strlen(T0), T1))) { t0 = q + strlen(T0); q = skip_ws(t1 + strlen(T1)); }
        if (!starts(q, V0)) continue;
        const char* v1 = strstr(q + strlen(V0), V1);
        if (!v1) continue;
        if (!first) {
            first = p;
            name = dup_trim(body, p);
        } else for (const char* g = end_prev; g < p; ++g) leftover |= !isspace((unsigned char) *g);
        char* key = dup_trim(p + strlen(K0), k1);
        Buf val = {0};
        buf_put(&val, q + strlen(V0), (size_t) (v1 - q - strlen(V0)));
        char* v = buf_take(&val);
        int str_arg = is_string_arg(tools, name, key);
        if (t0) {
            char* type = dup_trim(t0, t1);
            str_arg = !strcmp(type, "string");
            free(type);
        }
        json_set(args, key, str_arg ? json_str(v) : deserialize(v));
        free(key);
        free(v);
        end_prev = v1 + strlen(V1);
        p = end_prev - 1;
    }
    if (!first) name = dup_trim(body, body + strlen(body));
    else for (const char* g = end_prev; *g; ++g) leftover |= !isspace((unsigned char) *g);
    if (!name[0] || strstr(name, "<ifm|")) { free(name); json_free(args); fail(err, errlen, "tool call is missing a function name"); return NULL; }
    if (leftover) { free(name); json_free(args); fail(err, errlen, "tool call contains an incomplete argument"); return NULL; }
    Json* c = make_call(name, args);
    free(name);
    return c;
}
static Json* json_call(const Json* payload, const Json* tools, char* err, int errlen) {
    const char* name = json_gets(payload, "name");
    const Json* args = json_get(payload, "arguments");
    if (payload->t != J_OBJ || !name || !name[0]) { fail(err, errlen, "JSON tool call is missing a function name"); return NULL; }
    if (!args || args->t != J_OBJ) { fail(err, errlen, "JSON tool call arguments must be an object"); return NULL; }
    Json* out = json_new(J_OBJ);
    for (int i = 0; i < args->n; ++i) {
        const Json* v = args->v[i];
        if (is_string_arg(tools, name, args->k[i])) json_set(out, args->k[i], as_string(v));
        else if (v->t == J_STR) json_set(out, args->k[i], deserialize(v->s));
        else json_set(out, args->k[i], json_copy(v));
    }
    return make_call(name, out);
}

// The calls of one group body starting at s; returns the position after the last call, or NULL.
static const char* group(const char* s, const Json* tools, Json* calls, char* err, int errlen) {
    int n = 0;
    for (;;) {
        s = skip_ws(s);
        if (!starts(s, CALL_OPEN)) break;
        const char* b = skip_ws(s + strlen(CALL_OPEN));
        const char* end;
        Json* call;
        if (*b == '{') {
            size_t used = 0;
            Json* payload = json_parse_prefix(b, strlen(b), &used);
            if (!payload) { fail(err, errlen, "tool call contains invalid JSON"); return NULL; }
            call = json_call(payload, tools, err, errlen);
            json_free(payload);
            end = skip_ws(b + used);
        } else {
            end = strstr(b, CALL_CLOSE);
            if (!end) { fail(err, errlen, "tool group contains an incomplete call"); return NULL; }
            char* body = dup_trim(b, end);
            call = xml_call(body, tools, err, errlen);
            free(body);
        }
        if (!call) return NULL;
        if (!starts(end, CALL_CLOSE)) { json_free(call); fail(err, errlen, "tool group contains an incomplete call"); return NULL; }
        json_push(calls, call);
        ++n;
        s = end + strlen(CALL_CLOSE);
    }
    if (!n) { fail(err, errlen, "tool group contains no complete <ifm|tool_call>"); return NULL; }
    return s;
}

int tc_parse(const char* text, const Json* tools, char** visible, Json** calls, char* err, int errlen) {
    Buf vis = {0};
    Json* out = json_new(J_ARR);
    const char* pos = text;
    for (;;) {
        const char* st = strstr(pos, TC_OPEN);
        if (!st) { buf_puts(&vis, pos); break; }
        buf_put(&vis, pos, (size_t) (st - pos));
        const char* e = group(st + strlen(TC_OPEN), tools, out, err, errlen);
        if (!e || !starts(e, TC_CLOSE)) {
            if (e) fail(err, errlen, "incomplete or malformed tool-call envelope");
            free(vis.p);
            json_free(out);
            return -1;
        }
        pos = e + strlen(TC_CLOSE);
    }
    *visible = buf_take(&vis);
    *calls = out;
    return 0;
}
