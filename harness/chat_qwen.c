// harness/chat_qwen.c - Qwen3.8-Flash-Next's chat_template.jinja in C99 (see chat_template.h). The tests compare it
// with transformers' rendering byte for byte (tests/data/qwen4_template_golden.json, tools/qwen4_template_golden.py).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_template.h"

static const char* const XHIGH = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                                 "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
                                 "and clarity in the final answer.";
static const char* const LOW = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
                               "the conclusion without unnecessary elaboration.";
static const char* const CALL_HOWTO =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
    "parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

typedef struct {
    Buf o;
    char* err;
    int errlen, failed;
} Q;

static void raise_(Q* q, const char* msg) {
    if (!q->failed) snprintf(q->err, (size_t) q->errlen, "%s", msg);
    q->failed = 1;
}
static int ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
// render_content(content, ..., is_system)|trim, as a new string
static char* content(Q* q, const Json* c, int is_system) {
    Buf b = {0};
    buf_puts(&b, "");
    if (c && c->t == J_STR) buf_puts(&b, c->s);
    else if (c && c->t == J_ARR)
        for (int i = 0; i < c->n; ++i) {
            const Json* it = c->v[i];
            const char* type = json_gets(it, "type");
            const int image = json_get(it, "image") || json_get(it, "image_url") || (type && !strcmp(type, "image"));
            if ((image || json_get(it, "video") || (type && !strcmp(type, "video"))) && is_system)
                raise_(q, image ? "System message cannot contain images." : "System message cannot contain videos.");
            if (image) buf_puts(&b, "<|vision_start|><|image_pad|><|vision_end|>");
            else if (json_get(it, "video") || (type && !strcmp(type, "video"))) buf_puts(&b, "<|vision_start|><|video_pad|><|vision_end|>");
            else if (json_gets(it, "text")) buf_puts(&b, json_gets(it, "text"));
            else raise_(q, "Unexpected item type in content.");
        }
    else if (c && c->t != J_NULL) raise_(q, "Unexpected content type.");
    char* s = buf_take(&b);
    size_t a = 0, n = strlen(s);
    while (a < n && ws(s[a])) ++a;
    while (n > a && ws(s[n - 1])) --n;
    memmove(s, s + a, n - a);
    s[n - a] = 0;
    return s;
}
static const char* role(const Json* m) { const char* r = json_gets(m, "role"); return r ? r : ""; }
static void put(Q* q, const char* s) { buf_puts(&q->o, s); }

// The system turn: the effort's instructions, the tools and the system message
static void system_turn(Q* q, const Json* msgs, const Json* tools, const char* ri) {
    const Json* m0 = msgs->v[0];
    char* sys = !strcmp(role(m0), "system") ? content(q, json_get(m0, "content"), 1) : NULL;
    if (tools && tools->t == J_ARR && tools->n) {
        put(q, "<|im_start|>system\n");
        if (*ri) { put(q, ri); put(q, "\n\n"); }
        put(q, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (int i = 0; i < tools->n; ++i) { put(q, "\n"); json_dump(&q->o, tools->v[i], 0); }
        put(q, "\n</tools>");
        put(q, CALL_HOWTO);
        if (sys && *sys) { put(q, "\n\n"); put(q, sys); }
        put(q, "<|im_end|>\n");
    } else if ((sys && *sys) || *ri) {
        put(q, "<|im_start|>system\n");
        put(q, sys && *sys ? ri : "");
        if (sys && *sys && *ri) put(q, "\n\n");
        put(q, sys && *sys ? sys : ri);
        put(q, "<|im_end|>\n");
    }
    free(sys);
}

static void tool_calls(Q* q, const Json* calls, int text_before) {
    for (int i = 0; calls && calls->t == J_ARR && i < calls->n; ++i) {
        const Json* tc = json_get(calls->v[i], "function") ? json_get(calls->v[i], "function") : calls->v[i];
        const char* name = json_gets(tc, "name");
        put(q, i ? "\n<tool_call>\n<function=" : text_before ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
        put(q, name ? name : "");
        put(q, ">\n");
        const Json* args = json_get(tc, "arguments");
        for (int a = 0; args && args->t == J_OBJ && a < args->n; ++a) {
            put(q, "<parameter=");
            put(q, args->k[a]);
            put(q, ">\n");
            if (args->v[a]->t == J_STR) put(q, args->v[a]->s); else json_dump(&q->o, args->v[a], 0);
            put(q, "\n</parameter>\n");
        }
        put(q, "</function>\n</tool_call>");
    }
}

char* ct_render_qwen(const Json* messages, const Json* tools, const CtOpts* o, char* err, int errlen) {
    Q q = {{0}, err, errlen, 0};
    const char* effort = o && o->effort ? o->effort : "xhigh";
    if (!messages || messages->t != J_ARR || !messages->n) { snprintf(err, (size_t) errlen, "No messages provided."); return NULL; }
    const char* ri = "";
    if (!(o && o->no_thinking)) {
        if (strcmp(effort, "xhigh") && strcmp(effort, "medium") && strcmp(effort, "low")) {
            snprintf(err, (size_t) errlen, "Unexpected reasoning effort %s. Supported types are xhigh (default), medium, and low.", effort);
            return NULL;
        }
        ri = !strcmp(effort, "xhigh") ? XHIGH : !strcmp(effort, "low") ? LOW : "";
    }
    buf_puts(&q.o, "");
    system_turn(&q, messages, tools, ri);
    int last_query = -1;   // the last user message that is not a tool response
    for (int i = messages->n - 1; i >= 0 && last_query < 0; --i)
        if (!strcmp(role(messages->v[i]), "user")) {
            char* c = content(&q, json_get(messages->v[i], "content"), 0);
            const size_t n = strlen(c);
            if (!(n >= 31 && !strncmp(c, "<tool_response>", 15) && !strcmp(c + n - 16, "</tool_response>"))) last_query = i;
            free(c);
        }
    if (last_query < 0) raise_(&q, "No user query found in messages.");
    for (int i = 0; i < messages->n && !q.failed; ++i) {
        const Json* m = messages->v[i];
        const char* r = role(m);
        char* c = content(&q, json_get(m, "content"), 0);
        if (!strcmp(r, "system")) { if (i) raise_(&q, "System message must be at the beginning."); }
        else if (!strcmp(r, "user")) { put(&q, "<|im_start|>user\n"); put(&q, c); put(&q, "<|im_end|>\n"); }
        else if (!strcmp(r, "assistant")) {
            put(&q, "<|im_start|>assistant\n");
            if (!(o && o->no_preserve_thinking) || i > last_query) {
                const char* rc = json_gets(m, "reasoning_content");
                char* t = content(&q, rc ? json_get(m, "reasoning_content") : NULL, 0);
                put(&q, "<think>\n"); put(&q, t); put(&q, "\n</think>\n\n");
                free(t);
            }
            put(&q, c);
            tool_calls(&q, json_get(m, "tool_calls"), *c != 0);
            put(&q, "<|im_end|>\n");
        } else if (!strcmp(r, "tool")) {
            if (i && strcmp(role(messages->v[i - 1]), "tool")) put(&q, "<|im_start|>user");
            put(&q, "\n<tool_response>\n"); put(&q, c); put(&q, "\n</tool_response>");
            if (i == messages->n - 1 || strcmp(role(messages->v[i + 1]), "tool")) put(&q, "<|im_end|>\n");
        } else raise_(&q, "Unexpected message role.");
        free(c);
    }
    if (o && o->add_generation_prompt) put(&q, o->no_thinking ? "<|im_start|>assistant\n<think>\n\n</think>\n\n" : "<|im_start|>assistant\n<think>\n");
    char* s = buf_take(&q.o);
    if (q.failed) { free(s); return NULL; }
    return s;
}
