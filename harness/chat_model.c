// harness/chat_model.c - see chat_model.h.
#include "chat_model.h"

#include <stdio.h>
#include <string.h>

#include "qwen4_cfg.h"
#include "tool_calls.h"

static const char* const K2_TAGS[] = {"<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>",
                                      "</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"};
static const char* const QWEN_TAGS[] = {"<think>", "</think>"};

void cm_init(ChatModel* m, int qwen) {
    m->qwen = qwen;
    m->tags = qwen ? QWEN_TAGS : K2_TAGS;
    m->ntags = qwen ? 1 : 3;
    m->stop[0] = m->stop[1] = -1;
    m->top_k = qwen ? 20 : 0;   // Qwen: generation_config.json
}

int cm_open(ChatModel* m, const char* dir, Tok* tok, char* err, int errlen) {
    Qwen4Cfg c;
    cm_init(m, dir && !qwen4_cfg_load(&c, dir, err, errlen));
    err[0] = 0;
    const char* names[2] = {m->qwen ? "<|endoftext|>" : "<|ifm|endoftext|>", m->qwen ? "<|im_end|>" : "<|ifm|im_end|>"};
    for (int i = 0; i < 2 && tok; ++i)
        if (tok_encode(tok, names[i], 0, &m->stop[i], 1) != 1) {
            snprintf(err, (size_t) errlen, "the tokenizer has no %s token", names[i]);
            return -1;
        }
    return 0;
}

const char* cm_effort(const ChatModel* m, const char* effort) {
    if (effort && !strcmp(effort, "minimal")) return "low";
    return m->qwen && effort && !strcmp(effort, "high") ? "xhigh" : effort;
}

const char* cm_close_tag(const ChatModel* m, const CtOpts* o) {
    if (m->qwen) return o->no_thinking ? NULL : QWEN_TAGS[1];
    const char* e = o->effort ? o->effort : "high";
    return K2_TAGS[3 + (!strcmp(e, "low") ? 2 : !strcmp(e, "medium") ? 1 : 0)];
}

char* cm_render(const ChatModel* m, const Json* messages, const Json* tools, const CtOpts* o, char* err, int errlen) {
    return (m->qwen ? ct_render_qwen : ct_render)(messages, tools, o, err, errlen);
}

int cm_tool_calls(const ChatModel* m, const char* text, const Json* tools, char** visible, Json** calls, char* err, int errlen) {
    return (m->qwen ? tc_parse_qwen : tc_parse)(text, tools, visible, calls, err, errlen);
}
