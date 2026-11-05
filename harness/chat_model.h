// harness/chat_model.h - the parts of a chat that depend on the model: template, stop tokens, thinking tags, tool calls.
// K2-Horizon, or Qwen3.8-Flash-Next when config.json loads as qwen4_exp (engine/engine.c picks its engine the same way).
#pragma once
#include <stdint.h>

#include "chat_template.h"
#include "tokenizer.h"

typedef struct {
    int qwen;
    const char* const* tags;   // the thinking tags: ntags opening ones, then their closes
    int ntags;
    int32_t stop[2];           // end of text and end of turn (-1 without a tokenizer)
    int top_k;                 // the model card's sampling default (0: off)
} ChatModel;

void cm_init(ChatModel* m, int qwen);
// cm_init for the model in dir, with the stop tokens from tok (may be NULL). 0, or -1 with err.
int cm_open(ChatModel* m, const char* dir, Tok* tok, char* err, int errlen);
static inline int cm_stop(const ChatModel* m, int32_t t) { return t >= 0 && (t == m->stop[0] || t == m->stop[1]); }
// A request's effort as the template takes it (NULL: the template's default): minimal is low, and Qwen's high is xhigh.
const char* cm_effort(const ChatModel* m, const char* effort);
// The close of the thinking span the prompt opens, or NULL when it opens none.
const char* cm_close_tag(const ChatModel* m, const CtOpts* o);
char* cm_render(const ChatModel* m, const Json* messages, const Json* tools, const CtOpts* o, char* err, int errlen);
int cm_tool_calls(const ChatModel* m, const char* text, const Json* tools, char** visible, Json** calls, char* err, int errlen);
