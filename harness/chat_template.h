// harness/chat_template.h - K2-Horizon chat_template.jinja in C99: messages, tools and tool calls to the prompt text.
// Tool presentation: markdown (default) or json; tool-call format: xml (default), json or xml_typed.
#pragma once
#include "json.h"

typedef struct {
    const char* presentation;   // NULL = "markdown"
    const char* call_format;    // NULL = "xml"
    const char* effort;         // NULL = "high"
    int add_generation_prompt;
} CtOpts;

// The prompt as the template renders it (with BOS), or NULL with the template's error message in err.
char* ct_render(const Json* messages, const Json* tools, const CtOpts* o, char* err, int errlen);

// OpenAI request messages -> the template's input, in place: developer -> system; text-part lists -> text (not for
// tool messages); tool_calls[].function.arguments JSON strings -> objects; an assistant message without a thinking
// field gets one (split from <ifm|think...> tags in its content, else empty).  0, or -1 with err.
int ct_normalize(Json* messages, char* err, int errlen);
