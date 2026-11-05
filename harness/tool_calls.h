// harness/tool_calls.h - K2-Horizon tool-call envelopes (<ifm|tool_calls>, XML or JSON calls) and Qwen3.8-Flash-Next's
// <tool_call> blocks to OpenAI tool calls.
#pragma once
#include "json.h"

#define TC_OPEN "<ifm|tool_calls>"
#define TC_CLOSE "</ifm|tool_calls>"

// Parses every complete group in text.  *visible = the text outside the groups (malloc'd); *calls = a J_ARR of
// {"name": ..., "arguments": {...}} with arguments typed by the tools' schemas.  0, or -1 with err (malformed).
int tc_parse(const char* text, const Json* tools, char** visible, Json** calls, char* err, int errlen);
// The same for Qwen3.8-Flash-Next's <tool_call> blocks; *visible loses the whitespace beside them.
int tc_parse_qwen(const char* text, const Json* tools, char** visible, Json** calls, char* err, int errlen);
