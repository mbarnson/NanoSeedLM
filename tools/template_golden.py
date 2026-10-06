#!/usr/bin/env python3
"""Render tests/data/template_cases.json with the model's own chat template (transformers) into
tests/data/template_golden.json.  nslm-serve's C renderer must match every golden byte for byte.

  template_golden.py MODEL_DIR
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather.\nUse metric units by default.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and country, e.g. Paris, FR"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
            },
            "required": ["location"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}
COMPLEX = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a calendar event with \"quotes\", a back\\slash and unicode: café ☕.",
        "parameters": {
            "type": "object",
            "$defs": {
                "Person": {
                    "type": "object",
                    "description": "A person",
                    "properties": {"name": {"type": "string"}, "email": {"type": "string", "format": "email"}},
                    "required": ["name"],
                },
                "Color": {"type": "string", "enum": ["red", "it's green", "blue\\ish"]},
            },
            "properties": {
                "title": {"type": "string", "title": "Title", "examples": ["Standup", "1:1"]},
                "attendees": {"type": "array", "items": {"$ref": "#/$defs/Person"}, "description": "Who comes.\nAt least one."},
                "organizer": {"$ref": "#/$defs/Person", "description": "Overrides the def description"},
                "color": {"$ref": "#/$defs/Color", "default": "red"},
                "duration": {"anyOf": [{"type": "integer"}, {"type": "string", "pattern": "^[0-9]+m$"}]},
                "when": {"oneOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
                "priority": {"type": ["integer", "null"], "enum": [1, 2, 3, None], "default": None},
                "ratio": {"type": "number", "default": 0.1, "minimum": 1e-05, "maximum": 1e16},
                "weight": {"type": "number", "default": 2.0},
                "flags": {"type": "object", "additionalProperties": {"type": "boolean"}, "default": {"a": True, "b": False}},
                "meta": {"type": "object", "additionalProperties": False, "patternProperties": {"^x-": {"type": "string"}}},
                "tags": {"type": "array", "items": {"type": "string"}, "default": [], "maxItems": 10},
                "note": {"type": "string", "default": ""},
                "big": {"type": "integer", "default": 12345678901234},
            },
            "required": ["title", "attendees"],
        },
        "returns": {"type": "object", "properties": {"id": {"type": "string", "description": "Event id"}}},
    },
}
NO_PARAMS = {"type": "function", "function": {"name": "now", "description": "", "parameters": {"type": "object", "properties": {}}}}
RESPONSE = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
        "response": {"type": "string"},
    },
}
ROOT_ONEOF = {
    "type": "function",
    "function": {
        "name": "pick",
        "parameters": {"oneOf": [{"type": "object", "properties": {"a": {"type": "string"}}}, {"type": "object", "properties": {"b": {"type": "integer"}}}]},
    },
}
FALLBACK = {   # $ref outside the document: the template renders the whole toolset as JSON
    "type": "function",
    "function": {
        "name": "fetch",
        "description": "Fetch a URL.",
        "parameters": {"type": "object", "properties": {"url": {"$ref": "https://example.com/schemas/url"}}},
    },
}
BARE = {"name": "bare_fn", "description": "A tool without the function wrapper.", "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}}

U = {"role": "user", "content": "What's the weather in Paris?"}
CALL = {
    "role": "assistant",
    "reasoning_content": "The user wants weather. Call get_weather.",
    "content": "",
    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris, FR", "days": 2, "metric": True, "hours": [9, 12.5], "opts": {"lang": "fr", "n": None}}}}],
}
TWO_CALLS = {
    "role": "assistant",
    "reasoning_content": "",
    "content": "Let me check both.",
    "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Oslo"}}},
        {"type": "function", "function": {"name": "search", "arguments": {"query": "Oslo \"events\" café"}}},
    ],
}
RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 18, \"sky\": \"clear\"}"}

CASES = [
    {"name": "plain", "messages": [U]},
    {"name": "system_low", "messages": [{"role": "system", "content": "Be brief."}, U], "kwargs": {"reasoning_effort": "low"}},
    {"name": "medium", "messages": [U], "kwargs": {"reasoning_effort": "medium"}},
    {"name": "multi_turn", "messages": [U, {"role": "assistant", "reasoning_content": "Think.", "content": "Sunny."}, {"role": "user", "content": "And tomorrow?"}]},
    {"name": "think_fast_field", "messages": [U, {"role": "assistant", "think_fast": "Quick.", "content": "Sunny."}, {"role": "user", "content": "Thanks"}]},
    {"name": "empty_reasoning", "messages": [U, {"role": "assistant", "reasoning": "", "content": "Hi."}, {"role": "user", "content": "Bye"}]},
    {"name": "late_system", "messages": [U, {"role": "assistant", "reasoning_content": "", "content": "Ok."}, {"role": "system", "content": "Now answer in French."}, {"role": "user", "content": "Encore?"}]},
    {"name": "tools_markdown", "messages": [U], "tools": [WEATHER, SEARCH]},
    {"name": "tools_markdown_system", "messages": [{"role": "system", "content": "You are a weather bot."}, U], "tools": [WEATHER]},
    {"name": "tools_json", "messages": [U], "tools": [WEATHER, SEARCH], "kwargs": {"tool_presentation_format": "json"}},
    {"name": "tools_complex", "messages": [U], "tools": [COMPLEX, NO_PARAMS, RESPONSE, ROOT_ONEOF, BARE]},
    {"name": "tools_fallback", "messages": [U], "tools": [WEATHER, FALLBACK]},
    {"name": "tools_in_system", "messages": [{"role": "system", "content": "Sys.", "tools": [SEARCH]}, U]},
    {"name": "call_xml", "messages": [U, CALL, RESULT], "tools": [WEATHER]},
    {"name": "call_json", "messages": [U, CALL, RESULT], "tools": [WEATHER], "kwargs": {"tool_call_format": "json"}},
    {"name": "call_xml_typed", "messages": [U, CALL, RESULT], "tools": [WEATHER, COMPLEX], "kwargs": {"tool_call_format": "xml_typed"}},
    {"name": "two_calls", "messages": [U, TWO_CALLS, {"role": "tool", "content": "rain"}, {"role": "tool", "content": "concert"}], "tools": [WEATHER, SEARCH]},
    {"name": "tool_content_list", "messages": [U, CALL, {"role": "tool", "content": ["a", {"type": "text", "text": "b"}, {"k": 1}]}], "tools": [WEATHER]},
    {"name": "tool_content_dict", "messages": [U, CALL, {"role": "tool", "content": {"temp": 18.0, "ok": True}}], "tools": [WEATHER]},
    {"name": "typed_anyof", "messages": [U, {"role": "assistant", "reasoning_content": "", "content": "", "tool_calls": [{"type": "function", "function": {"name": "create_event", "arguments": {"title": "x", "duration": 30, "attendees": [{"name": "A"}], "color": "red", "ratio": 0.5}}}]}], "tools": [COMPLEX], "kwargs": {"tool_call_format": "xml_typed"}},
    {"name": "err_required_not_list", "messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "required": "a"}}}]},
    {"name": "err_required_missing", "messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]}}}]},
    {"name": "err_no_parameters", "messages": [U], "tools": [{"type": "function", "function": {"name": "f"}}]},
]


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    out = []
    for case in CASES:
        try:
            text = tok.apply_chat_template(case["messages"], tools=case.get("tools"), tokenize=False,
                                           add_generation_prompt=True, **case.get("kwargs", {}))
            out.append({"name": case["name"], "text": text})
        except Exception as error:   # the template's raise_exception
            out.append({"name": case["name"], "error": str(error)})
    data = ROOT / "tests" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "template_cases.json").write_text(json.dumps(CASES, ensure_ascii=False, indent=1) + "\n")
    (data / "template_golden.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(out)} cases, {sum('error' in o for o in out)} errors")


if __name__ == "__main__":
    main()
