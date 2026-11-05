#!/usr/bin/env python3
# tools/qwen4_template_golden.py TEMPLATE - Qwen3.8-Flash-Next's chat_template.jinja rendered by transformers for the
# cases below, into tests/data/qwen4_template_golden.json (harness/chat_qwen.c must match it byte for byte).
import json
import sys
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

ROOT = Path(__file__).resolve().parent.parent
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather.\nMetric by default.",
           "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City, e.g. Paris"},
                          "days": {"type": "integer", "minimum": 1}}, "required": ["location"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {"type": "object",
          "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
U = {"role": "user", "content": "Why do tide pools matter?"}
CALL = {"role": "assistant", "content": "", "tool_calls": [
    {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris, FR", "days": 3}}},
    {"type": "function", "function": {"name": "search", "arguments": {"query": "café ☕ \"quoted\""}}}]}
CASES = [
    ("user", [U], None, {}),
    ("system low", [{"role": "system", "content": "  Be brief.\n"}, U], None, {"reasoning_effort": "low"}),
    ("system medium", [{"role": "system", "content": "Be brief."}, U], None, {"reasoning_effort": "medium"}),
    ("no thinking", [{"role": "system", "content": "Be brief."}, U], None, {"enable_thinking": False}),
    ("empty system", [{"role": "system", "content": "  "}, U], None, {}),
    ("multi-turn", [U, {"role": "assistant", "content": "They hold life.", "reasoning_content": " Think. "},
                    {"role": "user", "content": "And why else?"}], None, {}),
    ("multi-turn, no preserve", [U, {"role": "assistant", "content": "They hold life.", "reasoning_content": "Think."},
                                 {"role": "user", "content": "And why else?"}], None, {"preserve_thinking": False}),
    ("text parts", [{"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Two."}]}],
     None, {}),
    ("no generation prompt", [U, {"role": "assistant", "content": "Life."}], None, {"add_generation_prompt": False}),
    ("tools", [U], [WEATHER, SEARCH], {}),
    ("tools and system", [{"role": "system", "content": "Use tools."}, U], [WEATHER], {"reasoning_effort": "medium"}),
    ("tool calls", [{"role": "user", "content": "Weather in Paris?"}, CALL,
                    {"role": "tool", "content": "{\"temp\": 18}"}, {"role": "tool", "content": "results"}], [WEATHER, SEARCH], {}),
    ("tool call after text", [{"role": "user", "content": "Weather?"},
                              dict(CALL, content="Let me check.", tool_calls=CALL["tool_calls"][:1]),
                              {"role": "tool", "content": "sunny"}, {"role": "user", "content": "Thanks!"}], [WEATHER], {}),
    ("bad effort", [U], None, {"reasoning_effort": "high"}),
    ("system not first", [U, {"role": "system", "content": "late"}], None, {}),
]


def main(template_path):
    template = Path(template_path).read_text()
    out = []
    for name, messages, tools, kw in CASES:
        kw = dict(kw)
        gen = kw.pop("add_generation_prompt", True)
        case = {"name": name, "messages": messages, "tools": tools, "kwargs": dict(kw, add_generation_prompt=gen)}
        try:
            text = render_jinja_template([messages], tools=tools, chat_template=template, add_generation_prompt=gen, **kw)
            case["text"] = text[0][0] if isinstance(text, tuple) else text[0]
        except Exception as e:   # the template's raise_exception
            case["error"] = str(e)
        out.append(case)
    dst = ROOT / "tests" / "data" / "qwen4_template_golden.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("%s: %d cases" % (dst, len(out)))


if __name__ == "__main__":
    main(sys.argv[1])
