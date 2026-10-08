#!/usr/bin/env python3
"""tools/tokenizer_golden.py - token ids of the Hugging Face tokenizer (the `tokenizers` library) for
tests/test_tokenizer.c: harness/tokenizer.c must give the same ids for every case, on every platform.

  MOVA_MODEL=MODEL_DIR python tools/tokenizer_golden.py      # writes tests/data/tokenizer_golden.json

Any platform: needs `tokenizers` only (no MLX).  MODEL_DIR is any folder with the model's tokenizer.json.
"""
import json
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    "Hello, world!",
    "Why do tide pools matter? Answer in two sentences.",
    "I'm sure they'll say you've done it, but we'd rather it's not THEIR'S.",
    "Numbers: 0 7 42 123 1234 12345678 3.14159 -2e10 1,000,000",
    "    four spaces, then a tab\there, then\n\nnewlines\r\nand CRLF\r\n\r\n   \n",
    "trailing spaces   ",
    "Café crème brûlée, naïve coöperate, Ångström",
    "decomposed: Café crème (NFC composes these)",
    "中文文本，日本語のテキスト、한국어 텍스트",
    "Emoji: \U0001F600 \U0001F44D\U0001F3FD \U0001F1FA\U0001F1F8 ☕ ❤️",
    "def f(x):\n    return {k: v for k, v in x.items() if v is not None}  # comment\n",
    "if (a && b || !c) { printf(\"%d\\n\", a[i++]); }",
    "URL: https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B?x=1&y=2#frag",
    "<|ifm|begin_of_text|><|ifm|im_start|>system\nYou are helpful.<|ifm|im_end|><|ifm|im_start|>user\nHi<|ifm|im_end|>"
    "<|ifm|im_start|>assistant\n<ifm|think>\n",
    "text<|ifm|im_end|>right after<|ifm|im_start|>",
    "Mañana, ¿qué? ¡Sí! «» “quotes” ‘single’ — em-dash …",
    "Αθήνα, Москва, עברית, العربية, हिन्दी",
    "ZWJ: ‍ ‌ in words: a‍b",
    "",
    " ",
    "\n",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes "
    "she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene "
    "Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.",
    # one word of 200000 letters: past ICU's default regex backtracking stack (which ends near 195000 here;
    # tokenizer.c lifts the cap), and long enough that a quadratic BPE would be slow
    "a" * 200000,
]


def main():
    d = Path(os.environ.get("MOVA_MODEL", sys.argv[1] if len(sys.argv) > 1 else "."))
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    out = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in CASES]
    p = ROOT / "tests" / "data" / "tokenizer_golden.json"
    lines = ",\n".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in out)   # one case per line
    p.write_text("[\n" + lines + "\n]\n", encoding="utf-8")
    print(f"{p}: {len(out)} cases, {sum(len(c['ids']) for c in out)} ids")


if __name__ == "__main__":
    main()
