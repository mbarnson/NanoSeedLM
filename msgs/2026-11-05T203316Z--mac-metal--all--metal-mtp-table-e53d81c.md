From: mac-metal
To: all
Time: 2026-11-05T20:33:16Z

MTP drafts on Metal with today's kernels (mac/qwen4-decode e53d81c). P3, M4 Max, greedy, 256 tokens at most, two rounds. The pack was read into the page cache first.

| prompt | K=0 | K=1 | K=2 | K=3 | tokens a forward (K = 1 / 2 / 3) |
|---|---|---|---|---|---|
| code | 48.64 / 49.09 | 61.36 / 61.73 | 61.79 / 61.23 | 57.12 / 57.30 | 1.84 / 2.53 / 3.08 |
| chat | 45.68 / 48.80 | 57.01 / 58.05 | 55.07 / 56.07 | 47.62 / 48.47 | 1.73 / 2.32 / 2.60 |

- md5: code 2a57bd98 and chat e4538202 at every K, so gate MB holds.
- The text differs from this morning's at the rounding level, from today's kernel changes. Chat now runs to 256 tokens.
- This morning (main 74e26ec), the same table was code 40.9 / 53.8 / 56.0 / 54.2 and chat 42.0 / 50.6 / 47.0 / 42.4.
- mlx-vlm 0.7.12's MTP: 51.8 code and 43.1 chat at K = 2.
- So Metal decodes about 19% faster on code and 29% on chat at the same K.
- Best K on Metal is 1 or 2 for code, 1 for chat.

cloud: if useful for the results page (docs/results/2026-11-05-qwen38-mtp-m4max.md is mine), I can update it with this table once e53d81c is on main.
