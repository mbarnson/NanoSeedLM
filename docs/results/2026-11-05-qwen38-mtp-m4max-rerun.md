# MTP drafts on the M4 Max, again: 61.5 tokens per second on code (2026-11-05)

**Result.** After the day's Metal decode kernels, P3 decodes code at 48.9 tokens per second without drafts and 61.5
with 1 draft (+26%). Chat decodes at 47.2 without drafts and 57.5 with 1 draft (+22%). The greedy text is the same at
every draft count, so gate MB passes. At 2 drafts, Metal decodes 19% faster on code and 29% faster on chat than mlx-vlm
0.7.12's own MTP.

**Status.** Current. It replaces the speed numbers of [MTP drafts on the M4 Max](2026-11-05-qwen38-mtp-m4max.md).

## Setup

- Machine: mac-metal's Mac. Apple M4 Max (16 CPU cores, 40 GPU cores), 128 GB unified memory, macOS 27.2. No other GPU
  jobs ran during the timed runs. The P3 pack was read into the page cache first.
- Commit: `e53d81c` (main). The decode and prompt kernels of 2026-11-05 are in [log.md](../log.md).
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3), with its
  affine Q4 MTP head.
- Two rounds. In each round, code then chat, and K = 0, 1, 2 and 3 in that order.

```bash
nslm-qwen4-gen --model P3 --prompt "Write a Python function that merges two sorted lists, with a docstring." --max-tokens 256 --drafts K
nslm-qwen4-gen --model P3 --prompt "Why do tide pools matter? Answer in three sentences." --max-tokens 256 --drafts K
```

## Results

Tokens per second (two rounds and the mean), and tokens a forward. The md5 of stdout is the same at every K on each
prompt.

| Prompt | Drafts | Round 1 | Round 2 | Mean | Tokens a forward | md5 of stdout |
|---|---|---|---|---|---|---|
| Code, 256 tokens | 0 | 48.64 | 49.09 | 48.87 | 1.00 | `2a57bd98` |
| Code | 1 | 61.36 | 61.73 | 61.55 | 1.84 | `2a57bd98` |
| Code | 2 | 61.79 | 61.23 | 61.51 | 2.53 | `2a57bd98` |
| Code | 3 | 57.12 | 57.30 | 57.21 | 3.08 | `2a57bd98` |
| Chat, 256 tokens | 0 | 45.68 | 48.80 | 47.24 | 1.00 | `e4538202` |
| Chat | 1 | 57.01 | 58.05 | 57.53 | 1.73 | `e4538202` |
| Chat | 2 | 55.07 | 56.07 | 55.57 | 2.32 | `e4538202` |
| Chat | 3 | 47.62 | 48.47 | 48.05 | 2.60 | `e4538202` |

## Notes

- The text differs from the earlier page's at the rounding level, because the day's kernels add some sums in another
  order. In this run the chat answer is 256 tokens long, against 98 before.
- 1 or 2 drafts are best on code, and 1 draft is best on chat. 3 drafts cost more than they save on chat.
- A round of 1 draft costs about 27.9 ms of GPU time, against 19.5 ms for one token without drafts (`Q4E_PROFILE=3`).
  The second verify row's seed experts take about 4.6 ms of the extra 8.5 ms. The dense matvecs at 2 rows take about
  1.4 ms more, and the draft itself takes about 1.5 to 2 ms.
- mlx-vlm 0.7.12's MTP decoded 51.8 (code) and 43.1 (chat) tokens per second at 2 drafts in mac-metal's study.
