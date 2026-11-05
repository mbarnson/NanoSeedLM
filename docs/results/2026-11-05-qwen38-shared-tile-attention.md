# Qwen3.8-Flash-Next on CUDA: prompt attention over 4 rows' shared key tiles is slower (2026-11-05)

**Result.** A prompt attention kernel that loads each key tile once for 4 rows made attention 1.7 times slower: 480 to
805 ms at 15,922 tokens, 895 to 1,600 ms at 29,120. It loads 2.7 times fewer keys, so the old kernel is not bound by its
loads. It is bound by the latency of its stages, which 4 blocks an SM hide. We reverted it (`c036709`).

**Status.** Current. It stops [the plan](../plans/2026-11-05-qwen38-shared-tile-attention.md) at gate G2.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB). Load average 2.3 to 11.9.
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`, an 8 GB expert cache. Prompts: the first 64,000 and 120,000 characters of the wikitext-2 test set, raw.
- `d2aedef` (one row a block) and `3ef4599` (4 rows a block), built on one node and run in turn, twice each
  (`scratchpad/rowsjob/abpre.sh`, job `6aec7fff`). The attention times are `Q4E_PROFILE` stages.
- KL against [the 4-bit MLX model](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp) at `825df62` on
  4,096 tokens (`kld4.sh`, job `6aec8000`).

## Results

| Prompt | Commit | Attention, ms | Prompt, tokens/s |
|---|---|---|---|
| 15,922 tokens | `d2aedef` | 485.3 / 478.7 | 4,236.9 / 4,224.4 |
| 15,922 tokens | `3ef4599` | 805.5 / 802.4 | 3,927.9 / 3,896.4 |
| 29,120 tokens | `d2aedef` | 894.9 / 895.0 | 4,349.0 / 4,341.9 |
| 29,120 tokens | `3ef4599` | 1,598.9 / 1,601.1 | 3,893.2 / 3,879.7 |

| Commit | KL against the 4-bit model, mean | Top-1 agreement |
|---|---|---|
| `d2aedef` | 0.16076 | 90.38% |
| `3ef4599` | 0.16175 | 90.28% |

## Notes

- The kernel took a KV group and 4 rows: 48 (row, head) MMA rows in 3 tiles, 12 warps. It sorted the rows' selected
  blocks and merged them with the rows' tails into one key order. Each row masked the keys that it did not select.
- It passed the kernel test at 1e-5 against `qwen4_attend`, and its KL was within 0.6%. Only the speed failed.
- One block of 12 warps fits on an SM, against 4 blocks of 4 warps before. One thread merged the lists for each tile,
  while the other 383 waited.
- Earlier results point the same way. Pipelined loads (`cdf722e`) were slower, and 22% fewer MMAs gained 6%.
