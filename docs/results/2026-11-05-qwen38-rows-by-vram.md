# Qwen3.8-Flash-Next prompt chunks of 16,384 rows on large GPUs (2026-11-05)

**Result.** On an RTX PRO 6000, a 29,120-token prompt of the P3 pack runs at 3,606 tokens per second in chunks of
4,096 rows, 4,096 in chunks of 8,192, and 4,222 in chunks of 16,384. The engine now takes 16,384 rows by default on
GPUs of 64 GB or more, for about 3 GB more VRAM. A 15,922-token prompt then runs at 4,165 tokens per second, in one
forward. Decode did not change (53.8 tokens per second).

**Status.** Current. One prompt text, one pack. GPUs below 64 GB, such as the RTX 4080, keep 4,096 rows.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB), and `a100-large` (A100 80 GB) for
  one row. Each row is a different job and node. The load average was 2 to 11 on the RTX PRO 6000 nodes and 7 to 15 on
  the A100 node.
- Pack, prompts and scripts as in [one-pass prompt attention](2026-11-05-qwen38-one-pass-attention.md).
  `prof32.sh` passes `--prompt-rows` from `ROWS`.

## Results

The 29,120-token prompt at `0dd88ab` on the RTX PRO 6000 (mean of two runs, within 1.3%):

| Rows of a forward | Tokens/s | Experts, ms | Activation VRAM |
|---|---|---|---|
| 4,096 | 3,606 | 2,897 | about 0.9 GB |
| 8,192 | 4,096 | 2,331 | about 1.9 GB |
| 16,384 | 4,222 | 2,114 | about 3.8 GB |

At `957a0f0` (the new default):

| GPU | Prompt | Tokens/s |
|---|---|---|
| RTX PRO 6000 | 15,922 tokens | 4,165 |
| A100 | 29,120 tokens | 1,800 to 2,104 (three runs) |

## Notes

- More rows a forward give each expert more rows in its grouped GEMM: at 4,096 rows, about 80 of a 128-row tile.
- The A100 node was busy (load average up to 15), and its runs differ by 15%.
- `0dd88ab` also gives decode rows (8 or fewer) their own QSA score kernel again. At one row the tiled kernel took
  13 us a call. Decode QSA time fell from 41 to 28 ms for 164 tokens.
- An L40S and an A10G job (Ada and Ampere, PCIe 4, nearer the RTX 4080) waited more than an hour for hardware.
