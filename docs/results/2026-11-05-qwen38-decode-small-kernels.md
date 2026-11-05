# Qwen3.8-Flash-Next on CUDA: faster decode norms and RoPE, and the expert copy is PCIe-bound (2026-11-05)

**Result.** `0233258` makes P3 decode 3.6% faster on an RTX PRO 6000 with an 8 GB expert cache: 54.6 to 56.5 tokens per
second, mean of three runs on one node. The greedy text is the same, bit for bit. Two norm kernels and RoPE took 7 to 8
µs a call at one row. They now take 2.0 to 3.6 µs. The decode expert copy is 46% of the GPU time, and it does not get
faster with more blocks: PCIe is its limit.

**Status.** Current.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB, PCIe 5). Load average 5.6 to 10.5.
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`. Prompt: "Why do tide pools matter? Answer in three sentences." with the chat template. 92 greedy tokens.
- A/B: `369c978` and `0233258` built on one node, three runs each, alternately (`scratchpad/rowsjob/ab.sh`, job
  `6aec7074`). Kernel times: nsys with CUDA graph nodes (`nsysdec.sh`, jobs `6aec6d42` and `6aec6fb4`). Greedy text:
  `text.sh` (jobs `6aec6fb5` and `6aec6fb6`). Copy blocks: `cpblocks.sh` with a job-only patch (job `6aec70bc`).

```bash
nslm-qwen4-gen --model PACK --prompt "Why do tide pools matter? Answer in three sentences." --max-tokens 256 --expert-cache-gb 8
```

## Results

Decode, tokens per second. The runs of the two commits alternate on one node:

| Commit | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| `369c978` | 54.93 | 53.72 | 55.00 | 54.55 |
| `0233258` | 55.55 | 57.05 | 57.01 | 56.54 |

Kernels changed by `0233258`, µs a call (nsys, the prompt and 92 decode steps):

| Kernel | `369c978` | `0233258` | Calls |
|---|---|---|---|
| `k_q4_norm1p` | 7.4 | 2.0 | 4,887 |
| `k_q4_hc_add_norm` | 7.6 | 2.4 | 4,512 |
| `k_q4_head_norm_rope` | 8.0 | 3.6 | 3,384 |
| `k_q4_qsa_blocks` | 2.7 | 1.7 | 1,128 |

The largest decode kernels at `0233258`, share of the GPU time:

| Kernel | Owner | Share | µs a call |
|---|---|---|---|
| `k_cache_copy` (expert misses over PCIe) | pc-cuda | 46.3% | 215 |
| `k_mv1_q4` (dense Q4 matvecs) | pc-cuda | 19.1% | 12.6 |
| `k_mv_gu_seed4` + `k_mv1_seed4` (routed experts) | pc-cuda | 12.8% | 38.5, 22.7 |
| `k_cache_admit` | pc-cuda | 2.3% | 10.7 |
| `k_q4_route` | cloud | 1.8% | 8.2 |
| `k_q4_attend` (decode attention) | cloud | 1.4% | 25.5 |

The decode copy by its blocks (two runs each, 56.5 to 56.8 tokens per second in all 12 runs):

| Blocks | 32 | 64 (the default) | 128 | 256 | 512 | 1,024 |
|---|---|---|---|---|---|---|
| Decode, tokens/s | 56.48 / 56.80 | 56.77 / 56.78 | 56.69 / 56.75 | 56.75 / 56.79 | 56.81 / 56.82 | 56.80 / 56.77 |

## Notes

- The norms read about ten values a thread in a chain of loads. They now load a thread's values into registers
  first. The summation order is the same, so the results are the same.
- RoPE took an f64 `pow` for each element. The engine now builds the table of inverse frequencies once, with the C
  reference's formula, and RoPE calls one `sincos`. Metal already reads such a table.
- Both commits give the same greedy text (md5 `5d12b308dd51`), and the same 32,562 hits and 20,022 misses.
- The copy reads mapped host memory. With 32 or 1,024 blocks it takes the same time, so the PCIe link is its limit.
  Fewer bytes a token would help: a higher hit rate, smaller experts, or several tokens a forward (MTP).
- The dense Q4 matvec takes about 10 times the time of its weight bytes. It is the largest decode cost that does not
  depend on PCIe. pc-cuda has the request for split-K.
