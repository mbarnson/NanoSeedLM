# Prefill of Qwen3.8-Flash-Next: larger chunks, an expert prefetch and five kernel changes (2026-11-05)

**Result.** On an RTX PRO 6000, a 15,922-token prompt of the P3 pack went from 1,343 to 2,801 tokens per second.
A 29,120-token prompt went from 1,414 to 2,782, and from 746 to 1,571 on an A100. Chunks of 4,096 prompt rows gave
the largest gain, because each expert's GEMM tiles now fill. The next layer's expert copies now run beside the layer's
compute. Decode did not change (44.0 tokens per second).

**Status.** Current. One prompt text, one pack. The RTX 4080 is not measured.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB), and `a100-large` (A100 80 GB) for
  one table. Each row is a different job and node. Other jobs shared the nodes: the load average was 3 to 12. Each
  row is the mean of two runs, and the two runs agree within 1.5%.
- Pack: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`, an 8 GB expert cache (3,255 slots).
- Prompts: the first 64,000 characters of the wikitext-2 test set (15,922 tokens, `--ctx 16384`), and the first
  120,000 characters (29,120 tokens, `--ctx 32768`).
- Scripts: `scratchpad/rowsjob/prof.sh` and `prof32.sh` (`nslm-qwen4-gen` twice, then once with `Q4E_PROFILE=1`).

```bash
Q4E_PROFILE=1 nslm-qwen4-gen --model PACK --prompt "$(cat prompt.txt)" --raw --max-tokens 2 --ctx 16384 --expert-cache-gb 8
```

## Results

The 15,922-token prompt on the RTX PRO 6000:

| Commit | Change | Tokens/s |
|---|---|---|
| `6ddfdc7` | Start: one pass holds the context ([prefill by stage](2026-11-04-qwen38-prefill-stages.md)) | 1,343 |
| `f2caa8b` | Attention: a thread issues all its tile loads before its stores | 1,415 |
| `840e3ed` | DeltaNet: the conv computes all tokens at once, and the step stages 16 tokens in shared memory | 1,512 |
| `a313e6d` | Prompt chunks of 4,096 rows, not 1,024 (`--prompt-rows`) | 2,058 |
| `b80776c` | The stages' buffers share one VRAM region (0.25 MB a row, not 0.7 MB) | 2,048 |
| `1d7d575` | Attention: shared tile rows padded by 8 floats, and 8 threads for each softmax row | 2,175 |
| `0a38b16` | Attention: V takes the K tile after S (39 KB a block: 2 blocks an SM, not 1) | 2,404 |
| `d8e8a63` | The next layer's experts come over PCIe on a second stream | 2,801 |

Chunk rows at `7ae98a6` (before the DeltaNet change), the same prompt:

| Rows | Tokens/s | Experts, ms | DeltaNet, ms |
|---|---|---|---|
| 1,024 | 1,414 | 3,268 | 1,113 |
| 2,048 | 1,610 | 2,172 | 1,492 |
| 4,096 | 1,789 | 1,590 | 1,671 |

At `a313e6d`, with the DeltaNet change, 2,048 rows give 1,802 tokens per second and 4,096 rows give 2,058.

The 29,120-token prompt:

| GPU | `6ddfdc7`, tokens/s | `d8e8a63`, tokens/s |
|---|---|---|
| RTX PRO 6000 | 1,414 | 2,782 |
| A100 | 746 | 1,571 |

GPU time by stage at `d8e8a63` (the 15,922-token prompt, RTX PRO 6000):

| Stage | ms | Share |
|---|---|---|
| Experts | 1,604 | 28.9% |
| Attention | 1,253 | 22.6% |
| Hyper-connections | 897 | 16.1% |
| Projections | 877 | 15.8% |
| Gated DeltaNet | 490 | 8.8% |
| QSA select | 120 | 2.2% |
| Norms, RoPE and stores | 110 | 2.0% |
| PLE | 105 | 1.9% |
| Router and cache | 89 | 1.6% |

## Notes

- At 1,024 rows, each of the 512 experts got about 20 of a chunk's 10,240 routed rows (top 10). That is a sixth of
  a 128-row GEMM tile. At 4,096 rows it is about 80.
- An nsys run at `b80776c` put the most GPU time in five kernels:

  | Kernel | ms | Calls |
  |---|---|---|
  | Prompt attention | 2,390 | 48 |
  | Dense Q4 GEMMs | 1,572 | 2,696 |
  | Seed expert GEMMs | 1,272 | 576 |
  | Expert copies | 1,096 | 192 |
  | DeltaNet step | 411 | 144 |
- Some dense GEMMs have few output rows and a long K, so few blocks run. Examples are the HC inject (4 rows) and the
  HC down (320 rows) at K = 10,240. These kernels are pc-cuda's (`kernels_moe.cu`). A split-K path may help.
- The prompt attention kernel used 56 KB of shared memory a block. The RTX PRO 6000 has about 100 KB an SM, so one
  block of 4 warps ran on each SM. At 39 KB, two blocks run.
- `cdf722e` loaded the next tile while the block worked on this one. It took 168 registers and 1 block an SM, and
  attention took 2.74 s against 2.39 s. `7ae98a6` reverts it.
- The prefetch admits all 512 experts of the next layer and copies them with 8 blocks on a high-priority stream. It
  needs a cache of 3 layers' experts (1,536 slots). Beside it, the other stages took about 4% longer.
  `Q4E_NO_PREFETCH=1` turns it off.
- At 4,096 rows the prompt buffers take about 0.9 GB of VRAM. Before `b80776c`, 1,024 rows took about 0.7 GB.
- Decode of 256 tokens with CUDA graphs: 43.7 tokens per second at `f2caa8b`, 44.0 at `d8e8a63`.
- Matt reports that Strata IQ3_S fills a 32K prompt at about 2,700 tokens per second on the RTX 4080. These numbers
  come from a faster GPU, so they do not compare with it.
