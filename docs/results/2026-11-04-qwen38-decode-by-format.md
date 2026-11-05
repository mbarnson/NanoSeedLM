# Qwen3.8-Flash-Next decode with affine, P4 and P3 experts on one GPU (2026-11-04)

**Result.** The test used an RTX PRO 6000 with an 8 GB expert cache. P3 decodes at 46.1 tokens per second, the
4-bit affine model at 42.4 and P4 at 40.0. P3 moves 33% fewer bytes over PCIe than the affine model, and its seed expansion costs
1.7 ms more for each token. P4 moves 14% fewer bytes, but its expansion costs more than it saves on this PCIe 5 GPU.

**Status.** Current. One prompt, two runs of each model. PCIe 4 (the RTX 4080) doubles the cost of each byte, so the
seed packs may gain more there. That is not measured.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell, 96 GB, PCIe 5), job `6aebdf6ac78863f9c6331627`, commit
  `3adfba7` (CUDA graphs). The load average was 5.5 to 11.2 during the runs.
- Models: mlx-community/Qwen3.8-Flash-Next-4bit-mtp at `825df62` (affine Q4, group 32),
  [Qwen3.8-Flash-Next-NanoSeedLM-P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4) and
  [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3). The three differ
  only in the routed experts.
- Two runs of each model in turn, then one run of each with `Q4E_PROFILE=1` (no CUDA graphs):

```bash
nslm-qwen4-gen --model MODEL --prompt "Why do tide pools matter? Answer in three sentences." --max-tokens 256 --expert-cache-gb 8
```

## Results

| Experts | Bits a weight | MB an expert | Decode, tokens/s | Misses a token | PCIe, MB a token |
|---|---|---|---|---|---|
| Affine Q4, group 32 | 5.0 | 3.07 | 42.38 / 42.44 | 189 | 581 |
| P4 seeds | 4.5 | 2.76 | 39.97 / 40.00 | 182 | 502 |
| P3 seeds | 4.0 | 2.46 | 45.74 / 46.38 | 159 | 391 |

The stages of the profiled runs (ms for the whole run):

| Experts | Router and cache | Experts | Hyper-connections | Gated DeltaNet | Attention |
|---|---|---|---|---|---|
| Affine Q4 | 1,965 | 888 | 897 | 470 | 281 |
| P4 | 1,643 | 1,298 | 857 | 446 | 265 |
| P3 | 1,360 | 1,165 | 889 | 465 | 279 |

## Notes

- P4 generated 153 tokens before EOS, the others 161. Each model gave the same text in its two runs.
- The router and cache stage scales with the bytes that cross PCIe: about 48 GB/s here.
- The expert stage of all three formats is far above the time that their bytes need from VRAM (below 1 ms a token).
  The matvec kernels of decode wait on memory latency, not on bandwidth or arithmetic.
- A seed block costs P x 8 multiply-adds and a read of the 256 KB state table for each token. The state values are
  16-bit LFSR states, not signs, so a sign lookup table cannot replace them.
