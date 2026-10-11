# NVIDIA GPUs (CUDA)

The CUDA engine (`engine/mova_cuda.c`, `engine/kernels_moe.cu`) implements the same engine API as the Metal engine,
with the same kernel arithmetic and BF16 rounding points. The tools, the tokenizer and the tests are the same C
sources. Read [Metal and CUDA parity](../concepts/parity.md). To build, read [Build and test](build.md).

A discrete GPU usually has less memory than the 22.8 GB model. The CUDA engine uses four methods to run the model.

## How the engine fits the model

1. **Expert cache.** All routed and value experts stay in pinned host memory. VRAM holds as many experts as fit. A
   GPU-side cache (least recently used, filled after each router) copies missing experts over PCIe before they run.
   In a conversation, the cache serves about 90% of expert uses. The results never depend on the cache contents.
   `test_engine` checks this bit for bit.
2. **Prompts layer by layer.** The engine computes prompts one layer at a time, over up to 8192 tokens. It uses the
   tensor cores (BF16 `mma` with f32 sums). While one layer computes, the engine copies the experts of the next layer.
3. **CUDA graphs for decode.** Each token is one graph launch.
4. **A split KV cache.** VRAM holds as many positions as fit. Pinned host memory holds the rest. While a prompt
   computes a layer, the engine copies the host rows of that layer to VRAM.

## Long contexts

Past a few tens of thousands of tokens, the KV cache sets the speed. This model has 48 full-attention layers, and
each token reads the full cache. On a 16 GB card, most of the cache is in host memory, behind PCIe.

- `--kv q8` (int8 with a scale for each token and head) halves the cache. Its rounding is about the size of BF16
  rounding. On a short text, KL against the reference went from 0.005 to 0.010.
- On a 64 GB PC, contexts above about 100K tokens need `--kv q8`. A 200K-token BF16 cache alone is 39 GB.
- MLA models take `--kv fp8` or `--kv fp4`. Read [The KV cache](../concepts/kv-cache.md).

## Measured speed

Read [CUDA and Metal measured side by side](../results/2026-10-07-cuda-vs-metal.md) and
[MLA kernels on Metal and CUDA](../results/2026-10-08-mla-kernels.md).

## Environment variables

Read [Environment variables](../reference/env.md). The most useful are `NSLM_VRAM_RESERVE_MB` (VRAM to keep free) and
`NSLM_CACHE_STATS=1` (expert cache hits at exit).

## See also

- [Build and test](build.md)
- [Metal and CUDA parity](../concepts/parity.md)
