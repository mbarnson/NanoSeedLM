# Engine KV cache codes (2026-10-08)

**Result.** On GQA, the 8-bit cache (`--kv q8`) halves the cache at a quality cost within one standard error. On
MLA, `fp8` and `fp4` cut the cache to 52% and 38% of BF16 with almost no quality cost. The all-FP4 layout of
DeepSeek-V4.1 costs much more, because of its FP4 RoPE key.

**Status.** Current for the engine. For the MLX loader caches (q8, rq4), read [The KV cache](../concepts/kv-cache.md).

## GQA: `--kv q8` (M4 Max)

| Cache | KLD held-out / chat | Top-1 agreement | Decode at 4K |
|---|---|---|---|
| BF16 | 0.0260 / 0.0126 | | |
| q8 | 0.0267 / 0.0136 | unchanged | about 10% slower |

The differences are each within about one standard error. A 200K-token cache takes about 20 GB instead of 39 GB.

## MLA: `--kv fp8` and `--kv fp4`

Model: J768 conversion with p4mx-q4v seeds. Held-out KLD against BF16, 31 windows of 2048 tokens,
standard error 0.003.

| MLA cache | Bytes per token and layer | 200K tokens | KLD | Top-1 agreement |
|---|---|---|---|---|
| BF16 | 1792 | 17.2 GB | 0.2199 | 80.35% |
| `fp8` | 924 | 8.9 GB | 0.2221 | 80.20% |
| `fp4` | 684 | 6.6 GB | 0.2223 | 80.22% |
| FP4 with 128 leading FP8 latent dimensions | 624 | 6.0 GB | 0.2264 | 80.14% |
| FP4 with 0 leading FP8 latent dimensions | 564 | 5.4 GB | 0.2307 | 79.81% |
| All FP4 (DeepSeek-V4.1 layout) | 504 | 4.8 GB | 0.2677 | 78.10% |

For comparison, the GQA cache at 200K tokens is 39 GB, or 20 GB with `--kv q8`.

## Notes

- TransMLA latents put most of their variance in the first dimensions. Thus `fp4` keeps the first 256 latent
  dimensions and the RoPE key in FP8.
- The CUDA engine has the same `fp8` and `fp4` codes, with the same held-out KLD as Metal.
