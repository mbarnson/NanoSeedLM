# Multi-head latent attention (MLA)

The original K2-Horizon-MoVA uses grouped-query attention (GQA). For each token and layer, it caches 8 keys and 8
values. [TransMLA](https://arxiv.org/abs/2502.07864) converts GQA to multi-head latent attention (MLA). The MLA model
caches one latent vector and one position key for each token and layer. The cache is less than half the size. The
cost is some quality and more arithmetic in attention.

## Key numbers

| | GQA (original) | MLA (rank 768) |
|---|---|---|
| Cache per token, BF16, all layers | 192 KiB | 84 KiB |
| Cache at 200K tokens, BF16 | 39 GB | 17.2 GB |
| KLD against the original BF16, held-out text | 0 (reference) | 0.144 (MLA BF16) |
| Long generations that fell into a loop (oMLX, 48 runs) | 0 | 2 |

Read [The KV cache](kv-cache.md) for the compact cache formats.

## How it works

### The conversion

The conversion has two steps. These tools are not in this repository. The
[MLA model card](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA) gives the details.

1. **TransMLA conversion.** A calibrated conversion with no training. It gave a KLD of 0.154 against the original on
   8 held-out windows (tag `naive-j768`).
2. **Distillation heal.** Training of only the latent attention projections (142 M parameters) on 49.9 M tokens. It
   reduced the KLD on the same windows to 0.091 (branch `heal-pilot`, now `main`).

### The cache

For layer `l`, with input `x`:

```
c      = kv_a_x x  (+ kv_a_v v, where v comes from the value experts)
k_rope = RoPE(k_rope_proj x)
```

`c` is the latent, with rank `r_l`. `k_rope` is one 128-dimension key. The dense layers 0 to 2 fold `v_proj` into
`kv_a_x`.

### The attention

For query head `a`, with query `q_a`:

```
score_a(t) = (q_lat_a q_a) . c_t + RoPE(q_rope_mix_a q_a) . k_rope_t      (scale 128^-0.5, as the original)
o_a        = v_up_a (sum_t p_t c_t)
```

The output then goes through the softplus gate and `o_proj`, as in the original.

There are two ways to compute this:

- **Absorbed.** The query goes into latent space. Attention is then multi-query attention over the latent. This mode
  reads the cache once, but it does about 6 times the arithmetic of GQA.
- **Expanded.** Each head computes its keys `q_latᵀ c` and values `v_up c` from the latent. Attention is then normal
  multi-head attention. This mode does about a quarter of the absorbed arithmetic, but it must compute keys for all
  cached positions.

### The engines

| Work | Metal | CUDA |
|---|---|---|
| Decode | Absorbed. `k_mla_attn`: flash style, 16 query heads per threadgroup, 64-key tiles, online softmax, split keys | Absorbed. `k_mla_attn_tc`: 16 query heads per block, 16-key tiles double-buffered by `cp.async`, BF16 `mma` |
| Prompts of 256 or more new tokens | Expanded. `k_mla_prefill`, keys expanded 512 positions at a time | Expanded. `k_mla_prefill`, keys and values from tensor-core GEMMs |
| Prompts of fewer than 256 new tokens | Absorbed, so the engine does not expand all cached positions | Absorbed, as on Metal |

On both engines, expansion stops at the last multiple of 256 positions (`mla_expand_min`). Thus a prompt that the cold cache restores in
256-token blocks gets the same cache, bit for bit. On an M4 Max, 15 new tokens after 12K cached tokens take 0.57 s
absorbed. Expanded, they took 2.8 s.

### Why MLA decode is slower than GQA decode

At rank 768, the MLA projections are 945 MB in BF16. Decode must read them for each token. This cost is larger than
the time that the smaller cache saves at short contexts. At long contexts, the smaller cache reads fewer bytes, and
the difference becomes smaller.

## The quality cost

MLA moves the model away from the original. In the R-02 comparison, MLA BF16 had a held-out KLD of 0.144. GQA with P = 4
seeds had 0.022. Three of 31 held-out windows cause most of this difference. They are ordinary web prose with no
repetition. We do not know the cause yet. Read [Seeds against affine quantization](../results/2026-10-10-seeds-vs-affine.md).

The MLA model also falls into a loop more often in long generations. Set a limit on `max_tokens`, or set a reasoning
budget.

## Model folder entries

`config.json` has `mla_ranks` (the latent rank of each layer, a multiple of 32, at most 1024) and `mla_rope_dim`
(128). Read [The model folder format](../reference/model-folder.md).

## Measurements

- [MLA kernels on Metal and CUDA](../results/2026-10-08-mla-kernels.md)
- [MLA projections in BF16, Q8, Q4 and seeds](../results/2026-10-08-mla-projections.md)
- [Engine KV cache codes](../results/2026-10-08-engine-kv-codes.md)
- [Seeds against affine quantization (R-02)](../results/2026-10-10-seeds-vs-affine.md)

## See also

- [The KV cache](kv-cache.md)
- [TransMLA](https://arxiv.org/abs/2502.07864), Meng, Yao and Zhang, 2025
