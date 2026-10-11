# The KV cache

While the model reads a conversation, it keeps a note for each token. The note is what the model looks back at when
it writes the next word. The set of notes is the KV cache. At long contexts, the KV cache can need more memory than
the model weights.

## Key numbers

| Model and cache | Bytes per token (all 48 layers) | 200K tokens | 512K tokens |
|---|---|---|---|
| GQA (original), BF16 | 192 KiB | 39 GB | ≈103 GB |
| GQA, engine `--kv q8` | ≈100 KB | 20 GB | ≈52 GB |
| MLA, BF16 | 84 KiB | 17.2 GB | 45.1 GB |
| MLA, MLX loader q8 (default) | 45 KB | ≈9.0 GB | 24.0 GB |
| MLA, MLX loader rq4 | 27 KB | ≈5.4 GB | 14.3 GB |
| MLA, engine `--kv fp8` | 43 KiB | 8.9 GB | ≈23 GB |
| MLA, engine `--kv fp4` | 32 KiB | 6.6 GB | ≈17 GB |

The MLA rows are for the rank-768 conversion. Values marked ≈ are calculated from the bytes per token. The other
values are measured.

## Why MLA keeps a smaller cache

The original model keeps 8 keys and 8 values of 128 numbers for each token in each layer. The MLA model keeps one
latent vector of 768 numbers and one position (RoPE) key of 128 numbers. Read
[Multi-head latent attention](mla.md).

## The MLX loader: q8 and rq4

The MLX loader (`tools/nanoseedlm_k2.py`, the `trust_remote_code` model file in the Hugging Face repositories) keeps
the MLA cache in one of two compact forms:

- **q8** (the default) rounds each number to 8 bits, in groups of 64 with their own scale. It is half the size of
  BF16. We did not measure a quality loss.
- **rq4** keeps the latent in 4 bits and the position key in 8 bits. It is a third of the size of BF16.

Plain 4-bit rounding loses too much. A few numbers in each latent are much larger than the others. Before rq4 rounds a
latent, it mixes the latent with a fixed Walsh-Hadamard transform. The signs of the transform come from the same
16-bit LFSR that makes the weights. Thus the transform is a seed, and the file stores nothing for it. The mix spreads
the large numbers out, and a small, measurable quality cost remains.

### The trade-offs

Measured with K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4 on an M4 Max (128 GB), MLX 0.32.3:

| | BF16 | q8 (default) | rq4 |
|---|---|---|---|
| cache per token | 84 KB | 45 KB | 27 KB |
| cache at 128K / 256K / 512K tokens | 11.3 / 22.5 / 45.1 GB | 6.0 / 12.0 / 24.0 GB | 3.6 / 7.1 / 14.3 GB |
| total memory at 512K (P = 4) | 69.3 GB | 48.4 GB | 38.5 GB |
| quality vs BF16 (held-out KL divergence, lower is better) | | +0.0004 | +0.0039 |
| quality at long context (log-likelihood of text 32K tokens in) | | +0.005% | +0.20% |
| picks the same next token as BF16 | | 96.4% | 94.6% |
| decode speed at 1K / 4K (tokens/s) | 19.5 / 18.8 | 19.3 / 18.1 | 17.7 / 16.9 |
| decode speed at 32K / 128K / 512K (tokens/s) | 12.5 / 6.2 / 1.3-2.0 | 12.7 / 5.8 / 1.84 | 12.5 / 6.3 / 2.13 |

For scale, the quality cost of rq4 is about half the difference between the P = 4 and P = 3 weights. The cost of q8
is about ten times smaller than the cost of rq4.

Speed:

- q8 decodes within a few percent of BF16 at short contexts, and as fast or faster from 32K.
- rq4 is a little slower at short contexts, because it mixes each latent.
- rq4 is the fastest from 128K. It reads the fewest bytes, and its kernel reads the 4-bit values without a separate
  unpack step.

### Prefix reuse in oMLX

Prefix reuse keeps a long conversation responsive. When oMLX loads the model, the loader registers q8 and rq4 with the
oMLX prefix cache, in memory and on SSD. Measured with a 13,000-token prompt on the M4 Max:

| Request | q8 | rq4 |
|---|---|---|
| First request, time to first token | 42 s | 42 s |
| Same prefix again | 3.8 s | 3.6 s |
| Same prefix after an oMLX restart (from the SSD cache) | 1.2 s | 1.1 s |

A future oMLX can change the internals that this feature uses. If that occurs, the loader keeps the BF16 cache and
writes a message. Prefix reuse is more important than the cache size.

### How the loader chooses

The loader uses q8. If q8 cannot hold the context that you ask for, the loader uses rq4. The available memory is the
GPU working set that MLX reports, minus the weights and a reserve for prompt processing. When the loader picks rq4,
it writes one message on stdout:

```
NanoSeedLM: KV cache rq4 (4-bit latent) instead of q8: max_context 262,144 tokens needs 12.0 GB as q8, more than the
10.0 GB available after the weights; rq4 needs 7.1 GB at a small quality cost (...)
```

If rq4 also cannot hold the context, the message gives the number of tokens that fit.

The loader sets the cache format when it creates the cache for a conversation. Thus you must tell the loader the
longest context before you start. Use one of these methods:

- Set `NSLM_MAX_CONTEXT` (tokens) in the environment of the process that loads the model, for example oMLX.
- Add `"mla_max_context": 262144` to the `config.json` of the model folder.
- If you call the model from Python, use `model.make_cache(max_context=262144)`.

If you use none of these methods, the loader uses q8.

### How to choose the format yourself

Set the `config.json` key `mla_kv` to one of these values:

| Value | Cache |
|---|---|
| `"auto"`, or no key | The loader chooses, as above. |
| `"bf16"` | The uncompressed cache. Fastest decode, most memory. |
| `{"bits": 4, "rope_bits": 8, "group_size": 64, "rotate": true}` | Always rq4. |
| `{"bits": 8, "rope_bits": 8, "group_size": 64, "rotate": false}` | Always q8. |

### How much context fits on your Mac

This table gives the longest context for each model and memory size. It uses the default macOS GPU memory limit. The
limit is about two thirds of memory up to 36 GB, and three quarters above 36 GB. MLX reports this limit as the
recommended working set. The table subtracts the weights (P = 3 20.1 GB, P = 4 22.2 GB, oQ4e 22.9 GB) and 2.5 GB for
prompt processing. 512K is the maximum context of the model.

| unified memory | GPU-usable | model | BF16 | q8 | rq4 |
|---|---|---|---|---|---|
| 32 GB | ~23 GB | P = 3 | none | 7K | 12K |
| 32 GB | ~23 GB | P = 4, oQ4e | none | none | none |
| 36 GB | ~26 GB | P = 3 | 36K | 68K | 114K |
| 36 GB | ~26 GB | P = 4 | 12K | 22K | 38K |
| 36 GB | ~26 GB | oQ4e | 4K | 8K | 13K |
| 48 GB | ~39 GB | P = 3 | 182K | 343K | **512K** |
| 48 GB | ~39 GB | P = 4 | 158K | 298K | 499K |
| 48 GB | ~39 GB | oQ4e | 150K | 283K | 475K |
| 64 GB | ~52 GB | any | ~300K | 512K | 512K |
| 96 GB and up | 77 GB+ | any | 512K | 512K | 512K |

Measured total memory at 512K tokens on the M4 Max:

| Model | Cache | Total |
|---|---|---|
| P = 3 | rq4 | 36.4 GB |
| P = 4 | rq4 | 38.5 GB |
| P = 4 | q8 | 48.4 GB |
| P = 4 | BF16 | 69.3 GB |

If you raise the GPU memory limit (`iogpu.wired_limit_mb`), a Mac moves up the table. The loader reads the
limit that is in effect.

## The engine: q8, fp8 and fp4

The C engine has its own cache options:

- **GQA models:** `--kv q8` keeps the cache as int8 with a scale for each token and head. It works on Metal and CUDA.
  It halves the cache. On an M4 Max, the quality cost is within about one standard error, and decode at 4K context
  is about 10% slower. Read [Engine KV cache codes](../results/2026-10-08-engine-kv-codes.md).
- **MLA models:** `--kv fp8` and `--kv fp4` use the codes of DeepSeek-V4.1 (`nslm/kvq.h`). The engine quantizes each
  value when it writes it. Each reader decodes the value exactly to BF16, so the attention arithmetic does not change.
  - `fp8` keeps the latent and the RoPE key as FP8 E4M3, with a power-of-two scale for each 32 values. It is 52% of
    BF16.
  - `fp4` keeps the RoPE key and the first 256 dimensions of the latent as FP8. It keeps the other latent dimensions
    as FP4 E2M1, with an E4M3 scale for each 16 values. It is 38% of BF16.
  - TransMLA latents put most of their variance in the first dimensions. An FP4 RoPE key caused most of the quality
    loss of the all-FP4 layout of DeepSeek. Thus `fp4` keeps those parts in FP8.

The engine also has two memory features for the cache:

- **Paged cache (Metal, macOS 26.4 or later).** The engine reserves `--ctx` for each slot but commits memory in 64 KB
  pages as it writes positions. It gives the pages back when a slot becomes free or holds a shorter conversation.
  `NSLM_KV_DENSE=1` commits all memory at the start.
- **Cold cache on disk.** `nslm-serve` saves long caches to disk in 256-token blocks and reads them back for a later
  request with the same prefix. Read [Run and serve](../guides/run-and-serve.md).

## See also

- [Multi-head latent attention](mla.md)
- [Engine KV cache codes](../results/2026-10-08-engine-kv-codes.md)
- [The MLX loader](../guides/mlx.md)
