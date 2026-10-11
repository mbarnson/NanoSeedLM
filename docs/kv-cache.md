# The KV cache: q8 and rq4

While the model reads your conversation it keeps a note for every token: what it looks back at when it writes the next
word. That set of notes is the KV cache. K2-Horizon-MoVA's MLA attention already keeps its notes small (one 768-number
summary and one 128-number position key per token per layer, instead of 8 full keys and values), but at long contexts
they still add up: written out in full (BF16), 512K tokens of notes take 45 GB, twice the model's weights.

So the MLX loader (`tools/nanoseedlm_k2.py`, the `trust_remote_code` model file of the HF repos) writes the notes in a
compact form. There are two:

- **q8** (the default): every number rounded to 8 bits, in groups of 64 with their own scale. Half the size of BF16,
  and no quality loss you can measure.
- **rq4**: the summary in 4 bits, the position key still in 8. A third of BF16. Plain 4-bit rounding loses too much,
  because a few numbers in each summary are much larger than the rest; so before rounding, each summary is shuffled by
  a fixed pattern (a Walsh-Hadamard transform with signs from the same 16-bit LFSR that generates the weights: a seed,
  nothing stored) that spreads the large numbers out. What is left is a small, measurable quality cost.

## The trade-offs

Measured with K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4 on an M4 Max (128 GB), MLX 0.32.3:

| | BF16 | q8 (default) | rq4 |
|---|---|---|---|
| cache per token | 84 KB | 45 KB | 27 KB |
| cache at 128K / 256K / 512K tokens | 11.3 / 22.5 / 45.1 GB | 6.0 / 12.0 / 24.0 GB | 3.6 / 7.1 / 14.3 GB |
| total memory at 512K (P = 4) | 69.3 GB | 48.4 GB | 38.5 GB |
| quality vs BF16 (held-out KL divergence, lower is better) | | +0.0004 | +0.0039 |
| quality at long context (log-likelihood of text 32K tokens in) | | +0.005% | +0.20% |
| picks the same next token as BF16 | | 96.4% | 94.6% |
| decode speed at 1K / 4K (tokens/s) | 20.6 / 19.6 | 20.0 / 19.1 | 19.2 / 17.6 |
| decode speed at 32K / 128K / 512K (tokens/s) | 13.1 / 6.5 / 2.0 | 11.0 / 5.1 / 1.55 | 11.3 / 5.7 / 1.8 |

For scale: rq4's quality cost is about half the difference between the P = 4 and P = 3 weights; q8's is about ten
times smaller than rq4's. Both compressed caches decode a little slower than BF16 (the notes are unpacked as they are
read); past 32K rq4 is slightly faster than q8 because it reads fewer bytes.

## How the loader chooses

q8, unless you ask for a context that q8 cannot hold in this Mac's memory; then rq4. "Memory" is the GPU working set
MLX reports, minus what is already loaded (the weights) and a reserve for prompt processing. When it picks rq4 it says
so once, on stdout:

```
NanoSeedLM: KV cache rq4 (4-bit latent) instead of q8: max_context 262,144 tokens needs 12.0 GB as q8, more than the
10.0 GB available after the weights; rq4 needs 7.1 GB at a small quality cost (...)
```

If even rq4 cannot hold the context, the message ends with how many tokens do fit.

The cache format is fixed when a conversation's cache is created, so tell the loader the longest context you plan to
use up front:

- with oMLX, or anything else that loads the folder: set `NSLM_MAX_CONTEXT` (tokens) in the environment of the process
  that loads the model, or add `"mla_max_context": 262144` to the model folder's `config.json`;
- calling the model yourself: `model.make_cache(max_context=262144)`.

Without any of these the loader uses q8.

## Choosing yourself

`config.json` key `mla_kv`:

- `"auto"` or absent: as above;
- `"bf16"`: the uncompressed cache (fastest decode, most memory);
- `{"bits": 4, "rope_bits": 8, "group_size": 64, "rotate": true}`: always rq4 (`{"bits": 8, "rope_bits": 8,
  "group_size": 64, "rotate": false}`: always q8).

## How much context fits on your Mac

The longest context each model can hold, by unified memory, with macOS's default GPU memory limit (about two thirds of
memory up to 36 GB, three quarters above; MLX reports it as the recommended working set), after the weights (P = 3
20.1 GB, P = 4 22.2 GB, oQ4e 22.9 GB) and 2.5 GB for prompt processing. 512K is the model's maximum.

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

Measured totals at 512K tokens on the M4 Max: P = 4 with rq4 38.5 GB, P = 3 with rq4 36.4 GB, P = 4 with BF16 69.3 GB.
A raised GPU memory limit (`iogpu.wired_limit_mb`) moves a Mac up the table; the loader reads the limit in effect.
