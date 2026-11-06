# The KL of the 4-bit PLE table (2026-11-06)

**Result.** The 4-bit MLX PLE table that the packs ship adds 0.0060 of KL against BF16. In llama.cpp, Q8_0 with
Unsloth's Q8_0 table gives 0.0199, and the same weights with the 4-bit table give 0.0259 (+30%). Top-1 agreement falls
from 95.16% to 94.59%. So the start scripts download Unsloth's Q8_0 table by default, and the cards recommend it.

**Status.** Current. It measures the table of the release packs, P3 and P4. Their KL on the cards is with the Q8_0
table.

## Setup

- HF job `6aee1d9d`: `rtx-pro-6000x2`, 7.3 minutes, about $0.66.
- llama.cpp `5ce06fc`, `llama-perplexity -c 2048 --chunks 64` on wikitext-2 (test). The reference is the saved BF16
  run (`ref-c64.kld`), with BF16's own table.
- Weights: Unsloth's Q8_0 shards of `unsloth/Qwen3.8-Flash-Next-GGUF` at `d9620119`.
- The 4-bit table: P3's `ple_embedding` shards (4-bit MLX, group 32, from mlx-community's 4-bit model).
  `nslm-qwen4-emulate --ple` (`4373d30`) wrote all 320,001,536 rows over `per_layer_token_embd.weight` in 12 s, rounded
  to Q8_0, as the engine reads them (`qwen4_ple_shard`).
- Command: `PACKS="Q8_0 ple:P3" REF=/b/kld-ref bash tools/hf_llama_kld.sh`.

## Results

| PLE table | Mean KL | Median KL | 99th percentile | Maximum | Top-1 agreement |
|---|---|---|---|---|---|
| Unsloth's Q8_0 (54.4 GB) | 0.01991 ± 0.00030 | 0.00501 | 0.232 | 7.30 | 95.16% |
| The packs' 4-bit MLX (32 GB) | 0.02588 ± 0.00038 | 0.00604 | 0.314 | 9.92 | 94.59% |

Earlier pages give BF16's own table against Q8_0's: 0.00005, inside the standard error
([trellis experts against BF16](2026-11-05-trellis-kld.md)).

## Notes

- The difference is 12 times its standard error. The 99th percentile rises 35%: the 4-bit table costs most on the hard
  tokens.
- KL is not exactly additive. On the release packs (0.0415 to 0.0778 with the Q8_0 table), the 4-bit table may add
  about 0.006. That pack-by-pack measurement waits for the next weekly budget.
- The emulation rounds the 4-bit values to Q8_0 again. That adds a little error, as for the expert emulations.
- The Q8_0 table costs 22.4 GB more disk than the 4-bit one. The engines memory-map it, so RAM use does not change: the
  page cache holds the rows that a session reads.
