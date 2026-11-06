# P4 as shipped against BF16 (2026-11-06)

**Result.** P4 as shipped gives a KL of 0.169 against BF16, with Unsloth's Q8_0 PLE table. With the Q8_0 dense
weights, P4's experts give 0.0408. So the 4-bit MLX dense weights that P4 ships add about 0.128, three times the KL of
the experts, as for P3 (0.182 against 0.0600). Top-1 agreement is 84.6%.

**Status.** Current. It fills the "not measured" cell for P4 on the model cards.

## Setup

- HF job `6aee650c`: `rtx-pro-6000x2`, 8.9 minutes, about $0.81.
- llama.cpp `5ce06fc`, `llama-perplexity -c 2048 --chunks 64` on wikitext-2 (test). The reference is the saved BF16
  run (`ref-c64.kld`), as on [the PLE table page](2026-11-06-ple-table-kld.md).
- Weights: Unsloth's Q8_0 shards of `unsloth/Qwen3.8-Flash-Next-GGUF` at `d9620119`, with each tensor that P4 ships
  written over them by `nslm-qwen4-emulate` (`4373d30`). The experts come from P4 (`4157e09`). The dense weights are
  P3's 4-bit MLX tensors (`5753a67`), which P4 ships too. The PLE table is the Q8_0 shard's own, as the start scripts
  download.
- Command: `PACKS="Q8_0 P4/mlx" REF=/b/kld-ref bash tools/hf_llama_kld.sh`.

## Results

| Weights | Mean KL | Median KL | 99th percentile | Maximum | Top-1 agreement | PPL against BF16 |
|---|---|---|---|---|---|---|
| Q8_0 | 0.01991 ± 0.00030 | 0.00501 | 0.232 | 7.30 | 95.16% | +0.17% |
| P4 as shipped | 0.1693 ± 0.0014 | 0.0648 | 1.69 | 8.55 | 84.57% | +9.6% |

The Q8_0 row repeats job `6aee1d9d` to five digits, so the two jobs use the same reference.

## Notes

- KL on the cards, as shipped and with the Q8_0 PLE table: TCQ4 0.0415, TCQ3-D4 0.0638, TCQ3 0.0778, P4 0.169,
  P3 0.182. The TCQ packs ship 8-bit dense weights, and P3 and P4 ship 4-bit ones.
- In mlx-vlm and oMLX, a pack uses its own 4-bit PLE table, which adds about 0.006
  ([the PLE table page](2026-11-06-ple-table-kld.md)).
