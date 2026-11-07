# The 8-bit PLE table: row error against BF16 (2026-11-07)

**Result.** On 10,000 rows, the 8-bit table of [plan ple-8bit](../plans/2026-11-07-ple-8bit.md) is closer to BF16 than
Unsloth's Q8_0 table: its RMS error is 0.88 times Q8_0's. The packs' 4-bit table gives 15 times Q8_0's error. The
encoder takes each code against the BF16 scale and bias that it stores. `mx.quantize` takes them against the unrounded
ones, and gives 1.19 times Q8_0's error, so gate PA fails with it. Both forms are MLX's 8-bit affine form, group 32.

**Status.** Current. Gate PA passes with `tools/hf_ple8.py`'s encoder.

## Setup

- Rows: 100 runs of 100 rows, at random shards and offsets (seed 7), 10,000 rows of 160 values.
- BF16: `Qwen/Qwen3.8-Flash-Next` at `7ecef55a`, read by range requests.
- Q8_0: `engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file` of `unsloth/Qwen3.8-Flash-Next-GGUF` at `d9620119`, the same
  rows of `per_layer_token_embd.weight`.
- The 4-bit table: P3's `ngram_embedding` shards, as the five packs ship them.
- Values in f32, as the engines compute them.
- No HF job: the check ran in the cloud session.

## Results

| Table | RMS error against BF16 | Against Q8_0 |
|---|---|---|
| The packs' 4-bit MLX table (32.0 GB) | 6.17e-4 | 15.15 times |
| Unsloth's Q8_0 (54.4 GB) | 4.07e-5 | 1.00 |
| 8-bit, `mx.quantize` | 4.84e-5 | 1.19 |
| 8-bit, codes against the stored BF16 scale and bias (57.6 GB) | 3.58e-5 | 0.88 |

The RMS of the values is 7.63e-3.

## Notes

- The two 8-bit forms differ only in their codes. A BF16 scale has 8 bits of precision. So the rounding of the scale
  and bias moves each value by up to about half a step. Codes taken against the rounded values remove that error.
- On the tiny test model, `mx.dequantize` and the engines' `ns_row_f32` read the new codes as the encoder wrote them,
  with no difference.
- The KL that a table adds follows from its error: Q8_0's table adds 0.00005 against BF16's, and the 4-bit one 0.006.
