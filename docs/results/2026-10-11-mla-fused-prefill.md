# A fused MLA prompt kernel in the MLX loader (2026-10-11)

**Result.** In the MLX loader, a fused Metal kernel processes MLA prompts 27% to 38% faster at 32K to 256K tokens of
context, and it uses 1.3 GB to 1.5 GB less transient memory. Decode does not change. The kernel is more accurate
than the path that it replaces, but the T-03 healed model scores slightly worse with it (held-out NLL +0.0003, KL
divergence +0.0009). That model was healed and selected with the old numerics. Prompt processing at long context is
still limited by attention arithmetic. MLX's own attention kernel in the hybrid model's GQA layers is faster per layer.

**Status.** Current.

## Setup

- Machine: M4 Max, 128 GB, AC power, MLX 0.32.3, oMLX 0.7.1.dev1 (its K2-Horizon patch).
- Models: K2-Horizon-MoVA-36B-A4B-MLA-oQ4e (all 48 layers MLA, rank 768), and the oQ4e of the hybrid T-03 model (38
  layers MLA, 10 layers GQA; `mla_ranks` 0). Cache: the loader's default for these runs.
- Command: NSLMv2 `tools/k02_longmem.py`. It fills each layer's cache to N tokens with values that have the statistics
  of a real prompt, then times one 2,047-token prompt chunk at N (the prompt worst case) and 8 decode steps. One run
  each. The unfused runs set `FUSED_PREFILL = False` in the model file.
- Machine state for each run is in its `.kv` file (start and end: power, load, other CPU).

## Results

Prompt chunk speed at context N (tokens per second):

| Model | Kernel | 32K | 128K | 256K |
|---|---|---|---|---|
| MLA oQ4e | Unfused | 103 | 29 | 15 |
| MLA oQ4e | Fused, 32 query rows per threadgroup | 115 | 33 | 17 |
| MLA oQ4e | **Fused, 64 query rows (default)** | **137** | **40** | **19** |
| Hybrid T-03 oQ4e | Unfused | 121 | 35 | 17 |
| Hybrid T-03 oQ4e | **Fused, 64 query rows (default)** | **155** | **45** | **22** |

At 128K, 128 query rows per threadgroup gave 41 tokens per second, the same as 64 rows. The default is 64, because
short prompt chunks waste fewer rows.

Transient memory of a prompt chunk: unfused 3.4 / 3.7 / 3.8 GB at 32K / 128K / 256K (MLA oQ4e), fused 2.0 / 2.1 / 2.3
GB.

Accuracy:

| Check | Unfused | Fused |
|---|---|---|
| Attention against a float32 reference (unit test, real head shapes) | 0.007 | 0.004 |
| Hybrid T-03 BF16, held-out NLL (31 windows; original 2.375645) | 2.397519 | 2.397847 |
| Hybrid T-03 BF16, held-out KL divergence | 0.10455 | 0.10542 |
| Hybrid T-03 BF16, 16K prompt: unfused against fused, last 2,048 positions | | KLD 0.0029, top-1 96.3% |

The 16K prompt in 2,048-token chunks took 58.5 s unfused and 48.9 s fused (BF16, cold).

## Notes

- The kernel stages each 32-key tile of expanded keys (`[k_a ; k_rope]`, 256 values) and values (128) in threadgroup
  memory. It computes the scores and P.V as 8 x 8 simdgroup products in float32, with an online softmax in registers.
  MLX mat-muls expand the latent for each block of keys. The block size comes from `SCORE_BUDGET` (32K keys at 512 MB).
- Per-kernel counters (Instruments, quiet GPU, 512 queries x 8K keys): the kernel has no register spills (L1 stack
  residency 0). F32 utilization is 51% (MLX `steel_gemm` 92%, `steel_attention` 85%), occupancy 16%, L1 register
  residency 54%, and the last-level cache limiter 93%. Two variants that cut registers and conversion work, Q and
  K^T as BF16 `simdgroup_load` operands, and the engine's `k_mla_attn` pattern (scores and softmax in threadgroup
  memory), were not faster on a quiet GPU (210 ms against 210 ms, and 379 ms against 239 ms, at 2,048 x 32K). The
  likely limit is cache traffic: every query tile reads each head's expanded keys and values again.
- MLX 0.32.3 `scaled_dot_product_attention` with 256-dim keys and 128-dim values does not use its fused attention.
  It runs `steel_gemm`, `looped_softmax_precise` and `steel_gemm`, with the score matrix in memory. At 2,048 x 32K
  that is 164 ms plus 28 ms of expansion, against 239 ms for this kernel, but it needs about 4 GB of scores per
  layer and block.
- The T-03 model was healed and selected with the unfused path. A heal that uses the fused path removes the small
  quality difference.
