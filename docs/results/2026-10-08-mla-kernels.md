# MLA kernels on Metal and CUDA (2026-10-08)

**Result.** Flash-style latent attention on Metal, and tensor-core attention on CUDA, brought MLA close to GQA speed.
On an RTX 4080, MLA decode is about 12% slower than GQA decode at 1K and 4K context. Prompt processing is within about
15%. Absorbed MLA attention does about 6 times the arithmetic of GQA attention. Expanded attention for long prompts
removes most of that cost.

**Status.** Current. The CUDA runs use the heal-pilot MLA delta over the p4mx-q4v seeds.

## Metal (M4 Max, q8mx)

| Prefill | Before (one pass) | Flash `k_mla_attn` | GQA |
|---|---|---|---|
| 2048 tokens | 21.3 s | 6.6 s | 3.2 s |
| 8192 tokens | 272 s | 64 s | 15 s |

Expanded attention for prompts of 256 or more new tokens (M4 Max, p4mx-q4v seeds):

| Prefill | Absorbed | Expanded |
|---|---|---|
| 2048 tokens | 6.4 s | 5.4 s |
| 8192 tokens | 57.9 s | 42.5 s |

Decode did not change. A short prompt after a long context stays absorbed: 15 new tokens after 12K cached tokens
take 0.57 s. Expanded, they took 2.8 s.

## CUDA (RTX 4080)

Command: `nslm-mova-bench --text holmes.txt --decode 256`. The one-pass kernel is the earlier `k_mla_attn` with
per-head mat-vecs.

| tok/s | MLA, one-pass kernel | MLA, current | GQA (same seeds) |
|---|---|---|---|
| prefill, 1K / 4K / 8K prompt | 499 / 186 / 86 | 1369 / 1726 / 1556 | 1416 / 1925 / 1787 |
| decode, 1K / 4K / 8K context | 37.7 / 32.1 / 20.0 | 54.1 / 51.2 / 47.7 | 61.2 / 58.6 / 50.6 |

## Notes

- On CUDA, an absorbed query row reads the 1.8 KB latent of each key again. On the RTX 4080, L2 bandwidth, not the
  tensor cores, limited the absorbed prompt kernel. The expanded prompt kernel does about a quarter of the arithmetic
  and of the L2 traffic.
- Expanded attention moves the rounding points. Held-out KLD against BF16 (8 windows, standard error 0.002): expanded
  0.0996, absorbed 0.1000, earlier one-pass kernel 0.1001.
- MLA decode is slower than GQA decode mostly because it reads the MLA projections: 945 MB per token at rank 768 in
  BF16.
