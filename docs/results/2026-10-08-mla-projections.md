# MLA projections in BF16, Q8, Q4 and seeds (2026-10-08)

**Result.** At the size of Q4, P = 4 seeds keep about two thirds of the quality that Q4 loses. GPTQ search helps a
little more. `v_up` is the most sensitive tensor, so `nslm-mova-pack --mla p4` keeps `v_up` in Q8 by default.

**Status.** Current for the differences between encodings. These runs use the J768 conversion and 31 windows of 2048
tokens. Do not compare their absolute KLD values with the R-02 values, which use other windows and the published model.

## Setup

- Model: J768 MLA conversion with p4mx-q4v seeds.
- Quality: held-out KLD against BF16, 31 windows of 2048 tokens. Paired differences use the same windows.
- Machine: M4 Max.

## Results

| MLA projections | Folder | KLD | top-1 agreement |
|---|---|---|---|
| BF16 | 23.56 GB | 0.2199 | 80.35% |
| Q8 | 23.12 GB | 0.2206 | 80.35% |
| P = 4 seeds, GPTQ | 22.88 GB | 0.2282 | 79.88% |
| P = 4 seeds, activation-weighted | 22.88 GB | 0.2296 | 79.91% |
| Q4 | 22.88 GB | 0.2471 | 79.00% |
| P = 4 seeds, GPTQ, `v_up` in Q8 | 22.96 GB | 0.2243 | 80.14% |
| P = 4 seeds, activation-weighted, `v_up` in Q8 | 22.96 GB | 0.2250 | |

Paired differences:

| Change | KLD difference |
|---|---|
| GPTQ against activation-weighted | -0.0014 ± 0.0009 |
| `v_up` in Q8, all layers | -0.0039 ± 0.0006 |
| `v_up` in Q8, layers 0 to 23 only (22.92 GB) | -0.0021 |
| `v_up` in Q8, layers 24 to 47 only | -0.0016 |

With the `fp4` KV cache: Q8 projections 0.2246, seed projections 0.2335.

## Notes

- On held-out rows, GPTQ reduces the output error of each projection by 10% to 45% (`v_up` about 45%). The KLD of the
  model moves less.
- The share of `v_up` is spread over the layers.
- Speed on Metal: prompt and decode with Q8 or seed projections run as fast as with BF16. Decode with seeds is about
  4% slower than with Q8.
- Search time on an M4 Max: activation-weighted 536 s for all 472 M weights. GPTQ 23 minutes.
