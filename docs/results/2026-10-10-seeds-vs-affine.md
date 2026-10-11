# Seeds against affine quantization (R-02, 2026-10-10)

**Result.** On both the GQA and the MLA model, P = 4 seeds come a little closer to the original than a calibrated
4-bit affine quantization (oMLX oQ4e), at a slightly smaller size. P = 3 seeds are smaller again, and less accurate
than oQ4e. All four hypotheses held. The MLA conversion costs much more quality than the choice between seeds and
affine.

**Status.** Current. We wrote the hypotheses and the pass rules before we measured.

## Setup

- Machine: M4 Max, 128 GB, on AC power. Each run records machine state.
- Scoring: MLX, with the K2 patch of oMLX, as oMLX loads the models.
- Reference: the original GQA model in MLX BF16
  ([BF16 reference log-probs](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-bf16-ref-logprobs)).
  Held-out: 31 windows, 31,713 positions. Chat: 29 windows, 29,667 positions.
- GQA-S4 and GQA-S3 are evaluation packs of the original GQA model. They are not published. S4: experts P = 4, other
  tensors P = 8, routers Q8. S3: experts P = 3, other tensors as S4.
- GQA-oQ4e is `mlx-community/K2-Horizon-MoVA-36B-A4B-oQ4e`. Current oMLX does not load it, so the scorer renames its
  experts and dequantizes its routers to BF16. These changes do not change values.

## Scores

KLD against the original GQA BF16. Lower is closer.

| Model | Weights | KLD held-out / chat | Top-1 held-out | Median window KLD |
|---|---|---|---|---|
| GQA-oQ4e | 22.02 GB | 0.0310 / 0.0156 | 92.44% | 0.0207 |
| GQA-S4 | 21.92 GB | 0.0218 / 0.0097 | 93.49% | 0.0140 |
| GQA-S3 | 19.79 GB | 0.0364 / 0.0154 | 91.76% | 0.0226 |
| MLA-BF16 | 75.57 GB | 0.1436 / 0.0738 | 84.35% | 0.0820 |
| MLA-oQ8e | 40.61 GB | 0.1436 / 0.0739 | 84.38% | 0.0827 |
| MLA-oQ6e | 31.74 GB | 0.1446 / 0.0745 | 84.39% | 0.0825 |
| MLA-oQ4e | 22.89 GB | 0.1548 / 0.0789 | 83.83% | 0.0919 |
| MLA-P4 | 22.20 GB | 0.1535 / 0.0780 | 83.79% | 0.0896 |
| MLA-P3 | 20.07 GB | 0.1623 / 0.0820 | 83.20% | 0.0965 |

## Paired comparisons

For each window, diff = KLD(second) - KLD(first). A positive diff means that the first model is closer to the
original. 95% bootstrap intervals over windows, 10,000 resamples, stratified by split.

| Comparison | diff [95% CI] | Verdict |
|---|---|---|
| GQA-S4 against GQA-oQ4e | +0.00764 [+0.00605, +0.00952] | S4 is closer |
| GQA-oQ4e against GQA-S3 | +0.00269 [+0.00081, +0.00492] | oQ4e is closer |
| MLA-P4 against MLA-oQ4e | +0.00110 [+0.00033, +0.00186] | P4 is closer |
| MLA-oQ4e against MLA-P3 | +0.00543 [+0.00397, +0.00718] | oQ4e is closer |
| MLA-P4 against GQA-oQ4e | -0.09342 [-0.12819, -0.06562] | GQA-oQ4e is closer |
| MLA-P3 against GQA-oQ4e | -0.09995 [-0.13698, -0.07130] | GQA-oQ4e is closer |

## Where MLA loses

Three of 31 held-out windows (11, 12 and 31) cause most of the MLA mean. They are ordinary web prose with no repetition
in the window. Their MLA-P4 KLD is 0.47, 0.86 and 0.90. GQA-S4 has 0.05, 0.11 and 0.13 on the same windows. Repetition
does not correlate with KLD (r = -0.09). We do not know the cause. This is why the median window KLD of MLA-P4 (0.090)
is much lower than its mean (0.153).

## Checks

- GQA-S4: MLX 0.02176, C engine 0.0218 (same windows).
- MLA-P4 on 8 held-out windows: MLX 0.0985, C engine 0.0988.

## Sources

- The model cards of the [MLA models](../reference/models.md).
