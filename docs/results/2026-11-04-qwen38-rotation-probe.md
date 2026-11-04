# Hadamard rotation and P = 4 seeds on Qwen3.8-Flash-Next experts (2026-11-04)

**Result.** The expert weights of Qwen3.8-Flash-Next are already close to Gaussian: kurtosis 3.06 and 3.00, where a
Gaussian has 3. A block Hadamard rotation does not change the plain error of P = 4 seeds. An activation-weighted fit
without rotation gives 6% to 10% less activation-weighted error than the rotated fits. Rotation removes this gain,
because it makes the activation weights almost uniform.

**Status.** Current. This probe measures weight error only, on 4 experts of one layer. KLD can show a different
result. It is the first test of hypothesis H3 of the
[Qwen3.8-Flash-Next plan](../plans/2026-11-04-qwen38-flash-next.md).

## Setup

- Machine: a cloud session, Xeon with 4 vCPUs, no GPU. The load average was below 0.1 before the run.
- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), BF16, layer 20, experts 0 to 3.
  `gate_up_proj`: 8 random rows of each expert (2560 inputs). `down_proj`: 32 random rows of each expert (640
  inputs). 10,240 blocks of 8 weights for each projection. Random seed 3407.
- Activation statistics: `imatrix_unsloth.gguf_file` of
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), 45 chunks of 18,432
  tokens. The 4 experts have 1,805 to 25,565 tokens each. `h` is the mean of x^2 for each input channel.
- Search: `nslm4_search_ref` (the C reference P = 4 search), all 65,535 seeds, exponents e, e - 1 and e + 1, refit
  on. The exponent bias comes from `nslm_choose_bias` over the blocks of each run. The weighting is `sqrt(h)` with no
  prior, which is not the same as the `n0` prior of `nslm-moe`.
- Rotation: a block-diagonal Hadamard matrix with random signs on the input dimension. The blocks are 512 wide for
  `gate_up_proj` and 128 wide for `down_proj`. The decoded weights go back to the original basis before scoring.
- Conditions:
  - A: activation-weighted fit, no rotation.
  - B: plain L2 fit, no rotation. This is the data-free fit of the SeedLM paper.
  - C: plain L2 fit, rotated.
  - D: rotated, weighted by the diagonal of `R diag(h) R^T`. Rotation makes these weights almost uniform.
- Scores, in the original basis:
  - Activation-weighted relative error: `sum(h * (w - w_hat)^2) / sum(h * w^2)`.
  - Plain relative error: `sum((w - w_hat)^2) / sum(w^2)`.

## Results

| Projection | Condition | Activation-weighted error | Against A | Plain error |
|---|---|---|---|---|
| `gate_up_proj` | A: weighted, no rotation | 0.00636 | 1.00x | 0.00710 |
| `gate_up_proj` | B: plain, no rotation | 0.00677 | 1.07x | 0.00677 |
| `gate_up_proj` | C: plain, rotated | 0.00674 | 1.06x | 0.00673 |
| `gate_up_proj` | D: weighted, rotated | 0.00674 | 1.06x | 0.00673 |
| `down_proj` | A: weighted, no rotation | 0.00615 | 1.00x | 0.00718 |
| `down_proj` | B: plain, no rotation | 0.00679 | 1.11x | 0.00675 |
| `down_proj` | C: plain, rotated | 0.00678 | 1.10x | 0.00673 |
| `down_proj` | D: weighted, rotated | 0.00678 | 1.10x | 0.00673 |

| Projection | Kurtosis | Kurtosis after rotation | Median over the 4 experts of max(h) / median(h) |
|---|---|---|---|
| `gate_up_proj` | 3.06 | 3.00 | 10 |
| `down_proj` | 3.00 | 2.99 | 10 |

## Notes

- The activation-weighted score uses only the diagonal of the input statistics. Fit A minimizes the same diagonal
  score, so the score favors A. The true output error also depends on the off-diagonal terms. QuIP# gets its gain
  from rotation together with error feedback over the full `X^T X`. This probe does not test that combination.
- SeedLM+O measured a Hadamard rotation at 29% more KL than an activation-weighted fit, on Qwen3-1.7B with the
  3-bit layout. Their reason was the same: the weight rows were already near-Gaussian.
- The probe also shows that the Unsloth imatrix can supply the statistics of the activation-weighted search.
- The scripts are not in the repository. A one-layer HF Job (step S2 of the plan) repeats this test on a full layer.
