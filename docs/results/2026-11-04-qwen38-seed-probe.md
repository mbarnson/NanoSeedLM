# Seed layouts against Unsloth's tensors on one layer of Qwen3.8-Flash-Next (2026-11-04)

**Result.** On layer 20, 3-bit seeds (C = 12, P = 4) have a weighted relative error of 0.213 on `gate_proj` and
`up_proj`. Unsloth's IQ3_XXS (3.06 bits) has 0.184 on the same weights, so gate G6 of the RAM-size plan fails. At 4.5
bits, P4 seeds have 0.079 against 0.070 for Q4_K. So plain seeds need about 0.2 bits more for each weight than these
formats for the same weight error.

**Status.** Current. One layer, 8 of 512 experts, weight error only. KL divergence against BF16 is not measured yet.

## Setup

- Machine: HF Jobs `cpu-performance` (32 vCPUs), jobs `6aebee83c78863f9c6331be8` (seeds) and
  `6aebefa9c78863f9c6331c4b` (Unsloth). No GPU.
- Weights: layer 20 of Qwen/Qwen3.8-Flash-Next at `de4b8e4`, BF16. Experts 0, 64, 128 ... 448.
- Weighting: each squared error times the input channel's energy from `imatrix_unsloth.gguf_file`, with the shrinkage
  prior n0 = 64, as in the seed search. The error of an expert is sqrt(sum h (w - w')^2 / sum h w^2). The tables give
  the mean over the 8 experts.
- Seeds: `nslm-moe --probe 20,PROJ,8,512`, the reference search of `nslm/searchc.h` with the full budget of 65,535
  seeds, on 512 sampled blocks of each expert.
- Unsloth: the layer-20 expert tensors of four UD packs of the V3.1 release. `tools/gguf_experts.py` dequantizes them
  with the gguf-py of llama.cpp `5ce06fc`. `--probe 20,PROJ,8,0 --probe-ref FILE` measures all their weights.

## Results

Seed layouts:

| Layout | Bits for each weight | `gate_proj` | `up_proj` | `down_proj` |
|---|---|---|---|---|
| P3 (C = 8, P = 3) | 4.00 | 0.1164 | 0.1158 | 0.1151 |
| P4 (C = 8, P = 4) | 4.50 | 0.0794 | 0.0796 | 0.0789 |
| S3 (C = 12, P = 4) | 3.00 | 0.2135 | 0.2134 | 0.2103 |
| S3 (C = 16, P = 7) | 3.00 | 0.2159 | 0.2167 | 0.2141 |

Unsloth's layer-20 tensors (type and bits for each weight):

| Pack | `gate_proj` and `up_proj` | Error | `down_proj` | Error |
|---|---|---|---|---|
| UD-IQ3_XXS | IQ3_XXS, 3.06 | 0.1843 / 0.1845 | IQ4_NL, 4.50 | 0.0749 |
| UD-Q3_K_XL | IQ3_XXS, 3.06 | 0.1843 / 0.1845 | Q5_0, 5.50 | 0.0407 |
| UD-IQ4_XS | IQ3_XXS, 3.06 | 0.1843 / 0.1845 | Q5_0, 5.50 | 0.0407 |
| UD-Q4_K_XL | Q4_K, 4.50 | 0.0702 / 0.0702 | Q5_1, 6.00 | 0.0334 |

## Notes

- The P3 and P4 errors agree with the full search's means over all layers (0.112 and 0.077), so the sample is large
  enough.
- At the same bits, the seed error is higher: 16% over IQ3_XXS at about 3 bits, 13% over Q4_K and 5% over IQ4_NL at
  4.5 bits. From P3 to P4, the error falls by a factor of 0.68 for 0.5 bits. So P4 needs about 0.16 bits more to reach
  Q4_K's error.
- Unsloth gives `down_proj` 1.5 to 3 bits more than `gate_proj` and `up_proj`. At layer 20, UD-IQ4_XS and UD-Q3_K_XL
  hold the same tensors. They differ in other layers.
- The two 3-bit layouts are almost equal. C = 12 needs a padded last block (2,560 and 640 are not multiples of 12).
- Weight error is not KL divergence: the gates of the plan use KLD against BF16 (step T1).
