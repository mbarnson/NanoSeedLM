# SeedLM

SeedLM stores a block of weights as the seed of a pseudo-random generator and a few coefficients. At run time, the
GPU runs the generator and makes the weights again. The model file is smaller, and the GPU does more arithmetic for
each weight. NanoSeedLM applies SeedLM to K2-Horizon models on Metal, CUDA and MLX.

## Key numbers

| Format | Block | Bits per weight | Where we use it |
|---|---|---|---|
| P = 4 (`SEED4P4`) | 8 weights: 16-bit seed, 4 int4 coefficients, 4-bit exponent code | 4.5 | Routed experts, MLA projections, most tensors of MLA-P4 |
| P = 3 (`SEED4`) | 8 weights: 16-bit seed, 3 int4 coefficients and a 4-bit exponent code in 16 bits | 4.0 | MLA-P3, K2-Horizon-0.9B |

Each tensor also has one 32-bit exponent bias.

Quality, in general terms: at about the same size, P = 4 seeds come a little closer to the original model than a
calibrated 4-bit affine quantization. P = 3 seeds are smaller and less accurate. Decode with seeds is slower than
decode with an affine model of the same size. Read [Seeds against affine quantization](../results/2026-10-10-seeds-vs-affine.md).

## How it works

### The generator

A 16-bit linear feedback shift register (LFSR) with taps 0, 1, 3 and 12 makes a sequence of states from a seed. For a
block of 8 weights with P coefficients, weight `c` is:

```
S[c][p] = state(P*c + p + 1) - 32768
w_c     = bf16(R32 * 2^(exp_bias + code) * sum_p S[c][p] * q_p),   R32 = fl(1/32767)
```

`q_p` are the int4 coefficients, `code` is the 4-bit exponent code of the block, and `exp_bias` is the exponent bias
of the tensor.

### The search

For each block, the search tries all 65,535 nonzero seeds. For each seed, it fits the coefficients and the exponent
code, and it keeps the seed with the smallest error. There are two error measures:

- **Activation-weighted (AW).** The error of each weight is weighted by the mean square of its input activation. The
  activations come from about 160,000 tokens of calibration text.
- **GPTQ.** The search goes through the columns in groups, in order. It feeds the error of each group forward to the
  columns that it did not search yet (`nslm/gptq4.h`). It needs the `X^T X` matrix of each input.

The search runs on Metal and on CUDA. On one M4 Max, the search over all routed experts of K2-Horizon-MoVA takes about
3.4 hours. Read [Make a model folder](../guides/make-a-model.md).

### The kernels

The SeedLM paper made the weights on an FPGA. NanoSeedLM makes them in GPU kernels:

- **Decode (one token).** The mat-vec kernel makes each block of weights in registers and multiplies it at once. No
  weight goes to memory.
- **Prompts (many tokens).** The tile GEMM kernel makes a tile of weights in threadgroup (Metal) or shared (CUDA)
  memory, then multiplies the tile.
- **MLA projections on Metal.** A prompt forward decodes the seed MLA tensors of a layer to BF16 once, not in each
  GEMM tile.
- **CUDA prompts.** The prompt GEMM rounds seed weights to BF16 for the tensor cores. `NSLM_SEED_GEMM_F32=1` keeps
  them exact in f32, as the decode mat-vec and the Metal engine do. This change moved neither KLD nor NLL against the
  BF16 reference, and it costs about 30% of prompt speed at 4K.

### Why seeds are slower

An affine quantization stores each weight as a few bits and a group scale. The kernel reads the bits and does one
multiply and one add. A seed kernel reads fewer bits, but it must run the generator and sum P products for each
weight. On Apple Silicon, decode of the GQA seed model is 14% to 21% slower than decode of the affine Q4 model, in the
C engine and in oMLX.

## Which tensors are seeds

| Model | Seeds | Affine | BF16 |
|---|---|---|---|
| GQA p4mx-q4v | Routed experts (P = 4) | Value experts (Q4), attention, shared experts, dense layers, embedding, LM head (Q8) | Routers, norms |
| MLA-P4 | All large tensors (P = 4) | | Routers, norms |
| MLA-P3 | All large tensors (P = 3 and P = 4) | | Routers, norms |
| K2-Horizon-0.9B | All weight matrices, embedding and LM head (P = 3) | | Norms |

The model cards on Hugging Face give the exact list.

## Why K2-Horizon-MoVA

K2-Horizon-MoVA is an unusual mixture-of-experts model. It has a mixture of values, gated attention and grouped
RMSNorm. It is fully open: IFM publishes the weights, the checkpoints and the training data. I chose it as the first
MoE target for SeedLM for these reasons. Also, I may later try to train the model on seeds, and not only use seeds in
place of quantization.

## Measurements

- [Seeds against affine quantization (R-02)](../results/2026-10-10-seeds-vs-affine.md)
- [GQA seed model against BF16 and Q4](../results/2026-10-06-gqa-seed-model.md)
- [MLA projections as seeds](../results/2026-10-08-mla-projections.md)

## See also

- [SeedLM](https://arxiv.org/abs/2410.10714), Shafipour et al., ICLR 2025
- [The model folder format](../reference/model-folder.md)
- [Related work](../reference/related-work.md)
