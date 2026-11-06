# Search objectives and KL on SmolLM2-135M: seeds and the trellis (2026-11-06)

**Result.** On one MLP layer of SmolLM2-135M, the weights of the squared error change the KL more than the seed width
does. For the trellis, the packer's imatrix weights h give the lowest KL when the input skew is mild, as in Qwen's
experts. When 1% of the input channels hold 98% of the input energy, the same weights give 7 to 12 times the KL of no
weights. Weights h^1/2 give the lowest KL there, and they are within 14% of the best at mild skew.

**Status.** Current. One small dense model, one layer, one text. Not measured on Qwen3.8-Flash-Next yet. Read
[the plan for skew-aware weights](../plans/2026-11-06-skew-aware-importance.md).

## Setup

- Machine: a cloud container with 4 CPU cores and 15 GB of RAM, and no GPU. The load average was 0.1 before the first
  run. The runs are deterministic, and the page gives no speed results.
- Model: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) at `93efa2f`, BF16. It is a
  Llama-type model with a SwiGLU MLP. A NumPy forward computes in FP32.
- Text: wikitext-2 raw from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) at `b08601e`.
  KL and perplexity use 16 windows of 512 tokens of the test split (8,192 tokens). X^T X uses 64 windows of the train
  split (32,768 tokens).
- Reference: the BF16 model, perplexity 24.319.
- Quantized: the MLP of layer 15, in order `gate_proj`, `up_proj`, then `down_proj`. The X^T X of `down_proj` comes
  from the quantized `gate_proj` and `up_proj`. The rows "down_proj only" quantize `down_proj` alone.
- Output error: tr(dW H dW^T) / tr(W H W^T), with H = X^T X of the calibration text, as the mean over the quantized
  matrices.
- Weights of the squared error, with h = diag(X^T X): none, h (the packer's choice for the imatrix), h^1/2 and h^1/4.
  GPTQ uses the full X^T X, with error feedback by groups of 8 columns as `nslm/gptq4.h`. Its damping is 0.01 or 1
  times the mean diagonal.
- Seeds: the probe search `tools/seed_kl_probe/seedlib.c`, P = 3 or 4, K-bit LFSR, every state or every 4th state.
- Trellis: `nslm_tq_encode_row` of `nslm/lib_trellis.c` at `475c02d`, L = 12. The block length T follows the packer:
  576 for `gate_proj` and `up_proj` (so one block for each row) and 256 for `down_proj`. The bits for each weight are
  K + (L - K + 16) / T: 4.04 and 4.09 at K = 4, 3.04 and 3.10 at K = 3.
- Q8_0 and Q4_0: ggml's rounding in blocks of 32 (8.5 and 4.5 bits).
- Qwen's skew: Unsloth's imatrix (`imatrix_unsloth.gguf_file`), blended with the n0 = 64 prior as `nslm_imatrix_h`.

```bash
cd tools/seed_kl_probe
clang -O3 -march=native -ffp-contract=off -fPIC -shared -pthread seedlib.c -o libseed.so -lm
clang -O3 -march=native -ffp-contract=off -fPIC -shared -pthread -I../../nslm ../../nslm/lib_trellis.c tqlib.c -o libtq.so -lm
python3 harness.py SMOL 15 q8_0 q4_0 plain:3:16 diag:3:16 diag25:3:16 gptq:3:16
python3 harness.py SMOL 15 tq:4:12:1 tq:4:12:0.5 tq:4:12:0.25 tq:4:12:0
MATS=down python3 harness.py SMOL 15 tq:4:12:1 tq:4:12:0.5 tq:4:12:0
python3 -I imat_skew.py imatrix_unsloth.gguf
```

## Results

The input skew: the share of trace(X^T X) in the top 1% of the input channels, and the largest diagonal entry over the
median.

| Input | Top 1% of channels | Largest / median |
|---|---|---|
| SmolLM2 layer 15, `gate_proj` and `up_proj` | 98.2% | 86,458 |
| The same, without the first 4 tokens of each window | 55.7% | 921 |
| SmolLM2 layer 15, `down_proj` | 11.0% | 51 |
| Qwen experts, `gate_proj` and `up_proj`, layers 0, 20, 40 (median expert) | 10.1%, 3.7%, 8.4% | 37, 10, 47 |
| Qwen experts, `down_proj`, layers 0, 20, 40 (median expert) | 14.4%, 4.0%, 6.7% | 59, 7, 15 |
| Qwen experts, `down_proj`, layers 0, 20, 40 (worst expert) | 52.4%, 28.8%, 40.7% | 1,062, 132, 315 |

KL of the trellis by weights. The MLP columns have the extreme skew. The `down_proj` columns have a skew like Qwen's.

| Weights | MLP, K = 4 | MLP, K = 3 | `down_proj` only, K = 4 | `down_proj` only, K = 3 |
|---|---|---|---|---|
| h (the packer) | 0.0356 | 0.0422 | 0.000408 | 0.00110 |
| h^1/2 | 0.00173 | 0.00386 | 0.000465 | 0.00115 |
| h^1/4 | 0.00179 | 0.00404 | 0.000546 | |
| None | 0.00304 | 0.00596 | 0.000697 | 0.00155 |
| Q4_0 (4.5 bits), for comparison | 0.00183 | | 0.000521 | |
| Q8_0 (8.5 bits), for comparison | 0.000007 | | | |

The perplexity of the trellis MLP with weights h is 25.40 at K = 4 and 25.53 at K = 3. All other trellis rows are 24.30
to 24.49.

Seeds, P = 3 and K = 16, by the objective of the search:

| Objective | Output error | KL | Perplexity |
|---|---|---|---|
| No weights | 0.137 | 0.00291 | 24.340 |
| Weights h | 0.0097 | 0.00378 | 24.404 |
| Weights h^1/2 | 0.0080 | 0.00236 | 24.349 |
| GPTQ, damping 0.01 | 0.0056 | 0.00307 | 24.379 |
| GPTQ, damping 1 | 0.0069 | 0.00249 | 24.392 |
| Weights h, X^T X without the first 4 tokens | | 0.00404 | 24.454 |
| GPTQ, X^T X without the first 4 tokens | | 0.00419 | 24.447 |

Seeds by width, with weights h^1/2:

| Format | Bits per weight | KL |
|---|---|---|
| P = 3, K = 16 | 4.0 | 0.00236 |
| P = 3, K = 16, every 4th state | 4.0 | 0.00185 |
| P = 3, K = 18 | 4.25 | 0.00141 |
| P = 3, K = 18, every 4th state | 4.25 | 0.00125 |
| P = 3, K = 20 | 4.5 | 0.00097 |
| P = 4, K = 16 | 4.5 | 0.00094 |

## Notes

- The lowest output error (GPTQ) does not give the lowest KL. The output error counts all output directions the same,
  and the layers after them do not. A KL objective needs the sensitivity of the later layers, for example a Fisher
  weight. The probe has no autograd for that yet.
- X^T X without the first 4 tokens makes the KL worse. So the errors at those tokens matter. The full weights h give
  them too much weight.
- A trellis block has one scale, and here T = 576 makes the scale cover the row. When one channel dominates h, the
  weighted least-squares scale fits that channel, and the other weights get the wrong scale. A seed block has an
  exponent for each 8 weights, so the same weights do less harm to seeds. Qwen's trellis blocks are shorter (256 and
  320), and most Qwen experts have a mild skew.
- In the mild case, the stronger the weights, the lower the KL. So the packer's choice suits most Qwen experts. A few
  Qwen `down_proj` experts approach the extreme case, and h^1/2 may help them.
- KL falls by about 0.80 for each seed bit from K = 16 to 20. At that rate, the KL of Q8_0 (0.000007) needs about 22
  more bits: K = 42, 7.3 bits for each weight, and 2^26 times the search. This is an extrapolation.
- The probe search had a precision fault. The expanded error (|x|^2 - 2 sc q.b) + sc^2 |L^T q|^2 lost precision for a
  nearly singular seed basis with a large scale. With 2^20 seeds and weights h, it chose exponents up to +11 and gave
  a KL of 0.0095. The probe now confirms each improvement with the direct residual |x - sc U q|^2. The results for
  K = 16 to 19 did not change, and the table gives the corrected K = 20. The repository's search uses the expanded
  error too. It clamps the exponent to a window of 16, which should prevent this at K = 16.
- The KL of one layer is small. The runs are deterministic and use the same tokens, so the order of the variants is
  reliable on this text. Differences under 5% may change on another text.
