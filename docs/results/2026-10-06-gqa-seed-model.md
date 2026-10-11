# GQA seed model against BF16 and Q4 (2026-10-06)

**Result.** With seeded experts, the GQA model is closer to BF16 than with affine Q4 experts of the same size. Decode
is about 18% slower. The smaller p4mx-q4v model (value experts in Q4) saves 3.8 GB more.

**Status.** Current. [Seeds against affine quantization](2026-10-10-seeds-vs-affine.md) compares seeds with
calibrated oQe affine models.

## Setup

- Machine: one M4 Max, 128 GB.
- Reference: MLX BF16 logits, held-out text and chat
  ([BF16 reference log-probs](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-bf16-ref-logprobs)).
- Speed: the C engine, `nslm-mova-bench`.

## Results

| Model | Size | KLD held-out / chat | Decode, 1K / 4K context |
|---|---|---|---|
| BF16 | 74.89 GB | reference | 35.8 / 34.0 tok/s |
| Affine Q4 experts, Q8 for the other tensors | 26.53 GB | 0.0233 / 0.0110 | 60.2 / 55.9 tok/s |
| P = 4 seed experts, Q8 for the other tensors (p4mx) | 26.53 GB | 0.0195 / 0.0095 | 49.5 / 46.5 tok/s |
| p4mx with Q4 value experts (p4mx-q4v, published) | 22.76 GB | 0.0260 / 0.0126 | 50.9 / 48.0 tok/s |

## Notes

- GPU memory in the C engine: 23.3 GB at 1K context, 23.9 GB at 4K. This fits the default GPU memory of a 32 GB Mac,
  up to about 4K tokens of context.
- Generations: four blind judges compared 80 outputs with BF16. 27 of 28 checkable answers were correct. Two outputs
  (one prompt, greedy and sampled) fell into a loop. The 26.53 GB p4mx model passed all checks.
- In oMLX on the same Mac: decode 41 tok/s, against 52 tok/s for the Q4 model. Prefill of 2048 tokens: 767 tok/s
  (Q4 model: 667 tok/s).

## Sources

- [K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v model card](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v)
