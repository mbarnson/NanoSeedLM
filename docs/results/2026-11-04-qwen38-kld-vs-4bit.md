# The Qwen3.8-Flash-Next seed packs against the 4-bit MLX model (2026-11-04)

**Result.** Against mlx-community's 4-bit model on 1,024 tokens of wikitext-2, the P4 pack has a mean KL divergence of
0.177 and the P3 pack 0.237. The median is 0.013 and 0.019, so a few tokens give most of the mean. The top-1 token
agrees on 91.9% (P4) and 88.2% (P3) of the tokens. The perplexity rises from 2.396 to 2.500 (P4) and 2.480 (P3).

**Status.** Current. The reference is the 4-bit model, not BF16. One text of 1,024 tokens is a small sample: it cannot
order P4 and P3 by perplexity.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000, 96 GB), `tools/hf_qwen4_kld.sh`, job `6aebad6bc78863f9c632ffcd`.
- Reference: [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp)
  at revision `825df62` (affine Q4, group 32). The packs keep all of its tensors except the routed experts.
- Models: the P4 pack (105.7 GB) and the P3 pack (98.2 GB). Read
  [The seed search over the Qwen3.8-Flash-Next experts](2026-11-04-qwen38-seed-search.md).
- Text: the first 1,024 tokens of the wikitext-2 raw test split, with no chat template.
- Engine: the CUDA engine (`nslm-qwen4-kld`), the same kernels for the reference and the packs, experts in host
  memory with a 48 GB VRAM cache. KL(reference || model) is computed in double over the full vocabulary.

```bash
hf jobs run --flavor rtx-pro-6000 --timeout 90m -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-364b9ad \
    -e REF=mlx-community/Qwen3.8-Flash-Next-4bit-mtp@825df623302aeebc434f44ca9122827b04f9a0f1 \
    -e "PACKS=/b/pack-p4 /b/pack-p3" -e "KLD_ARGS=--expert-cache-gb 48" \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-364b9ad/tools/hf_qwen4_kld.sh
```

## Results

| Model | Expert bits per weight | Mean KL | Median KL | 99th percentile KL | Top-1 agreement | Perplexity |
|---|---|---|---|---|---|---|
| 4-bit MLX model (reference) | 5.0 | 0 | 0 | 0 | 100% | 2.396 |
| P4 pack | 4.5 | 0.177 | 0.013 | 1.85 | 91.9% | 2.500 |
| P3 pack | 4.0 | 0.237 | 0.019 | 2.48 | 88.2% | 2.480 |

## Notes

- The divergence is between two quantizations. KL divergence against BF16 is the next measurement. It needs the BF16
  checkpoint (360 GB) in a form that the engine reads.
- The engine rounds the matrix-vector outputs to BF16 for both models, so this rounding is in the reference too.
- The run also shows that the engine runs the affine 4-bit model: its experts are affine Q4 with group 32.
- Each model took 12.6 to 15.6 s for the 1,024 tokens, in rows of 8.
