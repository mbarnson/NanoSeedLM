# CUDA and Metal measured side by side (2026-10-07)

**Result.** The CUDA engine on an RTX 4080 and the Metal engine on an M4 Max give the same quality on the same model.
The RTX 4080 decodes about 25% faster at short contexts. On a 16 GB card, a 188K-token prompt works with `--kv q8`,
but decode is slow, because most of the KV cache is in host memory.

**Status.** Current for GQA. For MLA, read [MLA kernels on Metal and CUDA](2026-10-08-mla-kernels.md).

## Setup

- CUDA: RTX 4080 (16 GB, PCIe 4.0 x16), Ryzen 7 5800X3D, 64 GB DDR4, Windows 11, CUDA 13.1.
- Metal: M4 Max, 128 GB.
- Model: [K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v).
- Speed: `nslm-mova-bench --text holmes.txt --ctx N --decode 256`.
- Quality: `nslm-mova-score` against the
  [BF16 reference log-probs](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-bf16-ref-logprobs).

## Results

| | RTX 4080 (CUDA) | M4 Max (Metal) |
|---|---|---|
| decode, 1K context | 63.9 tok/s | 50.9 tok/s |
| decode, 4K context | 59.1 tok/s | 48.0 tok/s |
| prefill, 1K / 4K prompt | 1432 / 1907 tok/s | |
| KLD against BF16, held-out / chat | 0.0260 / 0.0124 | 0.0260 / 0.0126 |
| 188K-token prompt, `--kv q8` | 572 s prefill, 1.1 tok/s decode. Passes a needle-in-a-haystack check | |

## Notes

- `NSLM_SEED_GEMM_F32=1` (exact f32 seed weights in the CUDA prompt GEMM) gave held-out KLD 0.0261 against 0.0260, and
  the same NLL. It cost about 30% of prompt speed at 4K.
- The 8-bit KV cache on CUDA: on a short text, KL against the reference went from 0.005 to 0.010.
