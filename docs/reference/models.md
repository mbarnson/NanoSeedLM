# Models

All models are on Hugging Face under [txgsync](https://huggingface.co/txgsync). Each model card has the full results,
the generation checks and the known issues.

## Lineage

```
IFM/K2-Horizon-MoVA-36B-A4B                              original (GQA)
├─ K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v                 seeded experts, 22.76 GB
└─ K2-Horizon-MoVA-36B-A4B-MLA                           TransMLA conversion + distillation heal, BF16
   ├─ K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4          seeds, 22.20 GB
   ├─ K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P3          seeds, 20.07 GB
   ├─ K2-Horizon-MoVA-36B-A4B-MLA-oQ4e                   affine (oMLX oQ), 22.89 GB
   ├─ K2-Horizon-MoVA-36B-A4B-MLA-oQ6e                   affine (oMLX oQ), 31.74 GB
   └─ K2-Horizon-MoVA-36B-A4B-MLA-oQ8e                   affine (oMLX oQ), 40.61 GB

IFM/K2-Horizon-0.9B                                      original (dense)
└─ K2-Horizon-0.9B-NSLM-seeds-4bpw                       seeds, 0.54 GB

Qwen/Qwen3.8-Flash-Next                                  original (qwen4_exp)
├─ mlx-community/Qwen3.8-Flash-Next-4bit-mtp             MLX 4-bit (group 32), with the MTP head
│  ├─ Qwen3.8-Flash-Next-NanoSeedLM-P4                   seed experts (P = 4), the rest as the 4-bit model, 105.7 GB
│  └─ Qwen3.8-Flash-Next-NanoSeedLM-P3                   seed experts (P = 3), the rest as the 4-bit model, 98.2 GB
├─ Qwen3.8-Flash-Next-NanoSeedLM-TQ4                     trellis experts (K = 4), 8-bit dense weights, 102 GB
├─ Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4                  trellis experts (K = 3, down_proj K = 4), 8-bit dense, 92 GB
└─ Qwen3.8-Flash-Next-NanoSeedLM-TQ3                     trellis experts (K = 3), 8-bit dense weights, 87 GB
```

The trellis packs code the BF16 experts. Their dense weights come from
[mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp), and
their PLE table from the 4-bit model.

The oQe models are for comparison. They let us compare seeds with a calibrated affine quantization of the same MLA
model.

## Runtimes

| Model | Loader | oMLX | Metal engine | CUDA engine |
|---|---|---|---|---|
| [NSLM-p4mx-q4v](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v) | `nanoseedlm_k2.py` | Yes | Yes | Yes |
| [MLA-NanoSeedLM-P4](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4) | `nanoseedlm_k2.py` | Yes | Yes | Not checked |
| [MLA-NanoSeedLM-P3](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P3) | `nanoseedlm_k2.py` | Yes | Yes | Not yet (P = 3 tensors) |
| [MLA](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA) (BF16) | `k2_mla.py` | Yes | Not checked | Not checked |
| [MLA-oQ4e](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-oQ4e), [oQ6e](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-oQ6e), [oQ8e](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-oQ8e) | `k2_mla.py` | Yes | No | No |
| [0.9B-NSLM-seeds-4bpw](https://huggingface.co/txgsync/K2-Horizon-0.9B-NSLM-seeds-4bpw) | `nanoseedlm_k2.py` | Yes | No | No |
| [Qwen3.8-Flash-Next-NanoSeedLM-P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4), [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | `nanoseedlm_qwen4.py` (mlx-vlm) | Not yet | Yes | Yes |
| [Qwen3.8-Flash-Next-NanoSeedLM-TQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4), [TQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4), [TQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3) | Not in the repositories yet | Not yet | Yes | Yes |

oMLX removes `model_file` from `qwen4_exp` configurations, so it does not load the Qwen3.8-Flash-Next loader yet. A
one-line fix (Matt's fork, `29ee79c`) lets it load P3 and, with `nanoseedlm_trellis.py`, TQ4.
Use mlx-vlm 0.7.12 or later with `--trust-remote-code`. To use a model in oMLX, read [The MLX loader](../guides/mlx.md).
To use a model in the engine, read [Run and serve](../guides/run-and-serve.md).

## Which model to choose

- **For the longest context on a Mac:** MLA-NanoSeedLM-P4 or MLA-NanoSeedLM-P3 in oMLX.
- **For the quality nearest to the original:** NSLM-p4mx-q4v. It keeps the original attention.
- **For a small download to test the seed format:** K2-Horizon-0.9B-NSLM-seeds-4bpw. Its quality is lower than
  IFM's Q4_K_M, and it is slower.
- **To run Qwen3.8-Flash-Next with the NanoSeedLM engines:** TQ4, TQ3-D4 or TQ3. The start script of the engine
  release picks one by the memory of the machine. Their KL against BF16 as shipped is 0.0415, 0.0638 and 0.0778
  ([the release packs as shipped](../results/2026-11-05-release-as-shipped.md)).
- **To try seed experts on Qwen3.8-Flash-Next:** Qwen3.8-Flash-Next-NanoSeedLM-P4 or -P3, in mlx-vlm or with
  the NanoSeedLM engines. Against the 4-bit MLX model, P4 agrees on the top token 91.9% of the time. As shipped, with
  4-bit MLX dense weights, the KL of P3 against BF16 is 0.182 ([packs as shipped](../results/2026-11-05-dense-as-shipped.md)).
- **For daily use with no interest in seeds:** an affine quantization of the original model.

## Related data

- [BF16 reference log-probs](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-bf16-ref-logprobs), for
  `nslm-mova-score`.
- [Seed bank](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-seedbank), the searched seed blocks.
