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
```

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

To use a model in oMLX, read [The MLX loader](../guides/mlx.md). To use a model in the engine, read
[Run and serve](../guides/run-and-serve.md).

## Which model to choose

- **For the longest context on a Mac:** MLA-NanoSeedLM-P4 or MLA-NanoSeedLM-P3 in oMLX.
- **For the quality nearest to the original:** NSLM-p4mx-q4v. It keeps the original attention.
- **For a small download to test the seed format:** K2-Horizon-0.9B-NSLM-seeds-4bpw. Its quality is lower than
  IFM's Q4_K_M, and it is slower.
- **For daily use with no interest in seeds:** an affine quantization of the original model.

## Related data

- [BF16 reference log-probs](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-bf16-ref-logprobs), for
  `nslm-mova-score`.
- [Seed bank](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-seedbank), the searched seed blocks.
