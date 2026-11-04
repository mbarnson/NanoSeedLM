# The seed search over the Qwen3.8-Flash-Next experts (2026-11-04)

**Result.** HF Jobs searched all 24,576 routed experts of Qwen3.8-Flash-Next (120.8 B weights) on 4 RTX PRO 6000
GPUs. The search took 66 min at P = 4 and 83 min at P = 3, and the two searches cost about $28. The mean
activation-weighted relative error is 0.077 at P = 4 and 0.112 at P = 3. The P = 4 pack is 105.7 GB and the P = 3 pack is 98.2 GB, against
113.3 GB for the 4-bit MLX model.

**Status.** Current. These are weight errors. KL divergence against the BF16 model is not measured yet.

## Setup

- Machine: HF Jobs `rtx-pro-6000x4` (4 NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each), image
  `nvidia/cuda:12.8.1-devel-ubuntu24.04`, 2 workers on each GPU.
- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), BF16, revision `de4b8e4`. Only
  the shards with `mlp.experts.` tensors (240 GB) were downloaded, in 76 s.
- Experts: 48 layers of 512 experts. `gate_proj` and `up_proj` are 640 x 2560 (the halves of `gate_up_proj`), and
  `down_proj` is 2560 x 640.
- Activation weights: `imatrix_unsloth.gguf_file` of
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (45 chunks of 18,432
  tokens), blended with the layer mean by the `n0` = 64 prior.
- Search: `nslm-moe` through `tools/hf_seedbank.sh`, all 65,535 seeds for each block. Commands:

```bash
hf jobs run --detach --flavor rtx-pro-6000x4 --timeout 95m --name nslm-qwen38-p4-experts -v ./src:/src \
    -v hf://buckets/txgsync/nslm-qwen38:/out -e SRC=/src \
    -e MODEL=Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540 -e ONLY=mlp.experts. \
    -e IMATRIX=unsloth/Qwen3.8-Flash-Next-GGUF:imatrix_unsloth.gguf_file -e OUT=/out/p4 -e JOBS=moe4 -e WORKERS=2 \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /src/tools/hf_seedbank.sh
```

The P = 3 search is the same command with `OUT=/out/p3` and `JOBS=moe3`. The packs come from `tools/hf_qwen4_pack.sh`
on `cpu-xl`, with the base
[mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp) at
revision `825df62`.

## Results

The relative error is `|w - w_hat| / |w|` for each expert. The tables give its mean over the experts and the 48
layers. The activation-weighted error weights each squared error by the input channel's `h`.

| Projection | P = 4 error | P = 4 weighted | P = 3 error | P = 3 weighted |
|---|---|---|---|---|
| `gate_proj` | 0.0862 | 0.0770 | 0.1261 | 0.1122 |
| `up_proj` | 0.0862 | 0.0772 | 0.1260 | 0.1125 |
| `down_proj` | 0.0869 | 0.0759 | 0.1272 | 0.1106 |

| Tier | Bits per expert weight | Search, wall time | GPU time | Cost | Pack | Pack time | Pack cost |
|---|---|---|---|---|---|---|---|
| P = 4 | 4.5 | 66 min | 8.3 GPU-hours | About $12.50 | 105.7 GB | 20 min | About $0.35 |
| P = 3 | 4.0 | 83 min | 10.6 GPU-hours | About $15.60 | 98.2 GB | 21 min | About $0.35 |

The activation-weighted error of a layer is lowest in layer 0 (0.062 at P = 4, 0.089 at P = 3). It is highest in
layer 27 (0.080 and 0.116).

## Notes

- The weighted error is lower than the plain error. The search spends its accuracy on the channels with large inputs.
- The P = 3 search took 26% longer than P = 4, at 11% less storage per weight.
- The cost is the job's run time at $11 per hour, with the model download and the build. The one-layer probe of
  [the plan](../plans/2026-11-04-qwen38-flash-next.md) estimated about $14 for each tier.
- The 4-bit MLX model stores its experts at 5.0 bits per weight (4-bit codes with a BF16 scale and bias for each 32).
