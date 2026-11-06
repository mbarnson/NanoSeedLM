# Trellis-coded experts against BF16: KL in llama.cpp (2026-11-05)

**Result.** Trellis-coded experts at 4.125 bits for each weight have a mean KL of 0.0398 against BF16. That is less than
P4 seeds at 4.5 bits (0.0408), and 16% below Unsloth's curve at the same bits. At 3.125 bits the KL is 0.0765. That is
about 4% below an extrapolation of Unsloth's curve, and 4% above UD-IQ3_XXS at 3.28 bits. A Q8_0 PLE table adds no KL
that we can measure.

**Status.** Current. Step T2 of the [trellis plan](../plans/2026-11-05-trellis-experts.md), at L = 12. Gate TB passes
at both bit rates, narrowly at 3.125 bits. The PLE result is part of step T5 (gate TP).

## Setup

- Encoder: `nslm-moe --codec tq` with the CUDA encoder (`nslm/trellis.cu`). It writes the same words as the C encoder,
  bit for bit (`tests/test_trellis_gpu.c`).
  - L = 12 state bits, K = 3 and K = 4 bits for each weight.
  - Blocks of 256 weights (`gate_proj`, `up_proj`) and 320 weights (`down_proj`), each with a BF16 scale.
  - Unsloth's imatrix with n0 = 64, as the seed packs.
- Encode: HF Jobs `rtx-pro-6000x8`, one process and 2 workers a GPU (`tools/hf_seedbank.sh`, `JOBS="tq3.12 tq4.12"`).
  - Job `6aed0ef00e0338d1bb80c50e` at `ece9722` wrote 112 of the 288 files. A race in the encoder stopped it.
  - Job `6aed142c0e0338d1bb80c76c` at `7e442c7` wrote the other 176. It skipped the finished files.
  - Each job ran 12 minutes, the model's download (65 s) and the build included. One K takes 280 to 310 s a GPU.
- KL: llama.cpp against BF16 with the BF16 PLE table, as in [the packs against BF16](2026-11-04-qwen38-kld-vs-bf16.md).
  `nslm-qwen4-emulate` writes the trellis experts into the Q8_0 expert tensors of Unsloth's Q8_0 shards.
  HF Jobs `rtx-pro-6000x4`, job `6aed17e4c78863f9c633b775`, 26 minutes.
- `Q8_0-PLE`: Unsloth's Q8_0 shards with their own Q8_0 PLE table, in place of BF16's table.
- Cost: about $14 for the bench, the two encodes and the KL job.

```bash
hf jobs run --flavor rtx-pro-6000x8 -v hf://buckets/txgsync/nslm-qwen38:/out -e SRC=/out/src-7e442c7 \
    -e MODEL=Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540 -e ONLY=mlp.experts. \
    -e IMATRIX=unsloth/Qwen3.8-Flash-Next-GGUF:imatrix_unsloth.gguf_file -e OUT=/out/tq -e "JOBS=tq3.12 tq4.12" \
    -e WORKERS=2 nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /out/src-7e442c7/tools/hf_seedbank.sh
hf jobs run --flavor rtx-pro-6000x4 -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-7e442c7 -e OUT=/b/kld-tq \
    -e TQBANK=/b/tq -e "PACKS=Q8_0 Q8_0-PLE TQ3L12 TQ4L12" \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-7e442c7/tools/hf_llama_kld.sh
```

## Results

KL against BF16. The rows without a job above come from [the packs against BF16](2026-11-04-qwen38-kld-vs-bf16.md).
Q8_0 reproduced its earlier mean KL (0.0199).

| Pack | Expert GB | Bits for each expert weight | Mean KLD | 99th percentile KLD | Top-1 agreement |
|---|---|---|---|---|---|
| Q8_0 | 128.4 | 8.50 | 0.01986 | 0.228 | 95.23% |
| Q8_0-PLE (its own Q8_0 PLE table) | 128.4 | 8.50 | 0.01991 | 0.232 | 95.16% |
| UD-Q4_K_XL | 77.0 | 5.10 | 0.0304 | 0.362 | 93.97% |
| P4 (seeds) | 68.0 | 4.50 | 0.0408 | 0.484 | 93.13% |
| **Trellis K = 4** | **62.3** | **4.125** | **0.0398** | **0.479** | **93.06%** |
| UD-IQ4_XS | 59.9 | 3.97 | 0.0509 | 0.608 | 92.31% |
| P3 (seeds) | 60.4 | 4.00 | 0.0600 | 0.723 | 91.55% |
| UD-Q3_K_XL | 55.7 | 3.69 | 0.0593 | 0.697 | 91.60% |
| UD-IQ3_XXS | 49.6 | 3.28 | 0.0735 | 0.865 | 90.79% |
| **Trellis K = 3** | **47.2** | **3.125** | **0.0765** | **0.931** | **90.50%** |

The standard error of each new mean KLD is 0.0003 to 0.0008.

Gate TB. We interpolate Unsloth's curve linearly in log KL against bits, between its two nearest packs.

| Pack | Bits | Unsloth's curve at those bits | Trellis | Below the curve | Bits saved at the same KL |
|---|---|---|---|---|---|
| Trellis K = 4 | 4.125 | 0.0474 (UD-IQ4_XS to UD-Q4_K_XL) | 0.0398 | 16% | about 0.38 |
| Trellis K = 3 | 3.125 | 0.0797 (UD-IQ3_XXS to UD-Q3_K_XL, extrapolated) | 0.0765 | 4% | about 0.08 |

Weight error of the whole encode: the square root of the error energy against the weight energy, the mean of 48 layers.

| K | Bits (gate and up, down) | gate | up | down | Imatrix-weighted, all three |
|---|---|---|---|---|---|
| 3 | 3.112, 3.150 | 0.146 | 0.145 | 0.147 | 0.129 to 0.131 |
| 4 | 4.112, 4.150 | 0.078 | 0.077 | 0.078 | 0.069 to 0.071 |

## Notes

- The trellis at 4.125 bits needs 1.9 GB more than P3 and has 34% less KL (0.0398 against 0.0600). On the RTX 4080
  its expert matvec is 1.5 times as fast as P3's in VRAM, and it moves 3% more bytes. On the M4 Max the matvec is as
  fast as P3's. Those are kernel benches, not decode in the engine (step T3).
- The gain at 3.125 bits is small. Unsloth's dynamic packs give more bits to some layers and fewer to others. Our
  trellis packs give every expert tensor the same K. A mix by layer or by use is the next step (step T4).
- The curve below 3.28 bits is an extrapolation. Unsloth's 2-bit packs, measured the same way, would fix it.
- L = 16 lowers the imatrix-weighted error energy by about 12% at K = 3 on layer 20 (square roots 0.126 against
  0.134). Its KL is not measured. The encode takes about an hour on `rtx-pro-6000x8`.
- PLE: Unsloth's Q8_0 table changes the mean KL by 0.00005, inside the standard error. In our engine, on 1,024 tokens
  of prompt, the same table costs 0.046. A BF16-level change of the PLE values gives 0.044 there, so that number is
  the engine measure's floor, not the table. That measure scores every position, the first ones with little context.
- The encode had a race. Two workers on one GPU set the kernel's shared-memory limit to their own sizes, and a T = 256
  call cut a T = 320 launch short. `a8ec03f` adds a test that fails on it. `7e442c7` sets the device's maximum less
  the kernel's static arrays, one value for every thread.
- The emulation rounds the decoded trellis weights to Q8_0. That adds a little error, as for the seed packs.
