# Make a model folder

This guide makes a seed model folder from the original K2-Horizon-MoVA-36B-A4B. `SNAPSHOT` is the local Hugging Face
snapshot of `IFM/K2-Horizon-MoVA-36B-A4B`. The [prepared models](../reference/models.md) are ready to use, so you do
not need this guide to run a model.

## GQA model with seeded experts

1. Collect activation statistics, from about 160,000 tokens of plain text:

   ```bash
   OMLX_K2_MODEL=/path/to/omlx/patches/k2_horizon/k2_horizon_model.py python tools/mova_capture.py --text calib.txt --out out/actsq_moe.bin
   ```

2. Search the seeds. This step takes about 3.4 hours on one M4 Max. The CUDA build runs the same search.

   ```bash
   out/bin/nslm-moe --model SNAPSHOT --act out/actsq_moe.bin --out out/blocks --scope gud --p4 --workers 2
   ```

3. Pack the folder. The packer writes safetensors shards of at most 4.5 GB, the index, `config.json`, the tokenizer
   files and the MLX loader.

   ```bash
   out/bin/nslm-mova-pack --model SNAPSHOT --config p4mx --blk4 out/blocks --q4 v --out MODEL_DIR
   ```

## Qwen3.8-Flash-Next experts

The search reads the fused expert tensors of the BF16 checkpoint. The Unsloth imatrix gives the activation weights, so
no capture run is necessary.

```bash
out/bin/nslm-moe --model SNAPSHOT --imatrix imatrix_unsloth.gguf_file --out out/blocks --scope gud --p4
```

On HF Jobs, `tools/hf_seedbank.sh` takes `IMATRIX=unsloth/Qwen3.8-Flash-Next-GGUF:imatrix_unsloth.gguf_file` and
`ONLY=mlp.experts.`. With `ONLY`, the job downloads only the shards that hold experts.

The release searches used these commands. `./src` is a copy of the repository (`git archive HEAD`). The bucket keeps
the block files. `JOBS=moe3` gives P = 3, and `JOBS=moe4` gives P = 4.

```bash
hf jobs run --detach --flavor rtx-pro-6000x4 --timeout 95m -v ./src:/src -v hf://buckets/USER/BUCKET:/out \
    -e SRC=/src -e MODEL=Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540 -e ONLY=mlp.experts. \
    -e IMATRIX=unsloth/Qwen3.8-Flash-Next-GGUF:imatrix_unsloth.gguf_file -e OUT=/out/p4 -e JOBS=moe4 -e WORKERS=2 \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /src/tools/hf_seedbank.sh
```

Layer 48 is the MTP head (`mtp.layers.0`). The imatrix has no statistics for it, so its search is unweighted.
`--layers 48-48` searches only the head. It needs the shards that hold `mtp.layers.0.mlp.experts`.

To pack, start from an MLX folder of the model (`mlx-community/Qwen3.8-Flash-Next-4bit-mtp`). The packer copies its
tensors and replaces the routed experts with seeds. `--blk` takes P = 3 blocks, and `--blk4` takes P = 4 blocks.

```bash
out/bin/nslm-qwen4-pack --base MLX_DIR --blk4 out/blocks --out MODEL_DIR
```

`L48_*.blk` block files seed the MTP head too. A seed pack can be the base: then only the layers with block files change.
For example, `--base P3_DIR --blk out/head` gives the P3 pack with a P3 head.

On HF Jobs, `tools/hf_qwen4_pack.sh` packs on a CPU node, from and to mounted buckets:

```bash
hf jobs run --detach --flavor cpu-xl --timeout 2h -v ./src:/src -v hf://buckets/USER/BUCKET:/b \
    -e SRC=/src -e BASE=mlx-community/Qwen3.8-Flash-Next-4bit-mtp@825df623302aeebc434f44ca9122827b04f9a0f1 \
    -e BLK=/b/p4/experts-p4 -e TIER=4 -e OUT=/b/pack-p4 python:3.12 bash /src/tools/hf_qwen4_pack.sh
```

From Git Bash on Windows, set `MSYS_NO_PATHCONV=1` first. Otherwise Git Bash changes the job's paths (`/b/...` to
`B:/...`) and the job stops at once.

## Qwen3.8-Flash-Next trellis experts

`nslm-moe --codec tq` writes trellis-coded experts as bank files, one for each layer and projection
(`L{layer}_{proj}.blkt`). It uses the GPU encoder (CUDA or Metal) when there is one, else the CPU. All three write the
same files. Read [Trellis-coded experts](../concepts/trellis.md).

```bash
out/bin/nslm-moe --model SNAPSHOT --imatrix imatrix_unsloth.gguf_file --out out/tq3 --scope gud --codec tq --tq-k 3 --tq-l 12
```

| Machine | One K at L = 12, all 121 G expert weights |
|---|---|
| 8 RTX PRO 6000 (HF Jobs, `tools/hf_seedbank.sh` with `JOBS=tq3.12`) | About 5 minutes, plus the download |
| M4 Max GPU (Metal) | About 1.5 to 2.6 hours |
| M4 Max CPU, 12 threads | About 10 hours |

The search is resumable: it skips a bank file that is complete.

A pack is a seed pack's folder, whole, with the bank files beside its shards. The engine opens the folder, then puts
each bank in place of the expert tensor of the same name. Hard links save the disk space.

```bash
mkdir TQ3_DIR && ln P3_DIR/* TQ3_DIR/ && cp out/tq3/*.blkt TQ3_DIR/
```

A mix takes each projection's banks from another K. For example, `gate_proj` and `up_proj` from K = 3 and `down_proj`
from K = 4 give 3.46 bits for each expert weight.

## MLA projections

An MLA model folder also has `model-mla-delta.safetensors`. Read [The model folder format](../reference/model-folder.md).
`nslm-mova-pack --mla` sets the encoding of the MLA projections and the per-head maps:

| Option | Encoding |
|---|---|
| (none) | BF16 |
| `--mla q8`, `--mla q4` | Affine Q8 or Q4 |
| `--mla p4` | P = 4 seeds, 4.5 bits per weight. `v_up` stays in Q8 by default |
| `--mla-q8 NAMES` | Keep some tensors in Q8. `NAME@A-B` limits it to layers A to B. `--mla-q8 none` makes all MLA tensors seeds |

To make P = 4 seeds for the MLA projections:

1. Capture the sums of squares of each projection input over the calibration text:

   ```bash
   out/bin/nslm-mova-mlacapture --model MLA_MODEL_DIR --text calib.txt --out out/actsq_mla.bin
   ```

   `--xtx DIR` also writes the `X^T X` matrix of each input, for GPTQ.
2. Search each projection. The search does the per-head maps one head at a time.
   - Activation-weighted: `nslm-moe --scope mla --p4`. All 472 M weights take 536 s on an M4 Max.
   - GPTQ: `nslm-moe --scope mla --p4 --xtx DIR`. This takes 23 minutes on an M4 Max.
3. Pack with `nslm-mova-pack --mla p4 --mla-blk4 BLOCKS_DIR`.

Read [MLA projections in BF16, Q8, Q4 and seeds](../results/2026-10-08-mla-projections.md) for the quality of each
choice.

## See also

- [SeedLM](../concepts/seedlm.md)
- [Command-line tools](../reference/tools.md)
