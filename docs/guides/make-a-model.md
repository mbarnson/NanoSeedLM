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
