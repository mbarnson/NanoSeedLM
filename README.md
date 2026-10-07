# NanoSeedLM

SeedLM weights for a modern Mixture-of-Experts model, on a Mac GPU without FPGA. NanoSeedLM implements and extends SeedLM for Apple Silicon, using activation-calibrated, 4.5-bit seeded MoE experts and fused weight reconstruction in a standalone Metal inference engine.


## What it does

[SeedLM](https://arxiv.org/abs/2410.10714) keeps a block of weights as a 16-bit seed and some coefficients. A linear
feedback shift register (LFSR) makes the weights again from the seed at run time. The paper uses an FPGA to make the
weights. NanoSeedLM makes them in Metal kernels on the Apple Silicon GPU.

The engine runs [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B). Only the routed
experts use seeds.

## Why MoVA

K2-Horizon-MoVA is an atypical MoE. Between mixture of values, gated attention, grouped RMSNorm, and a totally open source and open weight approach with published checkpoints and training data, it was an appealing first target for trying SeedLM on a MoE in case I want to try training the model on seeds rather than simply using seeds as an alternative to quantization.

## Requirements

- macOS on Apple Silicon. Xcode command-line tools with the Metal compiler.
- For `make test` and the search tools: Python 3 with `mlx`, `mlx-lm` and `huggingface_hub`.

## Build and test

```bash
make
make test
```

If you want to test the MLX remote code, use the k2_horizon_model.py script included with the model on HuggingFace.

```
OMLX_K2_MODEL=/path/to/k2_horizon_model.py make test-mlx   # the MLX loader
```

## Run

1. Get a model folder. Use the
   [prepared model](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v), or make one (see below).
2. Run a prompt:

```bash
out/bin/nslm-chat --model MODEL_DIR "Why do tide pools matter?"
```

The folder holds `config.json`, the tokenizer files and the safetensors shards. Use `--effort low|medium|high` to set
the reasoning length. Use `--temp` and `--seed` to sample.

## Serve (OpenAI-compatible API)

```bash
out/bin/nslm-serve --model MODEL_DIR --port 8080
```

- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`. Streaming (SSE) is supported.
- Tool calling: send `tools`. The answer has OpenAI `tool_calls`. Parallel calls are supported.
- The server renders the model's own chat template in C. It matches the Hugging Face output byte for byte
  (`tests/data/template_golden.json`).
- Template options go in `chat_template_kwargs`: `tool_presentation_format` (`markdown`, `json`) and
  `tool_call_format` (`xml`, `json`, `xml_typed`).
- `reasoning_effort` is `high` (default), `medium` or `low`. The reasoning text is in `reasoning_content`.
- Sampling defaults: `temperature` 1.0, `top_p` 0.95 (IFM). Use `temperature` 0 for greedy decode.
- The server keeps the KV cache of the previous request's common prefix. Repeated system prompts and tools are not
  computed again.
- `--ctx` sets the context (prompt and output). The default, 4096, fits a 32 GB Mac. On a larger Mac, use a larger
  value, for example `--ctx 32768`.

Example:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"low"}'
```

## MLX and oMLX

The model folder contains `nanoseedlm_k2.py`, an MLX loader with the seed kernels. `config.json` names it in
`model_file`. In oMLX, enable **Trust Remote Code** for the model. oMLX supplies the K2-Horizon model code.

Other tools:

| Tool | Use |
|---|---|
| `nslm-mova-bench` | Prefill and decode speed, memory, machine state |
| `nslm-mova-score` | Logits for KLD against a reference |
| `nslm-mova-gen` | Greedy and sampled outputs for a prompt set |
| `nslm-mova-plcheck` | Prompt-lookup decode against plain decode (must be identical) |
| `nslm-mova-kbench` | Single-kernel speed |

## Make a model folder

1. Collect activation statistics (approximately 160k tokens of plain text):

```bash
OMLX_K2_MODEL=/path/to/k2_horizon_model.py python tools/mova_capture.py --text calib.txt --out out/actsq_moe.bin
```

2. Search the seeds (approximately 3.4 hours on one M4 Max):

```bash
out/bin/nslm-moe --model SNAPSHOT --act out/actsq_moe.bin --out out/blocks --scope gud --p4 --workers 2
```

3. Pack the folder (safetensors shards of at most 4.5 GB, the index, `config.json`, the tokenizer files and the MLX
   loader):

```bash
out/bin/nslm-mova-pack --model SNAPSHOT --config p4mx --blk4 out/blocks --q4 v --out MODEL_DIR
```

`SNAPSHOT` is the local Hugging Face snapshot of IFM/K2-Horizon-MoVA-36B-A4B.

## Layout

| Folder | Contents |
|---|---|
| `engine/` | Engine (Objective-C Metal glue) and Metal kernels |
| `nslm/` | LFSR, seed search (C99 and Metal), model folder reader and writer, packer |
| `harness/` | Command-line tools, server, chat template and tool-call parser |
| `tests/` | One test binary for each part |
| `tools/` | MLX loader; Python tools for calibration and reference data |

## Model folder

A model folder is a Hugging Face folder with safetensors shards. Each tensor `NAME.weight` is stored by encoding:

| Encoding | Entries |
|---|---|
| BF16 | `NAME.weight` |
| Affine Q8, Q4 (group 64, MLX layout) | `NAME.weight` (U32), `NAME.scales`, `NAME.biases` |
| Seeds, P=4 (4.5 bits per weight) | `NAME.seeds` (U16), `NAME.coefs` (U16), `NAME.codes` (U8), `NAME.exp_bias` (I32) |

Each shard header is padded to a 16 KiB page, and entries whose size is a page multiple come first. The engine maps
those into GPU buffers without a copy. See `nslm/model_st.h`.

## Related work

- [SeedLM](https://arxiv.org/abs/2410.10714), Shafipour et al., ICLR 2025. The method.
- [Seed-Q](https://arxiv.org/abs/2609.38477). Sensitivity-aware LFSR seed quantization.
- [AWSRC](https://arxiv.org/abs/2608.23144). Seeded residual coding on top of INT4.
- [SeedLM+O](https://github.com/ropaes1/seedlm-o-results). A measured negative result on small dense models.
- [seedlm-repro](https://github.com/qiuyu-ren/seedlm-repro). A NumPy replication.
- [PRANC](https://arxiv.org/abs/2206.08464) and [NOLA](https://arxiv.org/abs/2310.02556). Networks and adapters from
  random bases.
- [QTIP](https://arxiv.org/abs/2406.11235) and
  [ik_llama.cpp trellis quants](https://github.com/ikawrakow/ik_llama.cpp/pull/113). Weights computed, not looked up.
- [Hacker News discussion of SeedLM](https://news.ycombinator.com/item?id=43599967).

## License

MIT. See `LICENSE`. Model weights have their own license.
