# NanoSeedLM

SeedLM weights for a modern Mixture-of-Experts model, on a GPU without FPGA. NanoSeedLM implements and extends SeedLM for Apple Silicon and NVIDIA GPUs, using activation-calibrated, 4.5-bit seeded MoE experts and fused weight reconstruction in a standalone inference engine: Metal on macOS, CUDA on Windows and Linux.


## What it does

[SeedLM](https://arxiv.org/abs/2410.10714) keeps a block of weights as a 16-bit seed and some coefficients. A linear
feedback shift register (LFSR) makes the weights again from the seed at run time. The paper uses an FPGA to make the
weights. NanoSeedLM makes them in GPU kernels: Metal on Apple Silicon, CUDA on NVIDIA cards.

The engine runs [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B). Only the routed
experts use seeds.

## Why MoVA

K2-Horizon-MoVA is an atypical MoE. Between mixture of values, gated attention, grouped RMSNorm, and a totally open source and open weight approach with published checkpoints and training data, it was an appealing first target for trying SeedLM on a MoE in case I want to try training the model on seeds rather than simply using seeds as an alternative to quantization.

## Requirements

**macOS:** Apple Silicon, Xcode command-line tools with the Metal compiler.

**Windows / Linux:** an NVIDIA GPU of the Ampere generation or newer (RTX 30, 40, 50 series; sm_80+) with 12 GB or
more of VRAM, 48 GB or more of RAM (64 GB for long contexts), the CUDA Toolkit 12.4 or newer (12.8 or newer for the
RTX 50 series) and CMake 3.24 or newer with Ninja (`pip install cmake ninja`).  The build compiles for sm_80, 86, 89 and,
with CUDA 12.8 or newer, sm_120; other GPUs (Hopper, datacenter Blackwell) run the compute_89 PTX, which the driver
compiles on first use.  `-DCMAKE_CUDA_ARCHITECTURES=native` builds for the installed GPU only.
- Windows: Visual Studio 2022 Build Tools (C++). ICU comes with Windows 10 1903 and later.
- Linux: GCC or Clang, and ICU (`libicu-dev`).

Python tools (optional; Python 3):
- Any platform, with `tokenizers` and `numpy`: `tools/tokenizer_golden.py` (regenerates `tests/data/tokenizer_golden.json`
  from a model's `tokenizer.json`) and `tools/mova_export.py windows | prompts | bench` (inputs for `nslm-mova-score`,
  `nslm-mova-gen` and `nslm-mova-bench`); `tools/template_golden.py` needs `transformers`.
- macOS only, with `mlx`, `mlx-lm` and `huggingface_hub`: the MLX loader and the tools that run the model in MLX
  (`mova_capture.py`, `mova_score.py`, `mova_affine_golden.py`, `mova_export.py routes`).

## Build and test

macOS:

```bash
make
make test
```

Windows (from any terminal; the script finds Visual Studio):

```bat
win\build.bat
win\build.bat test
```

Linux:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build
```

The binaries are in `out/bin` (macOS) or `build/bin` (Windows, Linux).  Both builds run the same tests from the same
C sources.  `test_mova_kernels` checks every GPU kernel against a C reference, through a small backend per GPU API
(`tests/kernel_backend_cuda.c`, `tests/kernel_backend_metal.m`).  The tests that read a model folder skip without one:
`test_mova_cfg` and `test_tokenizer` (`MOVA_DIR=...`, or CMake's `-DNSLM_MODEL_DIR=...`, or set `NSLM_MODEL_DIR` before
the first `win\build.bat`).  The GPU tests skip on a machine without a GPU.  `make test` builds
`test_affine`'s MLX goldens; the CMake builds have no MLX and skip it unless the goldens exist.

If you want to test the MLX remote code, use the k2_horizon_model.py script included with the model on HuggingFace.

```
OMLX_K2_MODEL=/path/to/k2_horizon_model.py make test-mlx   # the MLX loader
```

## Run

1. Get a model folder. Use the
   [prepared model](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v), or make one (see below).
2. Run a prompt:

```bash
out/bin/nslm-chat --model MODEL_DIR "Why do tide pools matter?"          # macOS
build\bin\nslm-chat --model MODEL_DIR "Why do tide pools matter?"        # Windows
```

The folder holds `config.json`, the tokenizer files and the safetensors shards. Use `--effort low|medium|high` to set
the reasoning length. Use `--temp` and `--seed` to sample, `--prompt-file FILE` for a long prompt.

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
- `reasoning_effort` is `low`, `medium` or `high` (the default when a request names none). The model always thinks
  first; the reasoning text is in `reasoning_content`.
- Sampling defaults: `temperature` 1.0, `top_p` 0.95 (IFM). Use `temperature` 0 for greedy decode. `top_k`, `min_p`
  and `seed` are per request.
- Continuous batching: concurrent requests share the GPU. Each gets one of `--max-seqs` sequence slots of `--ctx`
  tokens (default: as many as half of the memory left after the weights holds, up to 16); more requests wait in line.
  Prompts are computed in chunks of 256 tokens between decode steps, so a new prompt does not stall the others. A
  request's tokens do not depend on what else runs: with a `seed` (or greedy) it gets the same answer alone or among
  others (on Metal, bit for bit). Decode steps run in forwards of up to 8 slots. Experimental (Metal):
  `NSLM_BATCH_GEMM=32` runs steps of 32 or more slots through the prompt GEMMs instead, which is faster at large batches
  (M4 Max, q8mx: 171 tok/s at 64 streams against about 112) but gives up the bit-for-bit equality.
- Prefix reuse: a request takes the free slot whose cache shares the longest prefix with its prompt. Repeated system
  prompts, tools and earlier turns are not computed again.
- Cold cache: a request that ends with at least `--kv-disk-min` (2048) cached tokens has its cache saved to disk in
  256-token blocks (`--kv-disk`, default `~/.cache/nslm/kv`; `--kv-disk-gb` 32, least recently used blocks go first;
  0 turns it off). A later request with the same prefix reads it back instead of computing it. Blocks are keyed by the
  tokens and the model's weights, so another model or quantization never reuses them.
- `--ctx` sets each slot's context (prompt and output). On a 32 GB Mac, 4096 fits; on a larger Mac, use a larger value,
  for example `--ctx 32768`. On a CUDA GPU, see [NVIDIA GPUs](#nvidia-gpus-cuda) for how the context is held.
- `--kv q8` keeps the KV cache in 8 bits (int8 with a scale per token and head) on either engine: half the memory, so a
  200k-token cache takes about 20 GB instead of 39 GB.  On an M4 Max it costs little quality (KLD against BF16,
  held-out / chat: 0.0267 / 0.0136 against BF16 KV's 0.0260 / 0.0126, each within about one standard error; top-1
  agreement unchanged) and about a tenth of decode speed at 4k context.

Example:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"low"}'
```

## NVIDIA GPUs (CUDA)

The CUDA engine (`engine/mova_cuda.c`, `engine/kernels_moe.cu`) implements the same engine API with the same kernels'
arithmetic and BF16 rounding points as the Metal engine; the tools, the tokenizer and the tests are the same C sources
on both.  A discrete GPU usually has less memory than the 22.8 GB model, so:

- **Experts are cached.** Every routed and value expert is in pinned host memory; VRAM holds as many as fit, and a
  GPU-side cache (least recently used, admitted after each router) brings missing experts over PCIe before they run.
  Within a conversation the cache serves about 90% of expert uses.  Results never depend on what is cached
  (`test_engine` checks this bit for bit).
- **Prompts run layer by layer** over up to 8192 tokens at a time, on tensor cores (BF16 `mma` with f32 sums), with
  the next layer's experts copied while a layer computes.
- **Decode runs as CUDA graphs** (one launch per token).
- **The KV cache** keeps as many positions as fit in VRAM and the rest in pinned host memory (each layer's host rows are
  staged in VRAM while a prompt computes it).  `--kv q8` (int8 with a scale per token and head) halves it; its rounding
  is about BF16's size (KL against the reference rises from 0.005 to 0.010 on a short text).  Contexts above roughly
  100k tokens need `--kv q8` on a 64 GB PC: a 200k-token BF16 cache alone is 39 GB.

Measured on an RTX 4080 (16 GB, PCIe 4.0 x16), Ryzen 7 5800X3D, 64 GB DDR4, Windows 11, CUDA 13.1, with the
[prepared model](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v) (`nslm-mova-bench --text
holmes.txt --ctx N --decode 256`; quality with `nslm-mova-score` on the
[BF16 reference log-probs](https://huggingface.co/datasets/txgsync/K2-Horizon-MoVA-36B-A4B-bf16-ref-logprobs)):

| | RTX 4080 (CUDA) | M4 Max (Metal) |
|---|---|---|
| decode, 1k context | 63.9 tok/s | 50.9 tok/s |
| decode, 4k context | 59.1 tok/s | 48.0 tok/s |
| prefill, 1k / 4k prompt | 1432 / 1907 tok/s | |
| KLD against BF16, held-out / chat | 0.0260 / 0.0124 | 0.0260 / 0.0126 |
| 188k-token prompt, `--kv q8` | 572 s prefill, 1.1 tok/s decode; passes a needle-in-a-haystack check | |

Past a few tens of thousands of tokens this model's 48 full-attention layers make the KV cache the limit: every token
reads all of it, and on a 16 GB card most of it then sits behind PCIe.

Environment knobs: `NSLM_VRAM_RESERVE_MB` (VRAM left free, default 512), `NSLM_EXPERT_MIN_MB` (VRAM kept for experts
when the KV cache is large, default 6144), `NSLM_KV_VRAM_MB`, `NSLM_EXPERT_VRAM_MB` (fix the shares),
`NSLM_CACHE_STATS=1` (expert cache hits at exit), `NSLM_NO_GRAPH=1`, `NSLM_NO_PREDICT=1` (no next-layer expert
prediction in decode), `NSLM_SEED_GEMM_F32=1`. By default the prefill GEMM rounds seed weights to BF16 for the tensor
cores. `NSLM_SEED_GEMM_F32=1` keeps them exact in f32, as the decode matvec and the Metal engine do. Against the BF16
reference this changed neither KLD (held-out 0.0261 vs 0.0260) nor NLL, and it costs about 30% of 4k prefill (the
tensor cores do three times the work). `NSLM_PREFILL_RANGE` sets the prompt rows computed layer by layer at a time
(default 8192, or 32768 when the KV cache spills to host memory: each range re-stages the earlier host rows).

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
| `nslm-mova-kbench` | Single-kernel speed at MoVA's shapes (one program per GPU API: the Metal one also selects Metal kernel variants) |
| `nslm-mova-refcheck` | The GPU engine against a C reference forward on the CPU (KL, argmax, router choices) |
| `nslm-mova-routes` | Router choices of a prompt and a generation (expert-cache studies) |

## MLA models (experimental)

The engine also runs K2-Horizon-MoVA converted from grouped-query attention to multi-head latent attention
(MLA, after [TransMLA](https://arxiv.org/abs/2502.07864)), for example
[K2-Horizon-MoVA-36B-A4B-MLA](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA). `config.json` carries
`mla_ranks` (the latent rank per layer, a multiple of 32, at most 1024) and `mla_rope_dim` (128); the folder adds
`model-mla-delta.safetensors` with each layer's BF16 tensors `self_attn.mla.{kv_a_x, kv_a_v, k_rope_proj, q_rope_mix,
q_lat, v_up}`. `k_proj` is unused, and the dense layers fold `v_proj` into `kv_a_x`.

Per layer the cache holds the latent `c = kv_a_x x (+ kv_a_v v)` and one 128-dim RoPE key per token: at rank 768,
84 KiB per token instead of 192 KiB. Attention is computed absorbed, as multi-query attention over the latent, then
the per-head `v_up` and the gate. On Metal, `k_mla_attn` is flash style: 16 query heads per threadgroup, 64-key tiles
staged in threadgroup memory, scores and `P c` on simdgroup matrices, online softmax, for prompts and (split-key)
decode alike; prompt rows run the per-head maps as GEMMs. On an M4 Max (q8mx) that took prefill of 2048 tokens from
21.3 s to 6.6 s (GQA: 3.2 s) and of 8192 tokens from 272 s to 64 s (GQA: 15 s); absorbed attention still does about
6x the arithmetic of GQA's. The CUDA engine keeps its one-pass latent kernel (8 query heads per block); its KV cache
places MLA's per-layer latent widths in VRAM and host memory as it does GQA's rows. Not yet: the 8-bit cache for MLA
(`--kv q8` is refused), packed (Q8 / seed) MLA tensors (the MLA projections stay BF16), and the flash kernel on CUDA.

## Make a model folder

1. Collect activation statistics (approximately 160k tokens of plain text):

```bash
OMLX_K2_MODEL=/path/to/k2_horizon_model.py python tools/mova_capture.py --text calib.txt --out out/actsq_moe.bin
```

2. Search the seeds (approximately 3.4 hours on one M4 Max; the CUDA build runs the same search):

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
| `engine/` | Engines: Metal (`mova_gpu.m`, `kernels_moe.metal`) and CUDA (`mova_cuda.c`, `kernels_moe.cu`) |
| `nslm/` | LFSR, seed search (C, Metal and CUDA), model folder reader and writer, packer, CPU reference forward |
| `harness/` | Command-line tools, server, tokenizer (ICU), chat template and tool-call parser, platform layer |
| `tests/` | One test binary for each part; the same tests on every platform (GPU kernels through `kernel_backend.h`) |
| `win/` | POSIX shims for MSVC and `build.bat` |
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
