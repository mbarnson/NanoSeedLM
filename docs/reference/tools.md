# Command-line tools

The build puts the tools in `out/bin` (macOS) or `build/bin` (Windows, Linux).

## Run

| Tool | Use |
|---|---|
| `nslm-chat` | Run one prompt (K2-Horizon-MoVA or Qwen3.8-Flash-Next). Read [Run and serve](../guides/run-and-serve.md) |
| `nslm-serve` | OpenAI-compatible HTTP server (K2-Horizon-MoVA or Qwen3.8-Flash-Next) |

## Measure

| Tool | Use |
|---|---|
| `nslm-mova-bench` | Prompt and decode speed, memory and machine state |
| `nslm-mova-score` | Logits for KLD against a reference |
| `nslm-mova-gen` | Greedy and sampled outputs for a set of prompts |
| `nslm-mova-plcheck` | Prompt-lookup decode against plain decode. The two must be identical |
| `nslm-mova-kbench` | Speed of one kernel at the MoVA shapes. There is one program for each GPU API. The Metal program also selects Metal kernel variants |
| `nslm-mova-refcheck` | The GPU engine against a C reference forward on the CPU (KL, argmax, router choices) |
| `nslm-mova-routes` | Router choices for a prompt and a generation, for expert-cache studies |
| `nslm-qwen4-gen` | Greedy generation of one prompt through the Qwen3.8-Flash-Next CUDA engine, with prompt and decode speed |
| `nslm-qwen4-kld` | KL divergence, top-1 agreement and perplexity of a Qwen3.8-Flash-Next model against a reference, on one text. `--decode N`: the last N rows through the decode kernels. `--tail-skip R`, `--cpu-tail N`: the model's run with the tail skip or the CPU tail |
| `nslm-handoff-bench` | CUDA: the time of a GPU-host hand-off inside a CUDA graph, 48 a forward (plan step D4). `--work US`: the host work of each |

## Make models

| Tool | Use |
|---|---|
| `nslm-moe` | Seed search for experts (`--scope gud`) and MLA projections (`--scope mla`). Qwen3.8-Flash-Next experts take `--imatrix` |
| `nslm-mova-mlacapture` | Activation statistics and `X^T X` of the MLA projection inputs |
| `nslm-mova-pack` | Write a model folder |
| `nslm-qwen4-pack` | Write a Qwen3.8-Flash-Next MLX folder with seed experts, from an MLX base folder |
| `nslm-qwen4-emulate` | Write seed experts into Unsloth's Q8_0 GGUF shards in place, so that llama.cpp runs them. A map takes each expert from one of several packs |

Read [Make a model folder](../guides/make-a-model.md).

## Python tools (`tools/`)

| Tool | Use |
|---|---|
| `nanoseedlm_k2.py` | The MLX loader for K2-Horizon |
| `nanoseedlm_qwen4.py` | The MLX loader for Qwen3.8-Flash-Next (mlx-vlm) |
| `nanoseedlm_seeds.py` | The seed kernels and modules that the MLX loaders share |
| `qwen4_golden.py` | A tiny random Qwen3.8-Flash-Next model and its mlx-vlm logits, for `tests/test_qwen4_ref.c` |
| `qwen4_tiny_attn.py` | The tiny model with strong attention (`tests/data/qwen4_tiny_attn`), so engine tests see attention bugs |
| `qwen4_template_golden.py` | Qwen3.8-Flash-Next's chat template rendered by transformers, for `tests/test_qwen4_template.c` |
| `qwen4_cache_sim.py` | Replays a routing trace (`Q4E_ROUTES`) through expert cache policies: LRU, LFU, a hot set and the optimum |
| `hf_cuda_test.sh` | Build and run CUDA tests on an HF Jobs GPU node |
| `hf_qwen4_gen.sh` | Run a Qwen3.8-Flash-Next pack through `nslm-qwen4-gen` on an HF Jobs GPU node |
| `hf_qwen4_kld.sh` | Score packs with `nslm-qwen4-kld` against a reference model on wikitext-2, on an HF Jobs GPU node |
| `hf_llama_kld.sh` | Score Unsloth packs and emulated seed packs against BF16 in llama.cpp, with the BF16 PLE table |
| `hf_qwen4_routes.sh` | Record routing traces of a pack on chat and code text, on an HF Jobs GPU node |
| `qwen4_tier_map.py` | Write a per-expert tier map from routing traces (most used experts), at random, or by projection |
| `qwen4_tier_sim.py` | Simulate a high tier that follows the routing: its share of router choices and its loads |
| `mova_capture.py` | Activation statistics for the expert search, through MLX |
| `mova_score.py` | Scores in MLX |
| `mova_export.py` | Inputs for `nslm-mova-score`, `nslm-mova-gen`, `nslm-mova-bench` and route studies |
| `tokenizer_golden.py`, `template_golden.py`, `mova_affine_golden.py` | Golden files for the tests |
| `serve_bench.py` | Server speed. It waits for an idle GPU and records machine state |
