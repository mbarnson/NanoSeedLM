# Command-line tools

The build puts the tools in `out/bin` (macOS) or `build/bin` (Windows, Linux).

## Run

| Tool | Use |
|---|---|
| `nslm-chat` | Run one prompt. Read [Run and serve](../guides/run-and-serve.md) |
| `nslm-serve` | OpenAI-compatible HTTP server |

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

## Make models

| Tool | Use |
|---|---|
| `nslm-moe` | Seed search for experts (`--scope gud`) and MLA projections (`--scope mla`) |
| `nslm-mova-mlacapture` | Activation statistics and `X^T X` of the MLA projection inputs |
| `nslm-mova-pack` | Write a model folder |

Read [Make a model folder](../guides/make-a-model.md).

## Python tools (`tools/`)

| Tool | Use |
|---|---|
| `nanoseedlm_k2.py` | The MLX loader |
| `mova_capture.py` | Activation statistics for the expert search, through MLX |
| `mova_score.py` | Scores in MLX |
| `mova_export.py` | Inputs for `nslm-mova-score`, `nslm-mova-gen`, `nslm-mova-bench` and route studies |
| `tokenizer_golden.py`, `template_golden.py`, `mova_affine_golden.py` | Golden files for the tests |
| `serve_bench.py` | Server speed. It waits for an idle GPU and records machine state |
