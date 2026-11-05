# Run and serve

`nslm-chat` runs one prompt. `nslm-serve` is an OpenAI-compatible HTTP server.

## Get a model folder

Download a model from [the model list](../reference/models.md), or make one. Read
[Make a model folder](make-a-model.md). The folder holds `config.json`, the tokenizer files and the safetensors
shards.

## Run a prompt

```bash
out/bin/nslm-chat --model MODEL_DIR "Why do tide pools matter?"          # macOS
build\bin\nslm-chat --model MODEL_DIR "Why do tide pools matter?"        # Windows
```

| Option | Use |
|---|---|
| `--effort E` | The length of the reasoning: low, medium or high (K2), low, medium or xhigh (Qwen3.8-Flash-Next) |
| `--no-think` | Qwen3.8-Flash-Next: no reasoning block |
| `--temp`, `--seed` | Sampling |
| `--prompt-file FILE` | A long prompt from a file |

## Serve

```bash
out/bin/nslm-serve --model MODEL_DIR --port 8080
```

Example request:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"low"}'
```

### API

- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`. The server supports streaming (SSE).
- **Tool calls.** Send `tools`. The answer has OpenAI `tool_calls`. The server supports parallel calls.
- **Chat template.** The server renders the chat template of the model in C. The output is the same as the Hugging
  Face output, byte for byte (`tests/data/template_golden.json`).
- **Template options** go in `chat_template_kwargs`:
  - `tool_presentation_format`: `markdown` or `json`.
  - `tool_call_format`: `xml`, `json` or `xml_typed`.
- **Reasoning.** `reasoning_effort` is `low`, `medium` or `high`. The default is `high`. The model always thinks
  first. The reasoning text is in `reasoning_content`.
- **Reasoning budget.** `reasoning_budget_tokens` sets a limit on private reasoning in chat requests.
  - At the limit, the server forces the thinking-close token, and the model continues with the answer.
  - Missing or `0` sets no limit.
  - The closing token counts toward `max_tokens`. The server clamps the budget to leave a slot for it.
  - Raw completions reject a positive budget.
- **Sampling.** The defaults are `temperature` 1.0 and `top_p` 0.95, as IFM recommends. `temperature` 0 gives greedy
  decode. `top_k`, `min_p` and `seed` are per request.

### Continuous batching

Requests that arrive at the same time share the GPU.

- Each request gets one of `--max-seqs` sequence slots of `--ctx` tokens. The default number of slots is the number
  that fits in half of the memory left after the weights, up to 16. Other requests wait in a queue.
- The server computes prompts in chunks of 256 tokens between decode steps. Thus a new prompt does not stop the other
  requests.
- The tokens of a request do not depend on the other requests. With a `seed`, or with greedy decode, a request gets the
  same answer alone or with others, bit for bit, on Metal and CUDA.
- Decode steps run in forwards of up to 8 slots.
- Experimental (Metal): `NSLM_BATCH_GEMM=32` runs steps of 32 or more slots through the prompt GEMMs. This is faster at
  large batches (M4 Max, q8mx: 171 tok/s at 64 streams, against about 112). It gives up bit-for-bit equality.

### Prefix reuse

A request takes the free slot whose cache shares the longest prefix with its prompt. The server does not compute
repeated system prompts, tools and earlier turns again.

### Cold cache on disk

When a request ends with at least `--kv-disk-min` (2048) cached tokens, the server saves its cache to disk in
256-token blocks. A later request with the same prefix reads the blocks back. It does not compute them again.

| Option | Default | Use |
|---|---|---|
| `--kv-disk` | `~/.cache/nslm/kv` | The folder for the blocks |
| `--kv-disk-gb` | 32 | The maximum size. The server deletes the least recently used blocks first. 0 turns the cache off |
| `--kv-disk-days` | 30 | The server deletes blocks that it did not read or write for this number of days |
| `--kv-disk-min` | 2048 | The minimum number of cached tokens to save |

The key of each block is made from the tokens and the model weights. Thus a different model or quantization never
uses the blocks of another model.

### Context and the KV cache

- `--ctx` sets the context of each slot (prompt and output). On a 32 GB Mac, 4096 fits. On a larger Mac, use a larger
  value, for example `--ctx 32768`.
- On macOS 26.4 or later, the Metal engine pages the KV cache. It reserves `--ctx` for each slot, but it commits
  memory only for positions that it writes. Thus only the contexts in use take memory. `NSLM_KV_DENSE=1` commits all
  memory at the start.
- `--kv q8` keeps a GQA cache in 8 bits on both engines. MLA models take `--kv fp8` or `--kv fp4`. Read
  [The KV cache](../concepts/kv-cache.md).
- On a CUDA GPU, read [NVIDIA GPUs](cuda.md) for where the engine keeps the cache.

## Qwen3.8-Flash-Next on CUDA

`nslm-chat` and `nslm-serve` run Qwen3.8-Flash-Next. They take the engine from the model's `config.json`. The
server holds one sequence slot for this model. It does not save caches to disk.

```bash
build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-P3 --effort low "Why do tide pools matter?"
build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-P3 --ctx 32768
```

- `--effort` takes xhigh (the default), medium or low. `--no-think` turns the reasoning block off.
- In a request, `reasoning_effort: high` means xhigh. `chat_template_kwargs` takes `enable_thinking` and
  `preserve_thinking`.
- Tool calls come back as OpenAI `tool_calls`, typed by the tools' schemas.
- Send each reply back with its `reasoning_content`. Then the next turn keeps the cache. Without it, the engine starts the
  prompt again: its DeltaNet state cannot go back.
- The expert cache takes the free VRAM. `NSLM_EXPERT_VRAM_MB` sets its size. The `Q4E_*` variables set the other
  engine options ([Environment variables](../reference/env.md)).

`nslm-qwen4-gen` runs one prompt through the engine, with greedy decoding and the options below.

```bash
build/bin/nslm-qwen4-gen --model Qwen3.8-Flash-Next-NanoSeedLM-P4 --prompt "Why do tide pools matter?"
```

| Option | Use |
|---|---|
| `--vram-experts` | Keep the routed experts in VRAM. The default keeps them in mapped host memory |
| `--expert-cache-gb N` | Cache N GB of the host experts in VRAM (least recently used), at least one layer's. -1: all free VRAM. The cache leaves `NSLM_VRAM_RESERVE_MB` free (2560 on a GPU that drives a display) |
| `--stream-experts` | With a cache: read the missing experts from the model files. Use it when RAM is smaller than the experts |
| `--ple-gguf FILE` | Read the PLE table from an Unsloth engram GGUF (BF16, Q8_0, Q5_1 or IQ4_NL) |
| `--ctx N` | Context positions (4096) |
| `--prompt-rows N` | Prompt rows of one forward: 4096, or 16384 on GPUs of 64 GB or more, at most the context. A row takes about 0.25 MB of VRAM |
| `--drafts K` | The MTP head drafts K tokens (1 to 3), and one forward verifies them. The text does not change. VRAM: 115 MB a draft, and the head's routed experts share the expert cache (else 1.4 GB). On an RTX 4080 it is not faster ([results](../results/2026-11-05-qwen38-mtp-4080-rerun.md)) |
| `--tail-skip R` | Lossy, with an expert cache: a missed expert that every decode row ranks at R or later (rank 0 is the top choice) is not copied or computed. 0, the default, turns it off. It fails its KL gate at R = 7 to 9 ([results](../results/2026-11-05-qwen38-tail-hf.md)) |
| `--cpu-tail N` | With an expert cache, pinned P = 3 experts and a Clang build: the CPU computes up to N missed experts a decode layer (lowest rank first) beside the PCIe copies. Exact up to the order of sums, so the text can change at the rounding level. 0, the default, turns it off. Not faster on an RTX PRO 6000 ([results](../results/2026-11-05-qwen38-tail-hf.md)) |
| `--effort xhigh\|medium\|low` | The template's reasoning effort (xhigh) |
| `--no-think` | The template's `enable_thinking=False`: no reasoning block |
| `--raw` | No chat template. Otherwise the prompt is one user turn through the model's chat template |

## See also

- [Command-line tools](../reference/tools.md)
- [Environment variables](../reference/env.md)
