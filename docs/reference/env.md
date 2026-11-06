# Environment variables

## Both engines

| Variable | Default | Use |
|---|---|---|
| `NSLM_MODEL_DIR` | | The model folder for the tests (Windows build) |

## Metal engine

| Variable | Default | Use |
|---|---|---|
| `NSLM_KV_DENSE=1` | off | Commit all KV cache memory at the start. The default pages the cache (macOS 26.4 or later) |
| `NSLM_BATCH_GEMM=N` | off | Run decode steps of N or more slots through the prompt GEMMs. Faster at large batches. Gives up bit-for-bit equality |

## CUDA engine

| Variable | Default | Use |
|---|---|---|
| `NSLM_VRAM_RESERVE_MB` | 512. Qwen3.8-Flash-Next: 1024 on a GPU that drives a display | VRAM to keep free. The Qwen3.8-Flash-Next expert cache shrinks to keep it. Set it higher if another program on the GPU needs VRAM while the model runs |
| `NSLM_EXPERT_MIN_MB` | 6144 | VRAM to keep for experts when the KV cache is large |
| `NSLM_KV_VRAM_MB`, `NSLM_EXPERT_VRAM_MB` | | Fixed VRAM shares for the KV cache and the experts |
| `NSLM_CACHE_STATS=1` | off | Write the expert cache hits at exit |
| `NSLM_NO_GRAPH=1` | off | Do not use CUDA graphs for decode (K2-Horizon-MoVA and Qwen3.8-Flash-Next) |
| `NSLM_NO_PREDICT=1` | off | Do not predict the experts of the next layer during decode (K2-Horizon-MoVA) |
| `Q4E_PROFILE=1` | off | Qwen3.8-Flash-Next: write the GPU time of each forward stage at exit, and the CPU tail's times |
| `Q4E_ROUTES=FILE` | off | Qwen3.8-Flash-Next: write the routed experts of each forward, for `tools/qwen4_cache_sim.py` |
| `Q4E_NO_PREFETCH=1` | off | Qwen3.8-Flash-Next: do not copy the next layer's experts while a prompt layer computes |
| `Q4E_PLE_THREADS=N` | CPU cores, at most 16 | Qwen3.8-Flash-Next: host threads for the PLE rows. Decode rows split by head. The CPU tail uses one fewer |
| `Q4E_TAIL_RENORM=1` | off | Qwen3.8-Flash-Next `--tail-skip`: scale the kept weights of a row up to its sum |
| `Q4E_CPU_TAIL=N` | 0 | Qwen3.8-Flash-Next in `nslm-chat` and `nslm-serve`: `nslm-qwen4-gen --cpu-tail N` |
| `Q4E_VRAM_EXPERTS=1` | off | Qwen3.8-Flash-Next in `nslm-chat` and `nslm-serve`: `nslm-qwen4-gen --vram-experts` |
| `Q4E_STREAM_EXPERTS=1` | off | Qwen3.8-Flash-Next in `nslm-chat` and `nslm-serve`: `nslm-qwen4-gen --stream-experts` |
| `Q4E_PLE_GGUF=FILE` | | Qwen3.8-Flash-Next in `nslm-chat` and `nslm-serve`: `nslm-qwen4-gen --ple-gguf FILE` |
| `Q4E_ROWS=N` | 4096 or 16384 | Qwen3.8-Flash-Next in `nslm-chat` and `nslm-serve`: `nslm-qwen4-gen --prompt-rows N` |
| `Q4E_PASS_ROWS=N` | 32768 | Qwen3.8-Flash-Next on CUDA: the most prompt rows of one layer-major pass. Each row keeps 40 KB of VRAM |
| `Q4E_DRAFTS=K` | 0 | Qwen3.8-Flash-Next in `nslm-chat` at temperature 0: `nslm-qwen4-gen --drafts K` (1 to 3 MTP drafts a forward, the same text). Set, it opens the MTP head, which every forward runs. `nslm-serve` does not draft |
| `NSLM_SEED_GEMM_F32=1` | off | Keep seed weights in f32 in the prompt GEMM. This changed neither KLD nor NLL, and it costs about 30% of prompt speed at 4K |
| `NSLM_PREFILL_RANGE` | 8192, or 32768 when the KV cache is partly in host memory | Prompt rows that the engine computes layer by layer at a time. Each range copies the earlier host rows again |
| `NSLM_MLA_DEC_KEYS` | 8192 (128 MiB of VRAM) | MLA keys that the engine decompresses at a time for prompt rows. 0 attends prompts absorbed, as decode does |
| `NSLM_MLA_SCALAR=1` | off | Use the earlier one-pass latent kernel, for comparison |

## MLX loader

| Variable | Default | Use |
|---|---|---|
| `NSLM_MAX_CONTEXT` | | The longest context (tokens) that you plan to use. The loader picks the MLA cache format from it |
| `OMLX_K2_MODEL` | | The oMLX K2-Horizon model file, for `make test-mlx` and the MLX tools |
