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
| `NSLM_VRAM_RESERVE_MB` | 512 | VRAM to keep free |
| `NSLM_EXPERT_MIN_MB` | 6144 | VRAM to keep for experts when the KV cache is large |
| `NSLM_KV_VRAM_MB`, `NSLM_EXPERT_VRAM_MB` | | Fixed VRAM shares for the KV cache and the experts |
| `NSLM_CACHE_STATS=1` | off | Write the expert cache hits at exit |
| `NSLM_NO_GRAPH=1` | off | Do not use CUDA graphs for decode |
| `NSLM_NO_PREDICT=1` | off | Do not predict the experts of the next layer during decode |
| `NSLM_SEED_GEMM_F32=1` | off | Keep seed weights in f32 in the prompt GEMM. This changed neither KLD nor NLL, and it costs about 30% of prompt speed at 4K |
| `NSLM_PREFILL_RANGE` | 8192, or 32768 when the KV cache is partly in host memory | Prompt rows that the engine computes layer by layer at a time. Each range copies the earlier host rows again |
| `NSLM_MLA_DEC_KEYS` | 8192 (128 MiB of VRAM) | MLA keys that the engine decompresses at a time for prompt rows. 0 attends prompts absorbed, as decode does |
| `NSLM_MLA_SCALAR=1` | off | Use the earlier one-pass latent kernel, for comparison |

## MLX loader

| Variable | Default | Use |
|---|---|---|
| `NSLM_MAX_CONTEXT` | | The longest context (tokens) that you plan to use. The loader picks the MLA cache format from it |
| `OMLX_K2_MODEL` | | The oMLX K2-Horizon model file, for `make test-mlx` and the MLX tools |
