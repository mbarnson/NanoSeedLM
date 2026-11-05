# Metal and CUDA parity

NanoSeedLM keeps its Metal engine and its CUDA engine at parity. The two engines compile the same C sources for
everything except the GPU code. They implement the same engine API, with the same kernel arithmetic and the same BF16
rounding points. The tests check each GPU kernel against a C reference on both engines. The MLX loader also gives the
same results as the engine.

## Key numbers

| Check | Result |
|---|---|
| KLD against BF16, GQA p4mx-q4v, held-out / chat | RTX 4080: 0.0260 / 0.0124. M4 Max: 0.0260 / 0.0126 |
| KLD, GQA seeds, MLX loader against C engine | 0.02176 against 0.0218 |
| KLD, MLA-P4, MLX loader against C engine | 0.0985 against 0.0988 |
| A request alone against the same request in a batch (greedy or seeded) | Bit for bit the same, Metal and CUDA |
| Results with and without the CUDA expert cache | Bit for bit the same |

## How we keep parity

1. **One C code base.** The tokenizer, the chat template, the server, the tools, the model reader and the tests are
   the same C files on macOS, Windows and Linux. Only `engine/mova_gpu.m` with `engine/kernels_moe.metal`, and
   `engine/mova_cuda.c` with `engine/kernels_moe.cu`, are different.
2. **A C reference for each kernel.** `test_mova_kernels` runs each GPU kernel through a small backend for each GPU
   API (`tests/kernel_backend_metal.m`, `tests/kernel_backend_cuda.c`). It compares the output with a C reference.
3. **A C reference for the full forward.** `nslm-mova-refcheck` compares the GPU engine with a forward on the CPU. It
   reports KL divergence, argmax and router choices.
4. **The same tests on each platform.** `make test`, `win\build.bat test` and `ctest` run the same tests.
5. **A failing test first.** Each new kernel starts as a test against the C reference.

## Where the engines differ on purpose

| Difference | Why | Effect |
|---|---|---|
| CUDA prompt GEMMs round seed weights to BF16 | The tensor cores take BF16 | No change in KLD or NLL. `NSLM_SEED_GEMM_F32=1` keeps f32 |
| CUDA keeps experts in host memory and caches them in VRAM | A discrete GPU has less memory than the model | No change in results. Speed depends on the cache hit rate |
| CUDA keeps the KV cache in VRAM and in pinned host memory | The same | No change in results |
| Metal pages the KV cache (macOS 26.4 or later) | Unified memory allows sparse buffers | Less memory for short conversations |
| `NSLM_BATCH_GEMM` (Metal, experimental) | Faster decode at large batches | Gives up bit-for-bit equality between batch sizes |

## Feature matrix

| Feature | Metal engine | CUDA engine | MLX loader (oMLX) |
|---|---|---|---|
| GQA model with P = 4 experts | Yes | Yes | Yes |
| MLA projections in BF16, Q8, Q4 | Yes | Yes | Yes |
| MLA projections as P = 4 seeds | Yes | Yes | Yes |
| P = 3 seed tensors (MLA-P3) | Yes | Not yet | Yes |
| Dense K2-Horizon-0.9B | No | No | Yes |
| GQA KV cache q8 | Yes | Yes | oMLX options |
| MLA KV cache fp8, fp4 | Yes | Yes | No |
| MLA KV cache q8, rq4 | No | No | Yes |
| Hybrid MLA (`mla_ranks` 0: GQA layers) | Yes | Yes | Yes |
| Qwen3.8-Flash-Next with P = 3 / 4 seed experts | Yes (`nslm-qwen4-gen`, `qwen4_gpu.m`) | Yes (`nslm-qwen4-gen`) | Yes (mlx-vlm) |
| Qwen3.8-Flash-Next PLE table from an Unsloth engram GGUF | No | Yes (`--ple-gguf`) | No (mlx-vlm's SSD PLE view) |
| Qwen3.8-Flash-Next VRAM cache of host experts | No | Yes (`--expert-cache-gb`) | No |
| Qwen3.8-Flash-Next in `nslm-chat` and `nslm-serve` (`engine/engine.c`) | Yes | Yes | Yes (mlx-vlm) |
| Qwen3.8-Flash-Next MTP drafts (`q4e_draft`, `nslm-qwen4-gen --drafts`) | Yes (`qwen4_gpu.m`) | Yes | Yes (mlx-vlm) |
| Qwen3.8-Flash-Next tail skip of missed experts (`Q4EOpts.tail_skip`, `--tail-skip`) | Not needed: every expert is in memory | Yes | No |
| Affine Q4 with group 32 (the 4-bit MLX packs) | Matvec, GEMM | Yes (matvec, GEMM, embedding) | Yes |
| Qwen3.8-Flash-Next kernels (`test_qwen4_kernels`) | Yes (`kernels_qwen4.metal`) | Yes (`kernels_qwen4.cu`) | Yes (mlx-vlm) |
| Fused kernel for MLA prompts | `k_mla_prefill` | `k_mla_prefill` | `_MLA_PREFILL` |
| Paged KV cache | Yes (macOS 26.4 or later) | No | No |
| OpenAI-compatible server | `nslm-serve` | `nslm-serve` | oMLX |
| Seed search | Yes | Yes | No |
| Seed search of other block sizes (C = 12 and 16, `searchc.cu`) | No | Yes | No |

## See also

- [NVIDIA GPUs](../guides/cuda.md)
- [CUDA and Metal measured side by side](../results/2026-10-07-cuda-vs-metal.md)
- [Build and test](../guides/build.md)
