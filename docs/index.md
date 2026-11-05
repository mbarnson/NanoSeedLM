# NanoSeedLM documentation

This is the catalog of all pages. Each line gives a link and one sentence about the page. The
[README](../README.md) gives the purpose and the trade-offs. [log.md](log.md) gives the history.

## Concepts

What the methods are, how they work, and their current numbers.

- [SeedLM](concepts/seedlm.md): weights as seeds of an LFSR, the seed search, the kernels, and why seeds are slower.
- [Multi-head latent attention](concepts/mla.md): the conversion from GQA to MLA, the cache, the attention modes, and
  the quality cost.
- [The KV cache](concepts/kv-cache.md): cache sizes, the q8 and rq4 caches of the MLX loader, the fp8 and fp4 caches
  of the engine, and the longest context for each Mac.
- [Metal and CUDA parity](concepts/parity.md): how the two engines stay the same, where they differ, and a feature
  matrix.

## Guides

How to do a task.

- [Build and test](guides/build.md): requirements, build commands and tests for macOS, Windows and Linux.
- [Run and serve](guides/run-and-serve.md): `nslm-chat`, and the `nslm-serve` API, batching and caches.
- [The MLX loader](guides/mlx.md): how to run the models in oMLX and mlx-lm, and how to set long contexts.
- [NVIDIA GPUs](guides/cuda.md): how the CUDA engine fits the model on a small GPU.
- [Make a model folder](guides/make-a-model.md): calibration, seed search and packing.

## Reference

Facts to look up.

- [Models](reference/models.md): the published models, their lineage, and which runtimes load them.
- [Command-line tools](reference/tools.md): each tool and its use.
- [Environment variables](reference/env.md): each variable, its default and its use.
- [The model folder format](reference/model-folder.md): tensor encodings, shard layout and MLA entries.
- [Source layout](reference/source-layout.md): the folders of the repository.
- [Related work](reference/related-work.md): papers and projects on seeds and computed weights.

## Plans

Work that is not done yet, with its gates and hypotheses.

- [2026-11-04: SeedLM packs for Qwen3.8-Flash-Next](plans/2026-11-04-qwen38-flash-next.md): steps, gates, costs and
  the PLE table choice, for a PC with a 16 GB GPU.

## Results

Dated measurements. Each page gives the result, its status, the setup and the data.

- [2026-10-06: GQA seed model against BF16 and Q4](results/2026-10-06-gqa-seed-model.md): seeded experts against
  affine Q4 experts on the original model.
- [2026-10-07: CUDA and Metal measured side by side](results/2026-10-07-cuda-vs-metal.md): RTX 4080 against M4 Max.
- [2026-10-08: MLA kernels on Metal and CUDA](results/2026-10-08-mla-kernels.md): MLA attention speed against GQA.
- [2026-10-08: MLA projections in BF16, Q8, Q4 and seeds](results/2026-10-08-mla-projections.md): encodings for the
  MLA projections, GPTQ, and `v_up`.
- [2026-10-08: Engine KV cache codes](results/2026-10-08-engine-kv-codes.md): `--kv q8`, `fp8` and `fp4`.
- [2026-10-10: Seeds against affine quantization (R-02)](results/2026-10-10-seeds-vs-affine.md): P4 and P3 seeds
  against oMLX oQe, on GQA and MLA, with pre-registered hypotheses.
- [2026-10-11: A fused MLA prompt kernel in the MLX loader](results/2026-10-11-mla-fused-prefill.md): prompt speed at
  32K to 256K, memory, and accuracy, on MLA and hybrid models.
- [2026-11-04: Hadamard rotation and P = 4 seeds on Qwen3.8-Flash-Next](results/2026-11-04-qwen38-rotation-probe.md):
  weight error of 4 experts, with and without rotation and activation weighting.
- [2026-11-04: The seed search over the Qwen3.8-Flash-Next experts](results/2026-11-04-qwen38-seed-search.md): P = 4
  and P = 3 on 4 RTX PRO 6000 GPUs, errors, costs and pack sizes.
- [2026-11-04: The Qwen3.8-Flash-Next seed packs against the 4-bit MLX model](results/2026-11-04-qwen38-kld-vs-4bit.md):
  KL divergence, top-1 agreement and perplexity on wikitext-2.
- [2026-11-04: Qwen3.8-Flash-Next-NanoSeedLM-P3 on the RTX 4080](results/2026-11-04-qwen38-4080-p3.md): decode speed
  with experts in host memory, a VRAM expert cache, and streamed experts.
- [2026-11-04: Qwen3.8-Flash-Next-NanoSeedLM-P3 in mlx-vlm on the M4 Max](results/2026-11-04-qwen38-m4max-mlx-p3.md):
  decode speed and peak memory against a 4-bit affine quantization.
- [2026-11-04: The prompt time of Qwen3.8-Flash-Next on the CUDA engine, by stage](results/2026-11-04-qwen38-prompt-profile.md):
  the PLE prefetch and the grouped expert GEMMs. 1,024 tokens take 7.5 s, not 15.4 s.
- [2026-11-04: Prompt rows of 1,024 and a decode expert prediction](results/2026-11-04-qwen38-prompt-rows-and-prediction.md):
  151 to 541 tokens per second on a 3,703-token prompt. The prediction made decode 4% slower.
- [2026-11-04: CUDA graphs for Qwen3.8-Flash-Next decode](results/2026-11-04-qwen38-cuda-graphs.md): P3 decode 10.6%
  faster on an RTX PRO 6000, with the same text.
- [2026-11-04: Expert cache policies for Qwen3.8-Flash-Next decode](results/2026-11-04-qwen38-cache-policies.md): LRU
  beats LFU and a hot set. The optimum would halve the misses.

## For contributors and agents

- [AGENTS.md](../AGENTS.md): the rules for this documentation and for the code.
- [llms.txt](../llms.txt): a short entry point for AI agents.
