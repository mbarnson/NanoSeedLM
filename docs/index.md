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
- [2026-11-04: MTP drafts in the CUDA engine](plans/2026-11-04-qwen38-mtp.md): draft. The model's MTP head drafts
  tokens, and one main forward verifies them (+31% on code on the M4 Max).
- [2026-11-05: Trellis-coded experts at 2 to 3 bits](plans/2026-11-05-trellis-experts.md): a QTIP-style trellis code for
  the experts, against Unsloth's quality curve. Gate TA passes at 3 bits.
- [2026-11-05: Qwen3.8-Flash-Next decode on the RTX 4080 near Strata's speed](plans/2026-11-05-qwen38-decode-4080.md):
  draft. Cheaper decode rows: the MTP head in the cache, a matvec by expert, the tail skip, and the CPU for misses.
- [2026-11-04: A high seed tier that follows the routing](plans/2026-11-04-qwen38-adaptive-tiers.md): draft. The
  engine keeps the session's hot experts in P4 and the rest in S3 or P3.
- [2026-11-04: Three seed packs of Qwen3.8-Flash-Next by RAM size](plans/2026-11-04-qwen38-ram-tiers.md): packs for
  64, 96 and 128 GB of RAM and a 12 GB GPU. Each must match an Unsloth V3.1 pack.
- [2026-11-05: Prompt attention that loads each key tile once for several rows](plans/2026-11-05-qwen38-shared-tile-attention.md):
  stopped: 2.7 times fewer key loads, but 1.7 times slower.

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
- [2026-11-04: Expert cache misses of MTP verify windows](results/2026-11-04-qwen38-verify-windows.md): verify windows
  of 2 to 4 tokens do not reduce the misses for each token.
- [2026-11-04: Qwen3.8-Flash-Next decode with affine, P4 and P3 experts](results/2026-11-04-qwen38-decode-by-format.md):
  P3 46.1, affine 42.4, P4 40.0 tokens per second on one GPU.
- [2026-11-04: Layer-major prompt passes for Qwen3.8-Flash-Next](results/2026-11-04-qwen38-prompt-passes.md): 549 to
  977 tokens per second on a 15,922-token prompt.
- [2026-11-04: Seed layouts against Unsloth's tensors on one layer](results/2026-11-04-qwen38-seed-probe.md): 3-bit
  seeds have 16% more weighted error than IQ3_XXS. Gate G6 fails.
- [2026-11-04: Expert use by text domain](results/2026-11-04-qwen38-routing-domains.md): a hot third of the experts
  serves 80% to 93% of its own text's router choices, and 20% to 63% of another domain's.
- [2026-11-04: Seed packs and Unsloth packs against BF16](results/2026-11-04-qwen38-kld-vs-bf16.md): P4 0.0408 at
  4.50 bits is on Unsloth's curve. P3 0.0600 at 4.00 bits is 18% above it. Gate G5 passes.
- [2026-11-04: Hybrid P3 / P4 seed packs](results/2026-11-04-qwen38-hybrid-tiers.md): at 4.17 bits, 0.054 with a
  random or `down_proj` third in P4, 0.0450 with the in-domain hot third.
- [2026-11-04: 3-bit seeds (S3) for the 64 GB class](results/2026-11-04-qwen38-s3-tier.md): S3 alone 0.142, twice
  UD-IQ3_XXS. With the text's own hot third in P3, 0.0765 at 3.35 bits.
- [2026-11-04: Long-prompt prefill by stage](results/2026-11-04-qwen38-prefill-stages.md): 1,026 to 1,234 tokens per
  second on a 15,922-token prompt (RTX PRO 6000), from the attention kernel and one pass for the context.
- [2026-11-04: A high seed tier that follows the routing](results/2026-11-04-qwen38-tier-sim.md): a simulation. At
  10.4 GB the tier serves 64% to 71% of router choices and loads 3 to 5 experts for each token.
- [2026-11-05: Prefill with larger chunks and an expert prefetch](results/2026-11-05-qwen38-prefill-chunks.md): 1,343
  to 2,801 tokens per second on a 15,922-token prompt (RTX PRO 6000), 1,571 on an A100 at 29,120 tokens.
- [2026-11-05: cp.async attention, merged GEMMs and a register router](results/2026-11-05-qwen38-merged-gemms.md):
  2,801 to 3,403 tokens per second on 15,922 tokens, 3,423 at 29,120, and decode 44.0 to 54.2 (RTX PRO 6000).
- [2026-11-05: One-pass prompt attention and the same quality](results/2026-11-05-qwen38-one-pass-attention.md): 3,660
  tokens per second at 29,120 tokens (RTX PRO 6000), 1,879 on an A100. KL against the 4-bit model unchanged.
- [2026-11-05: Prompt chunks of 16,384 rows on large GPUs](results/2026-11-05-qwen38-rows-by-vram.md): 4,222 tokens
  per second at 29,120 tokens and 4,165 at 15,922 (RTX PRO 6000), the default on GPUs of 64 GB or more.
- [2026-11-05: P3 against llama.cpp UD-Q3_K_XL](results/2026-11-05-qwen38-vs-llamacpp.md): experts in host memory,
  4,222 and 53.8 tokens per second against 793 and 9.9. All in VRAM, 4,418 and 87.6 against 5,130 and 114.
- [2026-11-05: Faster decode norms and RoPE; the expert copy is PCIe-bound](results/2026-11-05-qwen38-decode-small-kernels.md):
  P3 decode 54.6 to 56.5 tokens per second (8 GB cache, RTX PRO 6000), the same text. More copy blocks do not help.
- [2026-11-05: Long-context decode 34% faster](results/2026-11-05-qwen38-long-context-decode.md): after 15,975
  tokens, P3 decode 43.8 to 58.8 tokens per second (split-key attention, 8-bit QSA top-k). Decode-row KL unchanged.
- [2026-11-05: Prompt attention over 4 rows' shared key tiles is slower](results/2026-11-05-qwen38-shared-tile-attention.md):
  480 to 805 ms at 15,922 tokens with 2.7 times fewer key loads. Reverted.
- [2026-11-05: MTP drafts on the RTX 4080: the same text, not faster](results/2026-11-05-qwen38-mtp-4080.md): P3 code
  29.7 tokens per second without drafts, 27.6 to 28.9 with 1 to 3. Each verify row pays its expert copies again.
- [2026-11-05: The RTX 4080 PC: CPU seed rate and streamed experts](results/2026-11-05-qwen38-4080-cpu-and-streamed.md):
  P3 on the 5800X3D at 20.7 G weights/s, so a CPU share saves at most about 6 of 34 ms a token. Streamed decode 14.7 to 19.8.
- [2026-11-05: PLE rows on a thread pool](results/2026-11-05-qwen38-4080-ple-pool.md): P3 decode on the RTX 4080 30.30
  to 33.17 tokens per second, and 37.69 with the automatic cache and the display reserve.
- [2026-11-05: MTP drafts on the M4 Max](results/2026-11-05-qwen38-mtp-m4max.md): P3 code 31% to 37% faster with 1 to 3
  drafts, chat 1% to 20%, the same text.
- [2026-11-05: MTP drafts on the M4 Max, again](results/2026-11-05-qwen38-mtp-m4max-rerun.md): after the day's Metal
  kernels, P3 code 48.9 -> 61.5 tokens per second with 1 draft, chat 47.2 -> 57.5, the same text.
- [2026-11-05: MTP drafts on the RTX 4080, again](results/2026-11-05-qwen38-mtp-4080-rerun.md): after the PLE and RAM
  fixes, drafts are 14% to 27% slower. Each verify row pays the PCIe copies of its missed experts.
- [2026-11-05: Strata on the RTX 4080 PC](results/2026-11-05-strata-4080-probe.md): IQ3_S at 85 tokens per second with
  MTP and 57 without. A step costs 10.5 ms + 7.1 ms a row, against our 9 ms + 25 ms.
- [2026-11-05: P3H, the MTP head with P = 3 seed experts](results/2026-11-05-qwen38-p3h-head.md): on the RTX 4080 the
  head joins the expert cache. Code with drafts is 8% to 9% faster, but drafts still lose there. No gain on the M4 Max.
- [2026-11-05: MTP drafts in nslm-chat on the M4 Max](results/2026-11-05-qwen38-mtp-chat-m4max.md): `Q4E_DRAFTS=2`
  decodes P3 code at 69.6 tokens per second, against 47.8 without drafts, with the same text. Chat: +8.5% at 1 draft, -13% at 3.
- [2026-11-05: The tail skip and the CPU tail on an RTX PRO 6000](results/2026-11-05-qwen38-tail-hf.md): the tail skip
  fails gate DL at R = 7 to 9. The CPU tail keeps the KL but halves decode there: one P3 expert is 1.6 ms of one core.
- [2026-11-05: Trellis-coded experts: weight error](results/2026-11-05-trellis-probe.md): at 3.1 and 4.1 bits a trellis
  has less weighted error than ggml's IQ3_S (3.44 bits) and IQ4_XS (4.25 bits). It saves about 0.4 to 0.5 bits a weight.

## For contributors and agents

- [AGENTS.md](../AGENTS.md): the rules for this documentation and for the code.
- [llms.txt](../llms.txt): a short entry point for AI agents.
