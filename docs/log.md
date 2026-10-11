# Log

This is the record of experiments, results and releases, in date order. Each entry starts with
`## [YYYY-MM-DD] kind | title`. The kinds are `release`, `engine`, `kernel`, `loader`, `result` and `docs`. To list
the entries of one kind, use `grep '^## \[.*\] result' docs/log.md`.

Entries are not changed after they are written. If a later result replaces an earlier one, a new entry says so. The
current state is in the [concept pages](index.md#concepts).

## [2026-10-06] release | NanoSeedLM: seeded experts for K2-Horizon-MoVA on Apple Silicon

The first release: C99 and Metal engine, P = 4 seed search, packer and MLX loader. The published model is
`K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v` (22.76 GB). Commit
[`bdb1657`](https://github.com/mbarnson/NanoSeedLM/commit/bdb1657). Results:
[GQA seed model against BF16 and Q4](results/2026-10-06-gqa-seed-model.md).

## [2026-10-07] engine | Windows and Linux with CUDA

The CUDA engine, with the same C sources and tests as Metal. Expert cache over PCIe, prompts layer by layer, CUDA
graphs for decode. Commit [`f5c6ac9`](https://github.com/mbarnson/NanoSeedLM/commit/f5c6ac9). Results:
[CUDA and Metal measured side by side](results/2026-10-07-cuda-vs-metal.md).

## [2026-10-07] engine | The 8-bit KV cache on Metal

`--kv q8` on Metal, as on CUDA. Commit [`e77b402`](https://github.com/mbarnson/NanoSeedLM/commit/e77b402). Results:
[Engine KV cache codes](results/2026-10-08-engine-kv-codes.md).

## [2026-10-08] engine | MLA models in the engine

The engine runs K2-Horizon-MoVA converted to MLA by TransMLA. Commit
[`b974f5c`](https://github.com/mbarnson/NanoSeedLM/commit/b974f5c). Read
[Multi-head latent attention](concepts/mla.md).

## [2026-10-08] engine | Continuous batching and the cold KV cache

`nslm-serve` gets sequence slots, chunked prefill and batched decode. A request gets the same tokens alone or in a
batch, bit for bit. Long caches go to disk in 256-token blocks. Commits
[`a70b9ec`](https://github.com/mbarnson/NanoSeedLM/commit/a70b9ec) and
[`d8add77`](https://github.com/mbarnson/NanoSeedLM/commit/d8add77).

## [2026-10-08] kernel | Flash-style MLA attention on Metal

`k_mla_attn` with key tiles on simdgroup matrices and online softmax. Prefill of 2048 tokens went from 21.3 s to 6.6 s.
Commit [`9da7e9b`](https://github.com/mbarnson/NanoSeedLM/commit/9da7e9b). Results:
[MLA kernels on Metal and CUDA](results/2026-10-08-mla-kernels.md).

## [2026-10-08] kernel | Tensor-core MLA attention on CUDA

Absorbed decode on tensor cores, and decompressed (expanded) prompt attention. Commit
[`6120fa4`](https://github.com/mbarnson/NanoSeedLM/commit/6120fa4). Results:
[MLA kernels on Metal and CUDA](results/2026-10-08-mla-kernels.md).

## [2026-10-08] kernel | Expanded MLA prompt attention on Metal

Prompts of 256 or more new tokens attend with the latent expanded per head. Shorter prompts stay absorbed. Commit
[`1d9e1c2`](https://github.com/mbarnson/NanoSeedLM/commit/1d9e1c2).

## [2026-10-08] result | MLA KV cache in FP8 and FP4

`--kv fp8` is 52% of BF16 and `--kv fp4` is 38%, with held-out KLD 0.2221 and 0.2223 against 0.2199. Commit
[`1551e9e`](https://github.com/mbarnson/NanoSeedLM/commit/1551e9e). CUDA followed in
[`6364ffb`](https://github.com/mbarnson/NanoSeedLM/commit/6364ffb). Results:
[Engine KV cache codes](results/2026-10-08-engine-kv-codes.md).

## [2026-10-08] engine | Paged KV cache on Metal

On macOS 26.4 or later, the engine commits KV memory as it writes positions. An earlier attempt with anonymous memory
(`f3ac812`) was reverted: its saving was in the accounting only. Commit
[`25a0b30`](https://github.com/mbarnson/NanoSeedLM/commit/25a0b30).

## [2026-10-08] result | MLA projections in Q8, Q4 and seeds

Q8, Q4 and P = 4 seed encodings for the MLA projections, then a GPTQ search, then `v_up` in Q8 by default. At the size
of Q4, seeds keep about two thirds of what Q4 loses. Commits
[`f155f70`](https://github.com/mbarnson/NanoSeedLM/commit/f155f70),
[`3919940`](https://github.com/mbarnson/NanoSeedLM/commit/3919940),
[`8c0684a`](https://github.com/mbarnson/NanoSeedLM/commit/8c0684a) and
[`dbe5bf6`](https://github.com/mbarnson/NanoSeedLM/commit/dbe5bf6). Results:
[MLA projections in BF16, Q8, Q4 and seeds](results/2026-10-08-mla-projections.md).

## [2026-10-10] release | MLA seed models and the R-02 comparison

Merged the MLA seed work ([`fdb0121`](https://github.com/mbarnson/NanoSeedLM/commit/fdb0121)). Published the MLA
models: BF16, P4, P3, oQ4e, oQ6e and oQ8e. In a comparison with pre-registered hypotheses, P = 4 seeds came a little
closer to the original than oQ4e on both GQA and MLA. Results:
[Seeds against affine quantization](results/2026-10-10-seeds-vs-affine.md).

## [2026-10-10] kernel | Faster resident MLA prefill on CUDA

Commit [`d9e3533`](https://github.com/mbarnson/NanoSeedLM/commit/d9e3533).

## [2026-10-10] engine | CPU one token ahead, and a CPU seed mat-vec

Engine decode with the CPU one token ahead (`gen_pipe`), and a P = 4 seed mat-vec on the CPU. Commit
[`c6f9c12`](https://github.com/mbarnson/NanoSeedLM/commit/c6f9c12).

## [2026-10-10] loader | MLA long context in the MLX loader

The MLX loader keeps the MLA cache in q8 or rq4, and picks one from the context that you ask for. Attention memory
stays bounded at long contexts. Commit [`0d5810d`](https://github.com/mbarnson/NanoSeedLM/commit/0d5810d). Read
[The KV cache](concepts/kv-cache.md).

## [2026-10-10] docs | A short README and a documentation wiki

The README now gives the purpose, the trade-offs and how to start. The technical content moved to `docs/`, with this
log, an index and `llms.txt`.

## [2026-10-11] kernel | Faster CUDA prefill: shared BF16 activations, no empty warp MMAs

The prompt GEMMs of a layer reuse one BF16 copy of their inputs. The kernels skip MMAs for output warps with no work.
On an RTX 4080 (Q4 attention, FP8 KV cache), short prefill is 9.1% faster, and decode does not change. Token and
logit checks and the 128K and 256K tests passed. The CPU seed mat-vec is now built only with Clang. Metal
verification is pending. Commit [`ab6029d`](https://github.com/mbarnson/NanoSeedLM/commit/ab6029d).

## [2026-10-11] loader | A fused MLA prompt kernel, and hybrid MLA models

The MLX loader processes MLA prompts with one Metal kernel per block of keys: 27% to 38% faster at 32K to 256K, and
1.3 GB to 1.5 GB less transient memory. `mla_ranks` 0 keeps a layer's GQA attention (a hybrid model). The MLX loader
runs hybrid models; the config parser, the packer and the seed search accept them; the engines refuse them for now.
Read [A fused MLA prompt kernel in the MLX loader](results/2026-10-11-mla-fused-prefill.md). Pull request
[#14](https://github.com/mbarnson/NanoSeedLM/pull/14).
