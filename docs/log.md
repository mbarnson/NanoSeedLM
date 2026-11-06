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

## [2026-10-11] engine | Hybrid MLA models in the Metal and CUDA engines

The engines and the C reference run hybrid MLA models (`mla_ranks` 0: the layer keeps GQA attention). A GQA layer
keeps a BF16 K and V cache when the MLA layers use FP8 or FP4; the CUDA engine now sizes K rows per layer, as it did
V rows. `test_engine` checks an all-MLA and a hybrid random model against the C reference on Metal and on CUDA (RTX
PRO 6000). The T-03 hybrid on the 31 held-out windows: P4 seeds on Metal, NLL 2.4094 and KL divergence 0.1118
(MLX loader 2.4087, 0.1132); BF16 on CUDA, NLL 2.3979 and KL divergence 0.1048 (MLX loader 2.3978, 0.1054).
Pull request [#14](https://github.com/mbarnson/NanoSeedLM/pull/14).

## [2026-11-04] docs | Plan: SeedLM packs for Qwen3.8-Flash-Next

A plan to make a family of seed packs for Qwen3.8-Flash-Next, with gates before each large cost. The pack does not
include the PLE table. The user downloads one of the four Unsloth table files to fit the disk. The plan also records
a quick check of the CPU P = 4 mat-vec in a cloud session: 3.3 G weights/s for each AVX2 thread. Read
[the plan](plans/2026-11-04-qwen38-flash-next.md).

## [2026-11-04] result | Hadamard rotation and P = 4 seeds on Qwen3.8-Flash-Next experts

On 4 experts of layer 20, a block Hadamard rotation did not change the plain error of P = 4 seeds. An
activation-weighted fit without rotation gave 6% to 10% less activation-weighted error. The expert weights are
already near-Gaussian (kurtosis 3.0 to 3.06). Read
[Hadamard rotation and P = 4 seeds](results/2026-11-04-qwen38-rotation-probe.md).

## [2026-11-04] engine | Qwen3.8-Flash-Next experts in the seed search

`nslm-moe` reads the expert shapes of a `qwen4_exp` model from `config.json`, and the fused expert tensors by slice.
`--imatrix` takes the activation weights from a llama.cpp imatrix GGUF (`nslm/gguf.h`). `tools/hf_seedbank.sh` takes
`IMATRIX` and `ONLY`. A CPU run on layer 20 of the BF16 checkpoint read 1,852 imatrix tensors and the expert counts.

## [2026-11-04] engine | PLE table rows from the Unsloth engram files

`nslm/ple.h` reads rows of `per_layer_token_embd.weight` (160 x 320 M) from a GGUF file in BF16, Q8_0, Q5_1 or IQ4_NL.
On rows at the start, the middle and the end of each Unsloth file, it matched gguf-py exactly. The engine port will
use it, so the user can choose the table precision to fit the disk.

## [2026-11-04] kernel | CPU seed mat-vec for several tokens

`cpu_seed4_tiled_matmul_mt` makes the LFSR states of each block once for up to 8 tokens. Each token's rows equal the
one-token kernel's, bit for bit. On a cloud Xeon with 4 vCPUs (AVX2 build, load average below 0.1), for each token:
1 token 3.3 G weights/s (12.3 on 4 threads), 2 tokens 4.4 (16.8), 4 tokens 6.5 (24.6). One token keeps the
one-token kernel, which is 12% faster than the general one.

## [2026-11-04] kernel | P = 3 seeds on the CPU

`cpu_seed3_tile` puts SEED4 (P = 3) matrices in the tiled CPU layout, and the CPU mat-vec and mat-mul take them. The
rows match the scalar spec within 8e-7. On the same cloud Xeon (4 vCPUs, AVX2 build, load average 0.07), for each
token: 13.9, 21.5 and 32.3 G weights/s at 1, 2 and 4 tokens, against 12.4, 16.8 and 25.1 at P = 4.

## [2026-11-04] engine | A C reference forward of Qwen3.8-Flash-Next

`nslm/qwen4_ref.h` computes the `qwen4_exp` forward on the CPU from an MLX model folder. The GPU engines will be checked
against it. `tools/qwen4_golden.py` makes a tiny random model in the formats of the 4-bit MLX pack, and its float32
logits from mlx-vlm (`tests/data/qwen4_tiny`). The reference matches them within 6e-7, with the same argmax on all 48
rows. The sequence has an EOS and passes the sparse attention budget. Two details mattered:

- Equal indexer scores (ReLU zeros) go to the higher block index, as MLX's `argpartition`.
- The model folder reader takes MLX affine Q4 with group 32. The MoVA engines refuse groups other than 64.

## [2026-11-04] kernel | Qwen3.8-Flash-Next CUDA kernels: Gated DeltaNet, hyper-connections, Q4 group 32

`engine/kernels_qwen4.cu` has the Gated DeltaNet conv and recurrent step, the head RMSNorm and gate, the (1 + w)
RMSNorm and the hyper-connection steps. The matvec, GEMM and embedding kernels take affine Q4 with group 32
(`WSlice.gs`). On an HF Jobs RTX PRO 6000 (`tools/hf_cuda_test.sh`), `test_qwen4_kernels` passed:

- Gated DeltaNet at the Qwen3.8-Flash-Next head widths (128): outputs within 3.8e-7 of `nslm/qwen4_ops.h`, the state
  within 2.2e-7.
- Hyper-connection steps within 1e-7.
- Q4 group 32 within 3.7e-3 of exact dequantization, from the BF16 output rounding.

`test_mova_kernels` passed on the same run, so the group change kept the MoVA kernels as they were. Metal does not have
these kernels yet.

## [2026-11-04] result | The seed search over the Qwen3.8-Flash-Next experts

HF Jobs searched all 24,576 routed experts on 4 RTX PRO 6000 GPUs: 66 min at P = 4, 83 min at P = 3, about $28 for
both. The mean activation-weighted relative error is 0.077 at P = 4 and 0.112 at P = 3. The packs are 105.7 GB (P = 4)
and 98.2 GB (P = 3), against 113.3 GB for the 4-bit MLX model. They are in the storage bucket, not published yet.
Read [The seed search over the Qwen3.8-Flash-Next experts](results/2026-11-04-qwen38-seed-search.md).

## [2026-11-04] engine | The Qwen3.8-Flash-Next CUDA engine

`engine/qwen4_eng.h` runs a Qwen3.8-Flash-Next MLX folder on CUDA: Gated DeltaNet, QSA attention with the indexer, the
softmax MoE with the shared expert, the hyper-connections and the PLE layer. `nslm-qwen4-gen` generates greedily with
timings.
On an HF Jobs RTX PRO 6000:

- Each new kernel is within 2.1e-7 of its C operation (`nslm/qwen4_ops.h`).
- On `tests/data/qwen4_tiny`, the logits are within 2.6% of mlx-vlm's float32 logits, with the same argmax on 47 of 48
  rows. The BF16 rounding of the matvec kernels causes the difference.
- Any split of the tokens into rows gives the same logits.

The PLE table comes from the model folder or, with `--ple-gguf`, from an Unsloth engram GGUF. The GGUF rows match the
folder's rows in order (correlation 0.996 on the first rows). Routed experts stay in mapped host memory by default.

## [2026-11-04] engine | Qwen3.8-Flash-Next-NanoSeedLM-P4 runs on the CUDA engine

On an HF Jobs RTX PRO 6000 (96 GB, experts in VRAM, `tools/hf_qwen4_gen.sh`), `nslm-qwen4-gen` loaded the P4 pack in
11.6 s. It answered "Why do tide pools matter?" with a coherent reasoning block and a correct answer. The 19-token
prompt took 0.18 s, and the decode ran at 57.7 tokens per second. This is one run on a shared cloud machine, with no
record of the machine state, so it is not a benchmark.

## [2026-11-04] engine | Qwen3.8-Flash-Next-NanoSeedLM-P3 runs on the CUDA engine, experts in host memory

On the same kind of machine, with the routed experts in mapped host memory (the default), `nslm-qwen4-gen` loaded the
P3 pack in 27.1 s. It gave a coherent answer to the same prompt. The prompt ran at 25.4 tokens per second and the
decode at 22.1 tokens per second. The GPU reads the experts over PCIe for each token. This is one run, not a benchmark.

## [2026-11-04] engine | A VRAM cache of host experts: 22.1 to 41.6 tokens per second on P3

`--expert-cache-gb` keeps the most recently used routed experts in VRAM, on the MoVA `CachePool` (`k_cache_admit` now
takes 512 experts per layer). The other experts stay in pinned host memory. With the smallest cache, the tiny model's
logits are the same bit for bit. The measurement used an HF Jobs RTX PRO 6000, the P3 pack and an 8 GB cache (3,255
of 24,576 experts). The decode ran at 41.6 tokens per second against 22.1 without the cache. The prompt ran at 48.5
tokens per second against 25.4. The cache hit 69% of the expert reads, and the greedy output was the same text. This is one run,
not a benchmark, and the PCIe link of this machine is not that of an RTX 4080.

## [2026-11-04] result | The Qwen3.8-Flash-Next seed packs against the 4-bit MLX model

The CUDA engine ran both models on 1,024 tokens of wikitext-2. The mean KL divergence against the 4-bit MLX model is
0.177 for P4 and 0.237 for P3. The top-1 token agrees on 91.9% and 88.2% of the tokens. The perplexity rises from
2.396 to 2.500 and 2.480. The reference is a quantized model, not BF16. Read
[The Qwen3.8-Flash-Next seed packs against the 4-bit MLX model](results/2026-11-04-qwen38-kld-vs-4bit.md).

## [2026-11-04] result | Qwen3.8-Flash-Next-NanoSeedLM-P3 on the RTX 4080

With all experts in pinned host memory, the P3 pack decodes at 9.8 tokens per second on the RTX 4080 (PCIe 4.0). A
VRAM expert cache gives 29.4 tokens per second at 8 GB and 31.7 at 10 GB. Two runs of each case gave the same text.
Read [Qwen3.8-Flash-Next-NanoSeedLM-P3 on the RTX 4080](results/2026-11-04-qwen38-4080-p3.md).

## [2026-11-04] engine | Qwen3.8-Flash-Next prompt rows of up to 128 tokens

The qwen4 engine now runs up to 128 rows in one forward, with the tensor-core GEMM for the dense projections past 8 rows.
The test used an HF Jobs RTX PRO 6000 and a 48 GB expert cache. 1,024 tokens of the P4 pack took 13.4 s, against 15.6 s
in rows of 8: 15% faster. The KL divergence against the 4-bit model did not change (0.179, against 0.177). So the dense
projections were not the main cost of a prompt. The expert misses of the cache may be: we did not profile it.

## [2026-11-04] result | Qwen3.8-Flash-Next-NanoSeedLM-P3 in mlx-vlm on the M4 Max

In mlx-vlm on the M4 Max, the P3 pack decodes at 34 tokens per second, as fast as a 4-bit affine quantization. Its MLX
peak memory is 65.5 GB against 74.3 GB. Spotlight ran during the runs, so the numbers are noisy. Read
[Qwen3.8-Flash-Next-NanoSeedLM-P3 in mlx-vlm on the M4 Max](results/2026-11-04-qwen38-m4max-mlx-p3.md).

## [2026-11-04] kernel | Qwen3.8-Flash-Next Metal kernels, and Q4 group 32 on Metal

`engine/kernels_qwen4.metal` has the Metal counterparts of the `engine/kernels_qwen4.cu` kernels that
`test_qwen4_kernels` checks:

- The Gated DeltaNet conv, step and output norm.
- The (1 + w) RMSNorm and the hyper-connection steps.
- The head RMSNorm with RoPE, and the router.
- The PLE gate and conv.
- The QSA block keys and the top-k block select.
- The sparse attention.

The file compiles without fast math. Metal has no double type, so the host computes the RoPE inverse frequencies as
the C ops do. The Metal matvec and GEMM take affine Q4 with group 32 through function constant 2. The Metal engine
does not set it, so its kernels keep group 64.

On the M4 Max, `test_qwen4_kernels` passed. Each qwen4 kernel is within 6.4e-7 of `nslm/qwen4_ops.h`. Q4 group 32 is
within 3.7e-3 of exact dequantization, from the BF16 output rounding. The rest of the `make test` output is the same
as on main. The Metal engine does not run Qwen3.8-Flash-Next yet. The MoE combine, the stream tile and the Q4
embedding are not on Metal yet.

## [2026-11-04] result | The prompt time of Qwen3.8-Flash-Next on the CUDA engine, by stage

`Q4E_PROFILE=1` showed where 1,024 prompt tokens of P4 spent 15.4 s (RTX PRO 6000, 8 GB expert cache). Two changes
bring this to 7.5 s, 2.1 times faster:

- `9d144b8`: the PLE prefetch. `q4e_forward` asks the OS to read all PLE rows of the call first (`madvise` WILLNEED).
  The PLE stage goes from 5,415 ms to 4 ms. The logits do not change.
- `3bec003`: prompt rows of more than 8 run the routed experts as grouped GEMMs (`kc_bucket`, `kc_mm_grouped_dev`). The
  expert stage goes from 3,820 ms to 1,118 ms. `kc_bucket` now takes up to 1,024 experts.

The grouped GEMMs round their inputs to BF16. The KL against the 4-bit model went from 0.179 to 0.176. The reference's
perplexity went from 2.364 to 2.402, so the rounding cost is not known yet. The router and expert cache stage (4.2 s) is
now 57% of the time. Read
[The prompt time of Qwen3.8-Flash-Next on the CUDA engine, by stage](results/2026-11-04-qwen38-prompt-profile.md).

## [2026-11-04] result | Prompt rows of 1,024 and a decode expert prediction for Qwen3.8-Flash-Next on CUDA

- `e6efe3f`: forwards without logits take 1,024 rows. On an RTX PRO 6000 with an 8 GB cache, a 3,703-token prompt of P4
  goes from 151 to 541 tokens per second. The cache misses go from 274k to 64k.
- `382af70`, removed in `8afd74d`: decode predicted the next layer's experts and copied them on a second stream. The
  misses went down by 44%, but decode went from 40.6 to 39.0 tokens per second (P3).
- `cd976f1`: QSA writes its selected blocks in block order. The atomic order made the logits differ from run to run
  past position 2,048. Now 128 and 1,024 rows give the same text.

Read
[Prompt rows of 1,024 and a decode expert prediction](results/2026-11-04-qwen38-prompt-rows-and-prediction.md).

## [2026-11-04] engine | CUDA graphs for Qwen3.8-Flash-Next decode

`10ea8df`: decode rows (up to 8) of the qwen4 engine run as CUDA graphs. The kernels read the position from device
memory, so one graph serves every position. On an RTX PRO 6000 with an 8 GB cache, P3 decodes at 46.5 tokens per second,
against 42.1 without graphs. The text is the same. `NSLM_NO_GRAPH=1` turns the graphs off. Read
[CUDA graphs for Qwen3.8-Flash-Next decode](results/2026-11-04-qwen38-cuda-graphs.md).

## [2026-11-04] result | Expert cache policies for Qwen3.8-Flash-Next decode, from a routing trace

A routing trace of 1,024 decode tokens of P3 (`Q4E_ROUTES`) went through `tools/qwen4_cache_sim.py`. At 3,255 slots
(8 GB), the engine's LRU serves 74.6% of the expert uses, LFU 67.4% and a static hot set at most 62.8%. Belady's
optimum serves 87.6%. So the plan's hot set (S8, V3) does not help, and only a prediction of future routes can reduce
the misses much. Read
[Expert cache policies for Qwen3.8-Flash-Next decode](results/2026-11-04-qwen38-cache-policies.md).

## [2026-11-04] result | Expert cache misses of MTP verify windows for Qwen3.8-Flash-Next

The routing trace of 1,024 P3 decode tokens went through the LRU simulation in windows of 1 to 4 tokens, as MTP verify
steps would be. The misses for each token stay at 118 (8 GB) and 94 (10 GB). So MTP drafts do not reduce the PCIe
traffic, about 290 MB a token for P3 at 8 GB. They can only share the other costs of a step over the accepted tokens.
Read [Expert cache misses of MTP verify windows](results/2026-11-04-qwen38-verify-windows.md).

## [2026-11-04] result | The decode expert prediction on PCIe 4, with CUDA graphs

The next-layer expert prediction of `382af70`, on top of the CUDA graphs (local commit `3390402`), on an HF Jobs A100
(PCIe 4, job `6aebdb82c78863f9c63314d5`): P3 decodes 161 tokens at 18.64 and 18.86 tokens per second with it, and at
20.40 and 20.52 without it (8 GB cache). The misses go from 25.6k to 14.4k, and the text is the same. So the prediction
stays out. Its wrong guesses also cost PCIe traffic, and decode is bound by that bandwidth. The A100 decodes slower than
the RTX 4080 did at `5ed8d1b` (28 tokens per second). So its absolute speeds do not model the 4080.

## [2026-11-04] result | Qwen3.8-Flash-Next decode with affine, P4 and P3 experts on one GPU

On an RTX PRO 6000 with an 8 GB cache and CUDA graphs, P3 decodes at 46.1 tokens per second, the 4-bit affine model at
42.4 and P4 at 40.0. The PCIe traffic is 391, 581 and 502 MB a token. The seed expansion makes the expert stage 1.7 ms
(P3) and 2.6 ms (P4) a token slower than affine. On this PCIe 5 GPU, that eats all of P4's byte saving. Read
[Qwen3.8-Flash-Next decode with affine, P4 and P3 experts](results/2026-11-04-qwen38-decode-by-format.md).

## [2026-11-04] result | Layer-major prompt passes for Qwen3.8-Flash-Next on CUDA

`87ef68a`: prompt rows without logits run in layer-major passes of up to 8,192 rows. Each layer's experts cross PCIe
once a pass, not once for each chunk of 1,024 rows. On an RTX PRO 6000 with an 8 GB cache, a 3,703-token P4 prompt goes
from 536 to 833 tokens per second, and a 15,922-token prompt from 549 to 977. The text does not change. Read
[Layer-major prompt passes for Qwen3.8-Flash-Next](results/2026-11-04-qwen38-prompt-passes.md).

## [2026-11-04] plan | Three seed packs of Qwen3.8-Flash-Next by RAM size

Matt approved a plan for three packs: for 64, 96 and 128 GB of RAM, each with a 12 GB GPU and the BF16 PLE table on
NVMe. Seeds encode every routed expert, in a mix of a new 3-bit tier (C = 12, P = 4), P3, P4 and P8. All new searches
weight the error by Unsloth's imatrix. Read
[Three seed packs of Qwen3.8-Flash-Next by RAM size](plans/2026-11-04-qwen38-ram-tiers.md).

## [2026-11-04] result | Seed layouts against Unsloth's tensors on one layer of Qwen3.8-Flash-Next

`6fe04dd`, `f714f50`: `nslm-moe --probe` measures the imatrix-weighted error of seed layouts on sampled blocks of one
layer. `tools/gguf_experts.py` gives the same measure for the tensors of a GGUF. On layer 20, 3-bit seeds have 0.213
and Unsloth's IQ3_XXS has 0.184, so gate G6 fails. P4 has 0.079 and Q4_K has 0.070. Read
[Seed layouts against Unsloth's tensors on one layer](results/2026-11-04-qwen38-seed-probe.md).

## [2026-11-04] result | Expert use of Qwen3.8-Flash-Next by text domain

`518147a`: `tools/hf_qwen4_routes.sh` records routing traces of the P3 pack on chat, code and wiki text. The most used
third of each layer's experts serves 80% to 93% of its own text's choices. Code's hot third serves 20% to 29% of the
other texts' choices, against 33% for a random third. Read
[Expert use by text domain](results/2026-11-04-qwen38-routing-domains.md).

## [2026-11-04] result | Seed packs and Unsloth packs of Qwen3.8-Flash-Next against BF16 in one harness

`7e6c9c4`, `e8426e0`: `nslm-qwen4-emulate` writes decoded seed experts into Unsloth's Q8_0 GGUF, and
`tools/hf_llama_kld.sh` scores packs against BF16 in llama.cpp with the BF16 PLE table. P4 has 0.0408 at 4.50 bits
for each expert weight, on Unsloth's curve. P3 has 0.0600 at 4.00 bits, and UD-IQ4_XS has 0.0509 at 3.97 bits. Read
[the packs against BF16](results/2026-11-04-qwen38-kld-vs-bf16.md).

## [2026-11-04] result | Hybrid P3 / P4 seed packs of Qwen3.8-Flash-Next against BF16

`2009d76`: the emulator takes each expert from one of several packs by a map (`tools/qwen4_tier_map.py`). At 4.17
bits, a random third, every `down_proj` and the chat and code hot third in P4 all give 0.054. The wiki hot third gives
0.0450 on wikitext. Read [hybrid tiers](results/2026-11-04-qwen38-hybrid-tiers.md).

## [2026-11-04] result | 3-bit seeds (S3) for the 64 GB class of Qwen3.8-Flash-Next

`e8426e0`, `047ad9f`: `nslm-moe --codec s3` searched all experts at C = 12, P = 4 on CUDA (12.9 GPU-hours). S3 alone
has a KLD of 0.142 against BF16, and UD-IQ3_XXS has 0.0735. A random third in P3 gives 0.116 at 3.35 bits. The text's
own hot third in P3 gives 0.0765. Read [3-bit seeds for the 64 GB class](results/2026-11-04-qwen38-s3-tier.md).

## [2026-11-04] result | A high seed tier that follows the routing: a simulation

`tools/qwen4_tier_sim.py` replays the routing traces through a high tier of the most used experts (a decayed count,
refreshed every 16 tokens). At 10.4 GB of P4 it serves 64% to 71% of router choices, and loads 3 to 5 experts for each
token. Read [the tier simulation](results/2026-11-04-qwen38-tier-sim.md).

## [2026-11-04] plan | A high seed tier that follows the routing of Qwen3.8-Flash-Next

A draft for Matt's go: S3 or P3 for all experts, and P4 for the session's hot set, moved by the engine. Read
[the adaptive tiers plan](plans/2026-11-04-qwen38-adaptive-tiers.md).

## [2026-11-04] result | Long-prompt prefill of Qwen3.8-Flash-Next by stage, and four kernel changes

`ac11534` to `d1cff61` change four things:

- The PLE rows run on 16 threads.
- Attention runs a block for each KV group and row.
- DeltaNet normalises q and k once a token.
- Attention takes one softmax step a tile of keys.

A 15,922-token prompt went from 1,026 to 1,234 tokens per
second on an RTX PRO 6000. `39967ad` fixes `Q4E_PROFILE` for long prompts. Read
[prefill by stage](results/2026-11-04-qwen38-prefill-stages.md).

## [2026-11-04] result | One layer-major pass for the whole context

`6ddfdc7`: a pass now holds the context (in chunks of 1,024 rows, at most 32,768), not 8,192 rows. A 15,922-token
prompt copies each layer's experts once, not twice: 1,343 tokens per second on an RTX PRO 6000 (+9%) and 682 on an
A100 (+15%). A 29,120-token prompt runs at 1,414 and 746. Read [prefill by stage](results/2026-11-04-qwen38-prefill-stages.md).

## [2026-11-04] plan | MTP drafts in the CUDA engine for Qwen3.8-Flash-Next

A draft for Matt's go. On the M4 Max, mac-metal measured the P3 pack with MTP: code 39.6 to 51.8 tokens per second at
2 drafts, chat 40.1 to 43.1. The plan adds the MTP head and a state snapshot to the CUDA engine. Read
[the MTP plan](plans/2026-11-04-qwen38-mtp.md).

## [2026-11-05] result | Prefill with larger chunks and an expert prefetch

`f2caa8b` to `d8e8a63`: a 15,922-token prompt of the P3 pack went from 1,343 to 2,801 tokens per second on an RTX PRO
6000. A 29,120-token prompt runs at 2,782 there and 1,571 on an A100. The changes:

- Prompt chunks of 4,096 rows (`--prompt-rows`), and one VRAM region for the stages' buffers.
- The next layer's experts come over PCIe on a second stream (`Q4E_NO_PREFETCH=1` turns it off).
- Attention: all tile loads before the stores, padded tile rows, a wider softmax, and two blocks an SM.
- DeltaNet: the conv computes all tokens at once, and the step stages 16 tokens.

Decode did not change. Read [prefill with larger chunks](results/2026-11-05-qwen38-prefill-chunks.md).

## [2026-11-05] result | cp.async attention, merged GEMMs and a register router

`6314b52` to `470d42a`: on an RTX PRO 6000, a 15,922-token prompt of the P3 pack went from 2,801 to 3,403 tokens per
second, and a 29,120-token prompt from 2,782 to 3,423. Decode went from 44.0 to 54.2. The changes:

- Attention: Q in registers, K and V tiles by cp.async, 4 blocks an SM.
- DeltaNet: double-buffered stages by cp.async.
- Weights of one input load as one GEMM: HC down and inject, each layer's input projections, the router and the shared
  expert gate.
- The router keeps its logits in registers. QSA scores run in shared-memory tiles.

`0708c6b` changed the greedy text of the decode prompt (a near-tie on its first token). Read
[merged GEMMs](results/2026-11-05-qwen38-merged-gemms.md).

## [2026-11-05] result | One-pass prompt attention, a faster DeltaNet step, and the same quality

`d525c44` to `b93125c`: a 29,120-token prompt of the P3 pack runs at 3,660 tokens per second on an RTX PRO 6000 and
1,879 on an A100. Prompt attention runs in one pass with an online softmax, and P V runs on mma.sync. On 4,096 tokens
against the 4-bit MLX model, P3's mean KL is 0.161 (0.164 with the engine of `364b9ad`). The expert prefetch now waits
for 64 router picks an expert: below that, prompts were up to 2.6 times slower. The engine test gained a tiny model
with strong attention. Read [one-pass prompt attention](results/2026-11-05-qwen38-one-pass-attention.md).

## [2026-11-05] result | Prompt chunks of 16,384 rows on large GPUs

`957a0f0`: on GPUs of 64 GB or more, the CUDA engine takes 16,384 prompt rows a forward by default (4,096 below). On
an RTX PRO 6000, 29,120 tokens of P3 run at 4,222 tokens per second (3,606 at 4,096 rows), and 15,922 tokens at 4,165.
It costs about 3 GB of VRAM. Read [chunks of 16,384 rows](results/2026-11-05-qwen38-rows-by-vram.md).

## [2026-11-05] result | P3 against llama.cpp UD-Q3_K_XL on one RTX PRO 6000

The two packs have about the same KL against BF16 (0.0600 and 0.0593). With the experts in host memory, P3 prefills
29,120 tokens at 4,222 tokens per second and decodes at 53.8. llama.cpp (`-ot exps=CPU`) does 793 and 9.9. With
everything in VRAM, llama.cpp leads: 5,130 and 114 against 4,418 and 87.6. Read
[P3 against llama.cpp](results/2026-11-05-qwen38-vs-llamacpp.md).

## [2026-11-05] result | Faster decode norms and RoPE, and the expert copy is PCIe-bound

`0233258`: P3 decode with an 8 GB expert cache goes from 54.6 to 56.5 tokens per second on an RTX PRO 6000. Each
commit ran three times on one node, and the greedy text is the same. Two norm kernels load a thread's values together, and RoPE reads a
table of inverse frequencies in place of an f64 `pow`. The decode expert copy is 46% of the GPU time, and 32 to 1,024
blocks give the same speed. Read [faster decode norms and RoPE](results/2026-11-05-qwen38-decode-small-kernels.md).

## [2026-11-05] result | Long-context decode 34% faster, the shared expert beside the copies

`fe32045` to `d6c0606`: after a 15,975-token prompt, P3 decode goes from 43.8 to 58.8 tokens per second on an RTX PRO
6000 (8 GB cache). Decode attention splits a row's keys over 32 blocks of a KV group. The QSA top-k selects in 8-bit
passes. The shared expert runs on a second stream in decode (+2.5% at short prompts). The decode rows' KL against the
prompt rows is the same (0.0271 and 0.0272). Read [long-context decode](results/2026-11-05-qwen38-long-context-decode.md).

## [2026-11-05] result | Prompt attention over 4 rows' shared key tiles is slower

`3ef4599`, reverted by `c036709`: a block of 4 prompt rows loaded each key tile of their union once, 2.7 times fewer
loads. Attention went from 480 to 805 ms at 15,922 tokens, and from 895 to 1,600 at 29,120. The old kernel is not
bound by its loads. Read [shared key tiles](results/2026-11-05-qwen38-shared-tile-attention.md).

## [2026-11-05] engine | Qwen3.8-Flash-Next on the Metal engine

`fd49f5e`, `30810a6`: `engine/qwen4_gpu.m` runs `engine/qwen4_eng.h` on Metal, so `nslm-qwen4-gen` builds on macOS.
The weights, the routed experts included, are mapped from the folder without a copy, so the Mac needs no expert cache.
On the M4 Max the P3 pack decodes the tide-pools prompt at 38.4 tokens/s warm and 33.0 cold. mlx-vlm 0.7.12 decodes it
at 40.1 (greedy). Both runs give the same text. `test_qwen4_metal` checks the engine against mlx-vlm on the tiny model,
with the CUDA test's limits. The QSA top-k now writes its blocks in block order on Metal too, so the logits do not
depend on timing. One difference from CUDA: decode rows split differently move the logits by BF16 steps, because
Metal's `k_mv` sums one token's row in another order than several tokens' rows.

## [2026-11-05] engine | Qwen3.8-Flash-Next on Metal: faster decode and prompt, and nslm-chat

`0c51ae1`, `2a3e88d`, `5dfc9b4`, `36f84db`, `6789f01`, `8e5b9de`, `c842a26`: on the M4 Max the P3 pack now decodes the
tide-pools prompt at 42 to 43 tokens/s, against 38.4 before and mlx-vlm 0.7.12's 40.1. It reads a 5,639-token prompt at
414 to 439 tokens/s, against 288 before and mlx-vlm's 457 to 500. The router's top-k now uses a threadgroup for each
row. Matvecs with few rows and a long input (`k_mv_split`) now spread each row over a threadgroup. Prompt forwards are
4,096 rows (`--prompt-rows`). Prompt attention (`k_q4_attend_g`) treats a KV group's 12 heads as the rows of simdgroup
matrices, so each key tile is loaded once for all of them. `nslm-chat` and `nslm-serve` now reach the Metal engine
through `engine/engine.c`. `test_qwen4_metal` also checks `qwen4_tiny_attn` against the C reference.

`af4e645`: after a 19,768-token prompt the Metal engine decoded at 25 tokens/s, because its decode attention loaded
each key once for each of a KV group's 12 heads. Rows with 512 keys or more now split their keys over
threadgroups for the whole group, and a second kernel merges the splits, as CUDA's `5ce032e` does. Decode there is
36 to 38 tokens/s.

## [2026-11-05] engine | The MTP head in the CUDA engine (plan steps M1 and M2)

`d0a227e`, `9e3fb7a`, `3e33894`, `1362d13`, `ecd552f`: the C reference has the head (`qwen4_ref_mtp`). It matches
mlx-vlm 0.7.12 to 5.7e-7 on the tiny model, with the same 3 drafts (gate MA). The CUDA engine loads it with
`Q4EOpts.mtp`. Every forward runs it one row behind, and stores only its keys. `q4e_draft` makes up to 3 greedy
drafts in one CUDA graph. `q4e_forward_keep` verifies them in one forward, and the kernels write the recurrent state
after each row. So `q4e_accept` keeps the accepted rows with a copy of about 115 MB, not a second forward.
`test_qwen4_mtp` checks gate MB on the tiny model: greedy text with 1 to 3 drafts equals greedy text without.
`nslm-qwen4-gen --drafts K` turns it on. The speed on the RTX 4080 is plan step M3. Metal has the calls as stubs
(`engine/qwen4_nomtp.c`) until plan step M4.

## [2026-11-05] result | MTP drafts on the RTX 4080: the same text, not faster

`ecd552f`, pc-cuda's RTX 4080, P3, 6 GB expert cache. Greedy text with 0 to 3 drafts has the same md5 on a code
prompt and on a chat prompt (gate MB). Code decodes at 29.7 tokens per second without drafts and 27.6 to 28.9 with drafts. Chat
decodes at 29.2 without drafts and 22.7 to 28.7 with drafts. Gate MC fails, so drafts stay off by default. A step of T
rows takes about 10 ms + 25 ms x T: each verify row pays its expert copies and its expert matvec again. Read
[MTP drafts on the RTX 4080](results/2026-11-05-qwen38-mtp-4080.md).

## [2026-11-05] result | The RTX 4080 PC: CPU seed rate and streamed experts

`35f17e0`, pc-cuda's PC. On 16 threads, the Ryzen 7 5800X3D computes P3 seed experts at 20.7 G weights per second for
one token. For each token of 4 tokens, it computes 51.8. A CPU share of the missed experts may save at most about 6 ms of 34 ms a
token, so the CPU expert path stays on hold on this PC. Streamed experts decode P3 at 14.7 to 19.8 tokens per second
with a warm page cache, about half of pinned. Read
[the CPU seed rate and streamed experts](results/2026-11-05-qwen38-4080-cpu-and-streamed.md).

## [2026-11-05] engine | PLE rows on a thread pool, and the expert cache keeps the VRAM reserve

`d946eab`: decode read its 16 PLE rows on one thread after a WILLNEED prefetch. On the RTX 4080 under Windows, the
prefetch took 4.2 ms a token. A verify forward of 3 rows took 42 ms when the prefetch did not hold the pages. Now each
(row, head) is an item on a persistent pool of `Q4E_PLE_THREADS` threads, so the page reads overlap. The speed on the
4080 is not measured yet. `35f17e0`: an explicit `--expert-cache-gb` did not look at `NSLM_VRAM_RESERVE_MB`. With
cache 8, the display driver of the 4080 that drives the desktop failed, and the screen went black while the runs went
on. Now the cache shrinks to keep the reserve. The reserve is 2560 MB on a GPU that drives a display, else 512 MB.

## [2026-11-05] engine | MTP drafts on Metal

`9b15787`, `9594126`, `fe89c07`: plan step M4. The Metal engine has the MTP head, `q4e_draft`, `q4e_forward_keep` and
`q4e_accept`, as CUDA's `ecd552f` and `7cfd557` do, so `nslm-qwen4-gen --drafts` works on macOS. The verify rows write the
recurrent state after each row to snapshot buffers, and `q4e_accept` swaps the kept row's buffers in, so a failed
draft copies nothing. Read [MTP drafts on the M4 Max](results/2026-11-05-qwen38-mtp-m4max.md) for the current
numbers. On the M4 Max, P3, greedy, 256 tokens at most (tokens/s, two runs within 1%):

| Prompt | No drafts | 1 draft | 2 drafts | 3 drafts |
|---|---|---|---|---|
| Code | 41.1 | 50.1 | 52.2 | 50.6 |
| Chat | 42.0 | 47.4 | 43.8 | 39.7 |

Gate MB holds: the text is byte-identical at every K, on both prompts and after a 20,131-token prompt (35.9 tokens/s
without drafts, 38.4 with 2). It holds by construction. The matvecs now run without fast math, so a token's row sums in
the same order for any row count. Each decode row's attention kernel and key splits now follow from its own keys, not
from the longest row of its forward. A 4-row forward then gives the logits of four one-row forwards bit for bit, after
20 to 2,400 tokens. Before the attention change, 4 rows that straddled 512 keys or a 128-key split boundary changed
about 240,000 logits a row. mlx-vlm 0.7.12's own MTP reached 51.8 (code) and 43.1 (chat) at 2 drafts.

## [2026-11-05] kernel | Metal matvecs: verify rows 20 to 35% cheaper, and the seed stream table in halves

`39cef39`: k_mv with Q4 and 2 to 8 tokens runs 20 to 35% faster, so each MTP verify row costs less. On the M4 Max,
P3, greedy, at most 256 tokens (tokens/s, two runs within 0.5%):

| Prompt | No drafts | 1 draft | 2 drafts | 3 drafts |
|---|---|---|---|---|
| Code | 40.9 | 53.8 | 56.0 | 54.2 |
| Chat | 42.0 | 50.6 | 47.0 | 42.4 |

Before, it was 50.1 / 52.2 / 50.6 (code) and 47.4 / 43.8 / 39.7 (chat). The text is the same at every K, and 2- to
4-row verify forwards still equal one-row forwards bit for bit on P3. The speedup is an effect of Apple's compiler,
found by A/B and not understood: the multi-token loop compiles faster with a never-taken path that reads x's block
sums from threadgroup memory. A plain barrier, the select alone, or limits on unrolling do not reproduce it.

The LFSR stream tables are linear over GF(2): G[s] = G[s & 255] ^ G[s & 0xFF00] for every seed. So the seed matvecs
read two 256-entry halves from threadgroup memory, not a 256 KB table: +2% decode on P3, bit for bit.

A simdgroup-MMA matvec (8 rows x 8 tokens a simdgroup) was tried first. It is T-invariant by construction and flat
from 1 to 8 tokens, but 1.4 to 1.5 times slower than k_mv at one token, so plain decode would lose.

## [2026-11-05] kernel | Metal decode: the DeltaNet step and short-context attention

`c24a34a`, `44d0dab`, `2374087`: P3 decode on the M4 Max goes from 41.3 to 43.3 tokens/s without drafts, and from 56.2
to 59.6 with 2 drafts (code prompt, two runs each). After a 20,131-token prompt it goes from 34.6 to 35.6. `Q4E_PROFILE=3`
times each decode dispatch by kernel. It showed two latency-bound kernels at one token:

- The DeltaNet step held each value row's state in 2 lanes. With 4 lanes a row it runs twice the threads: 43 -> 24 us
  a layer.
- `k_q4_attend` took 70 us a layer at about 100 keys. `k_q4_attend_g` with its key splits takes 16, so every decode row
  now uses it, not only rows of 512 keys or more. The merge kernel runs only when a row has more than one split.

Both change the order of some sums, so the text changes at the rounding level. The kernel tests stay within 4e-7 of
the C reference. MTP verify rows still equal decode rows bit for bit, so drafts do not change the text.

Tried and dropped: the router top-k in one simdgroup (13.5 -> 18.5 us a call, 46 with a sorting network). The
distinct-expert pass for verify rows was slower (51.5 against 56.9 tokens/s at 2 drafts), with the same text.

## [2026-11-05] kernel | Metal decode and prompt: smaller kernels, and a 64 x 64 prompt GEMM

`a5d5ea2`, `8c33e1e`, `e5ae64d`: P3 decode without drafts on the M4 Max goes from 43.3 to 44.7 tokens/s (code prompt).
- The experts' down projection (80 blocks a row) runs two rows a simdgroup, 16 lanes each: 56 -> 45 us a call.
- `k_mv_split` takes Q4 in MLX's qmv form, as `k_mv` does: 6.9 -> 5.9 us a call.
- The hyper-connections' up projection (40 blocks a row) runs four rows a simdgroup.

`2a83c97`, `6b284fc`: prompts. `Q4E_PROFILE=4` times prompt rows by kernel, and it showed the two GEMMs at 74% of a
5,607-token prompt, at about 9.4 TFLOPS. `k_mm64` uses 64 x 64 tiles, each simdgroup 32 x 32 with its loops fully
unrolled (without that, 1.6 TFLOPS). The dense Q4 GEMM goes from 9.4 to 11.6 TFLOPS. The grouped expert GEMM uses
tiles of 64 pairs, so each seed block expands fewer times: 4.61 -> 3.99 s a prompt. y is bit for bit `k_mm`'s, so the text
is the same. A 5,607-token prompt goes from 445 to 507 tokens/s (mlx-vlm 0.7.12: 457 to 500), and a 20,131-token prompt
from 284 to 493.

The decode changes reorder some sums, so the text changes at the rounding level. MTP verify rows still equal decode
rows bit for bit.

Tried and dropped: weight fragments built in registers with no threadgroup memory (8.7 TFLOPS), and 128 x 64 tiles over
8 simdgroups (10.7).

## [2026-11-05] kernel | Metal decode: the router, norm1p and short-context attention

`4360e3e`, `40c2393`, `c2dbb47`: P3 decode without drafts on the M4 Max goes from 44.7 to 47.5 tokens/s
(code prompt). The router's top k takes two phases: each simdgroup takes its own, then one merge, with a single
barrier (13.5 -> 7.6 us a call, the same experts and weights). `norm1p` at a decode row lets up to 8 threadgroups share
a group, each taking the whole sum in the same order (8.8 -> 6.5 us, bit for bit). Decode attention now splits a row's
keys every 32, at most 16 splits. A row of one split walked its tiles serially (86 us a layer at 100 keys, 39 in four
splits). The first two are bit for bit. The split change moves the text at the rounding level, and MTP verify rows
still equal decode rows.

## [2026-11-05] result | PLE rows on a thread pool: P3 decode 9% faster on the RTX 4080

`3f570f3`, pc-cuda's RTX 4080, P3, pinned experts, 6 GB cache: decode is 33.17 tokens per second against 30.30, with the
same text. The host PLE prefetch took 1,075 ms of a 256-token run, and now takes 2 ms. The automatic cache picks about
8.1 GB beside the 2,560 MB display reserve, and decodes at 37.69 tokens per second. With 2 drafts, the PLE rows took
4,315 ms on `ecd552f` and take 403 ms now. So the MTP measurement on the 4080 gets a rerun. `3f570f3` also drops the
experts' file pages after the pinned copy: the load peak fell from more than 111 GB to 59.7 GB. Read
[PLE rows on a thread pool](results/2026-11-05-qwen38-4080-ple-pool.md).

## [2026-11-05] result | MTP drafts on the RTX 4080, again: the same text, slower

`3f570f3`, pc-cuda's RTX 4080, P3, the automatic cache. Greedy text with 0 to 3 drafts has the same md5 on both
prompts. Code decodes at 37.6 tokens per second without drafts and 29.0 to 32.5 with drafts. Chat decodes at 38.9
without drafts and 24.1 to 32.7 with drafts. The head takes about 1.9 GB of the cache, and each verify row pays the
PCIe copies of its missed experts. Drafts stay off by default on CUDA. On the M4 Max they make code 31% to 37% faster.
Read [MTP drafts on the RTX 4080, again](results/2026-11-05-qwen38-mtp-4080-rerun.md).

## [2026-11-05] result | Strata on the RTX 4080 PC: why its MTP drafts pay

pc-cuda's PC, Strata with Matt's IQ3_S configuration: 85 tokens per second with MTP drafts and 57 without. A Strata step
of T rows takes about 10.5 ms + 7.1 ms x T, and ours about 9 ms + 25 ms x T. Strata misses fewer experts (2.2 a layer
against our 3.5), and computes most misses on the CPU. Each of our rows copies its misses over PCIe. Read
[Strata on the RTX 4080 PC](results/2026-11-05-strata-4080-probe.md). The plan
[decode on the RTX 4080 near Strata's speed](plans/2026-11-05-qwen38-decode-4080.md) follows from it.

## [2026-11-05] result | P3H, the MTP head with P = 3 seed experts

P3H is P3 with its MTP head's routed experts as P = 3 seeds. The CUDA (RTX 4080) and Metal (M4 Max) head searches gave
the same seeds. On the RTX 4080 the head joins the expert cache (plan step D1, `c7a58b0`). Draft runs then keep 3,008 to
3,101 slots, not 2,368 to 2,461, and code with drafts is 7.7% to 9.2% faster. Chat gains 0.4% to 4.0%. Drafts still lose
to no drafts on the 4080, and gate DA passes at 1 draft only. On the M4 Max, P3H is not faster. Read
[P3H](results/2026-11-05-qwen38-p3h-head.md).

## [2026-11-05] fix | D4: the CPU tail rounds as the GPU does

`4997a53`: the RTX 4080's first gate DC run of `--cpu-tail 7` gave a 13% higher decode-row KL. The GPU expert path rounds
gate, up, SwiGLU and the down output to BF16, and the CPU tail did not. Now it does. The engine test could not see the
error, because its seed experts were about 1e-5 of the residual. Its experts are now as large as the residual
(`38ef15e`). On an RTX PRO 6000, the first decode step then equals the GPU's bit for bit. After 20 steps the logits
differ by 1.1e-3, against 9.6e-2 with the tail dropped. A thrashing cache equals every expert in VRAM at all 20 steps.

## [2026-11-05] kernel | Metal long-context decode: the QSA top-k

`a00dbde`, `0a6f0a9`: after a 20K-token prompt, P3 decode on the M4 Max goes from about 40.5 to 45.2 tokens/s. At that
context the QSA top-k (512 of about 5,000 blocks a row) took 206 us a call, 10% of decode. Most of it was the end,
where one thread walked back from the last block for the scores equal to the k-th. That pass now runs in parallel,
and the k-th score comes from four 8-bit histogram passes, as on CUDA: 49 us a call, with the same blocks in the same
order. With that fixed, up to 32 key splits a row (was 16) are faster at 1K to 20K tokens (47.7, 45.3 and 45.2
tokens/s at 1K, 5.6K and 20K). MTP verify rows still equal decode rows bit for bit.

## [2026-11-05] result | MTP drafts on the M4 Max, again

`e53d81c`: after the day's Metal decode kernels, P3 decodes code at 61.5 tokens per second with 1 draft, against 48.9
without drafts, and chat at 57.5 against 47.2. The text is the same at every K. Read
[MTP drafts on the M4 Max, again](results/2026-11-05-qwen38-mtp-m4max-rerun.md).

## [2026-11-05] result | The tail skip and the CPU tail on an RTX PRO 6000 at the 4080's cache

HF jobs `6aece14c` and `6aece564`, an 8.1 GB cache. The tail skip (D3) fails gate DL at R = 8 and 9 (KL 0.073 and 0.053
against 0.036). It failed at R = 7 on the RTX 4080 too. Renormalized weights make it worse (0.123 and 0.088 at 7 and 8). The CPU tail (`--cpu-tail 7`, with the
BF16 fix) gives 0.030: rounding noise, not a gain. It halves decode on this machine, 66.5 to 33.4 tokens per second.
The hand-off adds about 30 us. One P3 expert takes 1.6 ms of one core, and the GPU's own work to overlap is about
150 us a layer. `068617c` caps the CPU at N experts a layer (`--cpu-tail N`). Read
[The tail skip and the CPU tail on an RTX PRO 6000](results/2026-11-05-qwen38-tail-hf.md).

## [2026-11-05] engine | MTP drafts in nslm-chat

`72316d9` and `ba71ba8`: `Q4E_DRAFTS=K` opens the MTP head in `engine/qwen4_api.c`, and `eng_generate` verifies up
to K drafts a forward. It drafts on a pending token when that token is the arg max of the logits it holds, so its calls
can be short. At temperature 0, `nslm-chat` takes the tokens after the first from `eng_generate`, K + 1 a call. The
text does not change: `test_qwen4_api_mtp` checks the tokens, the history and the next logits against K = 0, bit for
bit. Drafts stay off by default.

## [2026-11-05] result | MTP drafts in nslm-chat on the M4 Max

`ba71ba8`: P3 decodes code at 69.6 tokens per second with 2 drafts, against 47.8 without drafts (+46%). A short chat
answer is 8.5% faster with 1 draft, and 0.5% and 13% slower with 2 and 3. The text is the same at every K. Read
[MTP drafts in nslm-chat on the M4 Max](results/2026-11-05-qwen38-mtp-chat-m4max.md).

## [2026-11-05] result | Trellis-coded experts: weight error against ggml's types

The probe used layer 20's experts, a bitshift trellis with QTIP's computed code, and an imatrix-weighted Viterbi search.
At 3.11 bits the weighted error is 0.0158, against 0.0182 for IQ3_S at 3.44 bits. At 4.11 bits it has 0.0042, against 0.0057 for
IQ4_XS at 4.25 bits. At 2.12 bits it equals IQ2_S at 2.56 bits. It saves about 0.4 to 0.5 bits a weight, and it is
close to the bound for Gaussian weights. Read [Trellis-coded experts: weight error](results/2026-11-05-trellis-probe.md).

## [2026-11-05] result | The CPU tail on the RTX 4080: slower at every setting

`cb3d6b6`, pc-cuda's RTX 4080, P3, the automatic cache: `--cpu-tail 1` and `2` decode at 31.9 and 26.9 tokens per
second, against 37.2 without. The CPU removes 22% and 41% of the misses. But one CPU expert takes 240 to 340 us, more
than the PCIe 4 copy it saves. Plan step D4 closes. Read
[The CPU tail on the RTX 4080](results/2026-11-05-qwen38-cpu-tail-4080.md).

## [2026-11-05] result | Trellis-coded experts against BF16: KL in llama.cpp

`7e442c7`: `nslm-moe --codec tq` runs the CUDA encoder. On 8 RTX PRO 6000s it encodes all 121 G expert weights at
L = 12 in about 5 minutes a K. At K = 4 (4.125 bits, 62.3 GB) the KL is 0.0398, against P4's 0.0408 at 4.5 bits, and 16% below
Unsloth's curve. At K = 3 (3.125 bits, 47.2 GB) it is 0.0765, about 4% below an extrapolation of the curve. A Q8_0 PLE
table changes the KL by 0.00005. Read [Trellis-coded experts against BF16](results/2026-11-05-trellis-kld.md).

## [2026-11-05] result | Trellis packs at 2 to 4 bits, and Unsloth's curve corrected

`7e442c7`: Unsloth's expert sizes were too small on earlier pages, because a new GGUF type counted as 0 bytes. On
the corrected curve, the trellis is 16% below it at 3.125 bits and 17% below at 4.125 bits. K = 4 for `down_proj` and
K = 3 for the rest gives 0.0619 at 3.46 bits, 10% below. K = 2 gives 0.204 at 2.13 bits. Read
[Trellis packs and Unsloth's curve](results/2026-11-05-trellis-mixes.md).

## [2026-11-05] result | Trellis experts on the RTX 4080: faster than P3 at P3's quality

`54a7130`, pc-cuda's RTX 4080, the automatic expert cache: trellis experts with `down_proj` at K = 4 and the rest at
K = 3 decode at 49.7 tokens per second, against P3's 39.1 (+27%). Prompts are 22% faster, and the KL is about P3's
(0.0619 against 0.0600). K = 3 is 43% faster at a KL of 0.0765, and K = 4 is 8% slower. Read
[Trellis experts on the RTX 4080](results/2026-11-05-trellis-speed-4080.md).

## [2026-11-05] result | Trellis experts on the M4 Max: P3's speed, better quality or less memory

`6f90957`, mac-metal's M4 Max: every trellis pack decodes 1.7% to 2.3% faster than P3 without drafts, and reads an
8,506-token prompt at P3's speed. All experts stay in unified memory, so there is no PCIe copy to save. Trellis K = 4
has a KL of 0.0398 against P3's 0.0600 for 1.9 GB more, and K = 3 needs 13.2 GB less. Read
[Trellis experts on the M4 Max](results/2026-11-05-trellis-speed-m4max.md).

## [2026-11-05] engine | The VRAM reserve of a display GPU: 1,024 MB

The expert cache now keeps 1,024 MB free on a GPU that drives a display, not 2,560 (`engine/vram.h`). On pc-cuda's
RTX 4080 that gives about 20% more slots and 8% to 9% faster decode, for P3 (42.2 tokens per second) and the trellis
packs (the mix 54.4, K = 3 61.3). 512 MB is slower, because WDDM pages VRAM. Read
[The VRAM reserve on the RTX 4080](results/2026-11-05-vram-reserve-4080.md).

## [2026-11-05] result | Trellis K = 3 on the M4 Max: the pages that a session uses

`39afe70` (`tools/page_resident.py`), mac-metal's M4 Max: a session of about 13,000 tokens keeps about 55 GB of the
K = 3 pack in the page cache. That is 47.2 GB of banks, 3.1 GB of other tensors and 5.0 GB of the 32 GB PLE table.
The PLE part grows by about 0.5 MB for each generated token. A cold PLE table makes decode about 10% slower. So K = 3
may fit a 64 GB Mac, but no 64 GB Mac is measured. Read
[The pages that a session uses](results/2026-11-05-trellis-ple-working-set-m4max.md).

## [2026-11-05] plan | Trellis-coded dense weights

A plan to code the dense weights with the trellis, with more bits for a kind only where its KL needs them. Every KL of
the trellis packs so far uses Unsloth's Q8_0 dense weights. The engine runs the 4-bit MLX ones, and their KL is not
measured, so step D0 measures the mix as shipped. On the M4 Max the dense matvecs take 8.77 ms of an 18.7 ms decode
token. Read [Trellis-coded dense weights](plans/2026-11-05-trellis-dense.md).

## [2026-11-05] plan | An LFSR code for the trellis

Matt asked whether SeedLM has a future beside the trellis. The trellis packs replace every seed expert, and they are
ahead on each axis that we measured. But the trellis also computes its code from a pseudo-random function of its
state. This plan tests the trellis with NanoSeedLM's LFSR as that function, on KL and decode speed. At L = 12 the
LFSR code had the weighted error of 1MAD within 1%. Read [An LFSR code for the trellis](plans/2026-11-05-lfsr-trellis.md).

## [2026-11-05] result | The LFSR code against 1MAD at L = 12

On the trellis probe's rows, at L = 12, the LFSR code has the weighted error of 1MAD within 0.6% at K = 3. At K = 4 it
is within 0.2% for `gate_proj` and `up_proj`, and 1.5% higher for `down_proj`. Step L2 of the LFSR plan. Read
[The LFSR code against 1MAD](results/2026-11-05-lfsr-trellis-probe.md).

## [2026-11-05] result | MTP drafts with trellis experts on the RTX 4080

`7e42e52`, pc-cuda's RTX 4080: `ns_add_banks` now takes a bank L48 as the MTP head's. With its experts in trellis
banks, the head joins the expert cache. Drafts then cost 236 to 301 slots, not 976 to 1,120, and they break even.
The mix is 5% faster with one draft and 1% slower with two. K = 3 is 2% slower with two. Before, drafts were 6% to 12%
slower. Drafts stay off by default on CUDA. Read [MTP drafts with trellis experts](results/2026-11-05-drafts-trellis-head-4080.md).

## [2026-11-05] result | Trellis codes from tables on the M4 Max

mac-metal's kernel bench: a 4,096-value table in threadgroup memory makes the trellis gathers slower than the computed
1MAD (84 against 59 us a call). An LFSR code through two 64-value tables takes 95 us, about -15% in decode. So gate LA
of the LFSR plan fails on Metal. Read [Trellis codes from tables](results/2026-11-05-table-code-metal.md).

## [2026-11-05] plan | The trellis family on Hugging Face

Matt approved a public release of the trellis packs. The plan has three repositories: TQ4, TQ3-D4 (`down_proj` at
K = 4) and TQ3. Each holds P3's dense tensors, PLE table and head, without P3's seed experts, plus the trellis banks
and a card with the KL as shipped. Release 1 runs in the NanoSeedLM engines. A start script and the MLX loader follow.
Read [The trellis family on Hugging Face](plans/2026-11-05-release-family.md).

## [2026-11-05] result | Packs as shipped: the 4-bit MLX dense weights triple the KL

`7eb5d9b`, HF job `6aed6158`: P3's 4-bit MLX dense weights took the place of Unsloth's Q8_0 ones. Then the KL
against BF16 is 0.182 for P3 and 0.185 for the trellis mix, not 0.060 and 0.062. Top-1 agreement falls from 91.5% to 84%. Unsloth's
3-bit packs keep their dense weights at Q6_K or Q8_0. So release 1 takes 8-bit dense weights. Read
[Packs as shipped](results/2026-11-05-dense-as-shipped.md).

## [2026-11-05] result | The release packs as shipped

HF jobs `6aed6ff3` and `6aed75a2`: the three release folders are in the bucket. They hold P3's tensors without its seed
experts, oQ8e's 8-bit dense modules and the trellis banks. Their KL against BF16 is 0.0415 (TQ4), 0.0638 (TQ3-D4) and
0.0778 (TQ3). TQ3 gives the same text in two runs on CUDA. Read [The release packs as shipped](results/2026-11-05-release-as-shipped.md).

## [2026-11-06] plan | Dense plan: the 8-bit dense weights are the reference, and D2a ranks the kinds

`8856cc4`: release 1 ships 8-bit dense weights, and oMLX's oQ packs (oQ2.5e to oQ8e) keep all dense tensors at 8 bits.
So gate DB of the dense plan now compares with the 8-bit dense weights. These read about 4.5 GB a decode token. Step
D2a puts one more dense kind at a time in P3's 4-bit MLX format over Q8_0 (`dense:KIND` in `tools/hf_llama_kld.sh`).
The KL job waits for the next weekly budget. Read [Trellis-coded dense weights](plans/2026-11-05-trellis-dense.md).

## [2026-11-06] result | The release packs from the Hub on the RTX 4080

HF job `6aeda68a` uploaded the three packs (private). `hf download` of TQ3-D4 and TQ3 gives the same text in two
runs on the RTX 4080. The 8-bit dense weights cost speed there: TQ3-D4 decodes at 38.0 tokens per second, against 54.4
with 4-bit dense weights, because the expert cache has 23% fewer slots. `00918f0` keeps the token embedding in host
memory: 39.6 tokens per second, the same text. Read
[The release packs from the Hub on the RTX 4080](results/2026-11-06-release-downloads-4080.md).

## [2026-11-06] result | VRAM that grows with the context on the RTX 4080

`354e2e1`: at 32K positions the RTX 4080's expert cache had 2,530 slots, not 4,431. The layer-major pass kept 1.3 GB.
Passes of at most 8,192 rows give 474 more slots, decode +8% and prompts +43% to +47%, with the same text. That is now
the default under 32 GB of VRAM. Read [VRAM that grows with the context](results/2026-11-06-context-vram-4080.md).

## [2026-11-06] plan | The KV cache of Qwen3.8-Flash-Next in BF16

Both engines keep the keys and values of the 12 attention layers in f32: 48 KB a position, about 1.6 GB at 32K. The
plan stores them in BF16, as MLX does, and gives the RTX 4080's expert cache about 300 more slots at 32K. Read
[The KV cache in BF16](plans/2026-11-06-qwen4-kv-bf16.md).

## [2026-11-06] result | The release packs from the Hub on the M4 Max

`hf download` of TQ4 stalled on read timeouts on the Mac. curl for each file, as `start-here.sh` does, finished the
download at 119 MB/s. TQ4 gives the same text in each run: 38 to 42 tokens per second in `nslm-chat`, 67 to 69 with 2
drafts. oMLX loads it with the trellis loader: 49 tokens per second, and 81 to 84 with MTP. The v0.1.0 macOS archive
passes gate RE. `373499f` fixes the pack choice of the start scripts on 64 GB PCs and 96 GB Macs. Read
[The release packs from the Hub on the M4 Max](results/2026-11-06-release-downloads-m4max.md).

## [2026-11-06] result | The KV cache of Qwen3.8-Flash-Next in BF16

`bb40b77`, `20666ef`, `b8dc128`: both engines keep the attention keys and values in BF16. On the RTX 4080 at 32K the
expert cache has 379 more slots and decode is 5.7% faster. On the M4 Max a 15K prompt is 5% to 6% faster. The KL of
the BF16 cache against the f32 cache is at or below a half-step noise floor. Read
[The KV cache in BF16](results/2026-11-06-qwen4-kv-bf16.md).

## [2026-11-06] result | The KL of the 4-bit PLE table

`4373d30` adds `nslm-qwen4-emulate --ple`. In HF job `6aee1d9d`, the 4-bit MLX PLE table that the packs ship gives a KL
of 0.0259 on Q8_0. Unsloth's Q8_0 table gives 0.0199. `ff92415` makes the start scripts download the Q8_0 table
by default. Read [The KL of the 4-bit PLE table](results/2026-11-06-ple-table-kld.md).

## [2026-11-06] release | The trellis packs are TCQ4, TCQ3-D4 and TCQ3

The three repositories moved from `-TQ4`, `-TQ3-D4` and `-TQ3` to `-TCQ4`, `-TCQ3-D4` and `-TCQ3`, at the same
revisions. TCQ is trellis-coded quantization. In oMLX and in the community, TQ means TurboQuant. The Hub redirects the
old names. The start scripts, the cards and the current pages use the new names. The dated results pages keep the
names of their measurements.

## [2026-11-06] plan | MTP drafts in nslm-serve, with sampling

`nslm-serve` does not draft: `eng_generate` keeps only greedy drafts, and the server samples. The plan makes the draft
rule speculative sampling, with one sampler for `nslm-chat` and `nslm-serve`, and the MTP head's first input from the
caller. Read [MTP drafts in nslm-serve](plans/2026-11-06-serve-drafts.md).

## [2026-11-06] result | Long decode and long prompts on the RTX 4080

The model cards get long runs in place of 256-token decodes. On the stock path of v0.1.0 (the Q8_0 PLE table, 32K
context), the RTX 4080 decodes 4,096 tokens at 44 (TCQ3), 38 (TCQ3-D4) and 31 (TCQ4) tokens per second. After 30K
tokens of prompt it decodes at 42, 36 and 30. Read [Long decode on the RTX 4080](results/2026-11-06-long-bench-4080.md).

## [2026-11-06] plan | MTP drafts that the sampler keeps more often

At temperature 1.0 on the RTX 4080, 2 drafts in `nslm-serve` keep 38% to 46% and make decode 16% slower (gate SC). The
plan takes Strata's rule: drafts sampled from the head's distribution, kept with min(1, p / q), and a confidence gate.
Cheaper verify rows need a plan of their own. Read [Drafts the sampler keeps](plans/2026-11-06-draft-acceptance.md).

## [2026-11-06] release | NanoSeedLM engine v0.1.0

Matt published [v0.1.0](https://github.com/mbarnson/NanoSeedLM/releases/tag/v0.1.0) as a pre-release. The tag is
`47714b8`. It has `nslm-chat` and `nslm-serve` for Windows x64 and Linux x64 (CUDA 12.8) and macOS (Metal, signed and
notarized), with the start scripts. The scripts download a TCQ pack and Unsloth's Q8_0 PLE table. The stock path passes
on the M4 Max (tarball `003648f3`) and on the RTX 4080 (zip `6becc62c`). Read [the release
plan](plans/2026-11-05-release-family.md).

## [2026-11-06] result | Long decode and long prompts on the M4 Max

On the stock path of v0.1.0, the M4 Max decodes 4,096 tokens of TCQ4 at about 39 tokens per second, and also after
30,000 tokens of prompt. Warm prompts run at about 480. The Q8_0 PLE table costs 0% to 4%: the host reads its rows on one
thread. Read [Long decode on the M4 Max](results/2026-11-06-long-bench-m4max.md).

## [2026-11-06] release | The TCQ repositories get their MLX model files

The three TCQ repositories shipped without `nanoseedlm_qwen4.py`, `nanoseedlm_seeds.py`, `nanoseedlm_trellis.py`, the
`model_file` key and the LICENSE. So mlx-vlm and oMLX had nothing to load with Trust Remote Code. The packer learned to
write them (`d56a17c`) after the release folders were built, and R5 uploaded the folders without a rebuild. The files
are on the Hub now (TCQ4 `e53ffdd2`, TCQ3-D4 `30012655`, TCQ3 `d1353e4a`). They are the `17eeb5d` loaders that oMLX ran
on TQ4's download. `tools/hf_release_publish.sh` now refuses a folder without them. The v0.1.0 start scripts pin the
earlier revisions.

## [2026-11-06] result | MTP drafts in nslm-serve on the RTX 4080

Drafts sampled from the head's distribution keep 63% of 2 drafts at temperature 1.0, against 38% to 46% before. The
confidence gate raises that to 77% to 84%. Decode is break-even, within 2%. Strata decodes the same request at 71 tokens
per second, against 38, mostly by decode without drafts. Read [the gates](results/2026-11-06-draft-gates-4080.md).

## [2026-11-06] result | TCQ3 on a 12 GB GPU

On the RTX 4080 with 4 GB more of VRAM kept free, TCQ3 decodes at 27 tokens per second, against 44 with all 16 GB. Read
[TCQ3 on a 12 GB GPU](results/2026-11-06-tcq3-12gb.md).

## [2026-11-06] result | MTP drafts in nslm-serve on the M4 Max

With A1m, A2m and the merged projections (`b911cb5`), 2 drafts at `Q4E_DRAFT_MIN_P=0.3` make TCQ4 decode 32% faster at
temperature 0, with the same text, and 16% faster at 1.0. The start script of v0.1.1 turns them on for a Mac. Read
[the gates on the M4 Max](results/2026-11-06-draft-gates-m4max.md).

## [2026-11-06] result | The PLE rows on a host pool on the M4 Max

mac-metal's pool for the PLE rows (`c9fcca7`) gives the same text and no speed beyond the noise, so it does not join
v0.1.1. The Q8_0 table's difference with drafts comes from its draft counts, not from row reads. Read
[the PLE rows on a host pool](results/2026-11-06-ple-pool-m4max.md).

## [2026-11-06] plan | An nslm-serve that a stranger cannot use

pc-cuda found that any web page could use the server: replies allowed every origin, and there was no key or Host
check. The plan adds an API key off loopback, a loopback Host, origins only by `--allow-origin`, a sanitizer job and
hardening flags for v0.1.1. Fuzzing and CodeQL follow. Read [the plan](plans/2026-11-06-server-hardening.md).
