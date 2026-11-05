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

- `d5cd048`: the PLE prefetch. `q4e_forward` asks the OS to read all PLE rows of the call first (`madvise` WILLNEED).
  The PLE stage goes from 5,415 ms to 4 ms. The logits do not change.
- `cafb3a2`: prompt rows of more than 8 run the routed experts as grouped GEMMs (`kc_bucket`, `kc_mm_grouped_dev`). The
  expert stage goes from 3,820 ms to 1,118 ms. `kc_bucket` now takes up to 1,024 experts.

The grouped GEMMs round their inputs to BF16. The KL against the 4-bit model went from 0.179 to 0.176. The reference's
perplexity went from 2.364 to 2.402, so the rounding cost is not known yet. The router and expert cache stage (4.2 s) is
now 57% of the time. Read
[The prompt time of Qwen3.8-Flash-Next on the CUDA engine, by stage](results/2026-11-04-qwen38-prompt-profile.md).

## [2026-11-04] result | Prompt rows of 1,024 and a decode expert prediction for Qwen3.8-Flash-Next on CUDA

- `15b4fab`: forwards without logits take 1,024 rows. On an RTX PRO 6000 with an 8 GB cache, a 3,703-token prompt of P4
  goes from 151 to 541 tokens per second. The cache misses go from 274k to 64k.
- `1cd7c57`, removed in `09e04cd`: decode predicted the next layer's experts and copied them on a second stream. The
  misses went down by 44%, but decode went from 40.6 to 39.0 tokens per second (P3).
- `6fef57c`: QSA writes its selected blocks in block order. The atomic order made the logits differ from run to run
  past position 2,048. Now 128 and 1,024 rows give the same text.

Read
[Prompt rows of 1,024 and a decode expert prediction](results/2026-11-04-qwen38-prompt-rows-and-prediction.md).

## [2026-11-04] engine | CUDA graphs for Qwen3.8-Flash-Next decode

`03e10ff`: decode rows (up to 8) of the qwen4 engine run as CUDA graphs. The kernels read the position from device
memory, so one graph serves every position. On an RTX PRO 6000 with an 8 GB cache, P3 decodes at 46.5 tokens per second,
against 42.1 without graphs. The text is the same. `NSLM_NO_GRAPH=1` turns the graphs off. Read
[CUDA graphs for Qwen3.8-Flash-Next decode](results/2026-11-04-qwen38-cuda-graphs.md).

## [2026-11-04] result | Expert cache policies for Qwen3.8-Flash-Next decode, from a routing trace

A routing trace of 1,024 decode tokens of P3 (`Q4E_ROUTES`) went through `tools/qwen4_cache_sim.py`. At 3,255 slots
(8 GB), the engine's LRU serves 74.6% of the expert uses, LFU 67.4% and a static hot set at most 62.8%. Belady's
optimum serves 87.6%. So the plan's hot set (S8, V3) does not help, and only a prediction of future routes can reduce
the misses much. Read
[Expert cache policies for Qwen3.8-Flash-Next decode](results/2026-11-04-qwen38-cache-policies.md).

