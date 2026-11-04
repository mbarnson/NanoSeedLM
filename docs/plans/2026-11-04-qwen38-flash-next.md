# Plan: SeedLM packs for Qwen3.8-Flash-Next (2026-11-04)

**Goal.** Make a family of SeedLM packs for
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). On a PC with one 16 GB GPU and 128 GB
of DDR4, one pack must decode faster than a published quantization of similar quality. A second target is a Mac with
128 GB of unified memory. The user chooses the precision of the PLE table to fit the free disk space.

**Status.** Draft. No gate is measured yet. S1 items 1 to 3 and S2's speed measurement are done. S3 runs. A first probe for H3 is done:
[Hadamard rotation and P = 4 seeds](../results/2026-11-04-qwen38-rotation-probe.md). Each gate gets a results page
when we measure it.

Qwen3.8-Flash-Next keeps most of its weights in 24,576 small experts, and each token uses 480 of them. On a PC, most
experts stay in RAM. The speed of decode then depends on how many bytes of experts come from DRAM for each token.
Seeds can make a pack faster only if they need fewer bytes than other formats at the same quality. This plan tests
that claim in small steps. Each step has a gate. If a gate fails, the plan stops or changes before the next large
cost. The model also has a per-layer n-gram embedding table (the PLE table) of 51.2 B parameters. Each token reads
only a few rows of it, so NanoSeedLM keeps it in a stock format. The pack does not include the PLE table. The user
downloads one of the four PLE table files from Unsloth, and the engine reads it from NVMe.

## Key numbers

### The model

From `config.json` and `model.safetensors.index.json`:

| Item | Value |
|---|---|
| Layers | 48: 36 Gated DeltaNet layers, and full attention in every fourth layer (12) |
| Experts | 512 per layer. Each token uses 10 routed experts and 1 shared expert |
| One expert | `gate_proj` and `up_proj` 640 x 2560, `down_proj` 2560 x 640: 4.92 M weights |
| All experts | 120.8 B weights. 2.36 B routed weights for each token |
| PLE table | 20 M rows x 2560: 51.2 B parameters |
| Other parts | Hyper-connections (4 streams), a sparse-attention indexer (budget 2048), one MTP layer, a vision encoder |
| Other weights | About 8 B parameters: 360 GB of BF16, less the experts and the PLE table |
| Checkpoint storage | Experts as one fused `gate_up_proj` and one `down_proj` for each layer, under `model.language_model.` |

### The PLE tables from Unsloth

From the [Unsloth guide](https://unsloth.ai/docs/models/qwen3.8-next) and the `engram/` folder of
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):

| File | Size | Relative RMSE against BF16 |
|---|---|---|
| `Qwen3.8-Flash-Next-engram-BF16.gguf_file` | 102.4 GB | 0 |
| `Qwen3.8-Flash-Next-engram-Q8_0.gguf_file` | 54.4 GB | 0.0053 |
| `Qwen3.8-Flash-Next-engram-Q5_1.gguf_file` | 38.4 GB | 0.0376 |
| `Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file` | 28.8 GB | 0.0757 |

### Baselines

Unsloth measured these on its own text set and harness, against the BF16 model. The source is the Unsloth guide on
2026-11-04, the day of the Dynamic V3.1 release.

| Pack | Size | Top-1 | Mean KLD |
|---|---|---|---|
| UD-IQ3_XXS | 82.0 GB | 85.41% | 0.1651 |
| UD-Q3_K_XL | 90.0 GB | 88.32% | 0.1065 |
| UD-IQ4_XS | 93.7 GB | 89.55% | 0.0836 |
| UD-Q4_K_XL | 111.3 GB | 92.26% | 0.0469 |
| UD-Q5_K_XL | 158.3 GB | 93.68% | 0.0304 |
| Q8_0 | 188.2 GB | 94.12% | 0.0266 |

Strata, against Unsloth Q8_0, on its text set v1: GSQ-RCO IQ3_S 87.69% and 0.144, UD-IQ4_XS 88.91% and 0.101,
UD-Q4_K_XL 93.09% and 0.056. These measurements may use the Unsloth files from before V3.1.

Strata speed on a PC near ours, with ISTA IQ3_XXS and MTP on:

| PC | Context | Decode | Prompt |
|---|---|---|---|
| Radeon RX 9070 16 GB, Ryzen 7 5800X3D, 64 GB RAM | 4K, 33K, 130K tokens | 58.7, 57.9, 57.9 tok/s | 586, 804, 833 tok/s |

We do not compare our numbers with numbers from another harness. Each gate measures its baseline again, in our
harness, on the same machine.

### Expert bits per weight

| Format | Expert bits per weight | Source |
|---|---|---|
| ISTA GSQ-RCO IQ3_XXS | 2.83 | The per-tensor allocation file of the ISTA repository |
| Seeds, 3 bits (C = 12, P = 4, the 3-bit layout of the SeedLM paper) | 3.0 | Not in NanoSeedLM yet |
| UD-IQ4_XS | About 3.9 | Expert size in the Strata documentation |
| P3 (`SEED4`) | 4.0 | |
| P4 (`SEED4P4`) | 4.5 | |
| UD-Q4_K_XL | About 5.1 | Expert size in the Strata documentation |
| P8 (`SEED6P8`) | 6.5 | |

## How it works

### The pack

| Part | Format | Place at run time |
|---|---|---|
| Routed experts | Seeds: P4 and P3. P8 for sensitive tensors. The 3-bit format if it passes its probe | RAM. A VRAM cache holds the most used experts |
| Attention, DeltaNet, shared expert, routers, hyper-connections, indexer | Q8 or BF16 first. P8 later if it frees VRAM at a small cost | VRAM |
| Embedding, LM head | Q8 first | VRAM |
| MTP layer | The same rules as the main layers | VRAM and RAM |
| PLE weights (convolution, projections, norms) | BF16 | VRAM |
| PLE table | Not in the pack. The user downloads one Unsloth file | NVMe, read through mmap |
| Vision encoder | Not in the first packs | |

### The PLE table

- The engine reads the table by its tensor name from the Unsloth GGUF file. It does not use `set_split.py`. llama.cpp
  needs that script because it loads the table as one shard of a split set.
- For each token, the CPU decodes the rows that the token needs from the mmap. BF16, Q8_0, Q5_1 and IQ4_NL are
  simple row formats. Each type gets a C decoder and a test against llama.cpp output.
- The operating system keeps recently read rows in its page cache. A smaller table leaves more RAM for the experts
  and for the cache. Step V6 measures decode speed with each table.
- The quality gates use the BF16 table. Then only the experts change between two candidates. Hypothesis H5 measures
  the cost of each smaller table.

To download a table:

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "engram/*Q8_0*" --local-dir MODEL_DIR
```

Disk space for a P4 pack. The pack size is an estimate: experts at 4.5 bits (68 GB) and the other weights in Q8
(about 8.5 GB).

| PLE table | Table | Pack and table |
|---|---|---|
| IQ4_NL | 28.8 GB | About 105 GB |
| Q5_1 | 38.4 GB | About 115 GB |
| Q8_0 | 54.4 GB | About 131 GB |
| BF16 | 102.4 GB | About 179 GB |

### The speed limit on the PC

Missed experts go to the GPU over PCIe, or the CPU computes them in RAM. Both paths read DRAM. So the DRAM bandwidth
`B` limits the sum. For one decode step:

```
steps per second <= B / ((1 - h) * 2.36e9 * b / 8)
h = share of routed expert uses that the VRAM cache serves
b = expert bits per weight
```

An example with `B` = 42 GB/s and `h` = 0.5. These are assumptions. Step S0 measures `B`, and step S8 measures `h`.

| Experts | b | Upper limit, steps per second |
|---|---|---|
| ISTA IQ3_XXS | 2.83 | 101 |
| Seeds, 3 bits | 3.0 | 95 |
| P3 | 4.0 | 71 |
| P4 | 4.5 | 63 |
| UD-Q4_K_XL | 5.1 | 56 |

MTP drafts let one step make more than one token. Strata reports 2.4 to 3.2 tokens for each verify step. Unsloth
reports 1.3x to 1.7x with MTP.

The CPU kernel sets how much of the miss traffic the CPU can take. We did a quick check of the current P4 mat-vec in a
cloud session on 2026-11-04. The machine was a Xeon with 4 vCPUs, and the load average was below 0.3. The build used
AVX2 only, as on a Zen 3 CPU.

| Kernel | 1 thread | 4 threads |
|---|---|---|
| `cpu_seed4_tiled_matvec_mt` | 3.3 G weights/s | 12.6 G weights/s |
| A table of all LFSR states (the state cache of the paper) | 0.8 to 2.9 G weights/s | 3.1 to 11.0 G weights/s |

The table did not make the kernel faster. For 8 Zen 3 cores, an estimate is 25 to 30 G weights/s. About 42 GB/s of
DDR4 supplies about 75 G weights/s of P4. So the CPU can take about a third of the miss traffic, and PCIe must carry
the rest. Step S0 measures the kernel on the 5800X3D.

## The hill-climb method

1. Fix the measurement before the climb: the text sets, the reference, the PLE table (BF16), the machine and the
   commands.
2. Change one setting in each step.
3. Measure the old and the new setting in alternated runs. Record machine state: power, load and other CPU use.
4. Keep the change if it passes its rule. For quality, the 95% bootstrap interval of the paired KLD difference must
   not show harm. For speed, the change must be at least 2% faster in alternated runs. Run a noisy benchmark again.
5. Add each step to `docs/log.md`. Write a results page for each gate.

The climb has two objectives. So we keep a Pareto front of expert bytes against KLD, and later of decode speed
against KLD. Each pack of the family is one point on the front.

### Measurement

- **Reference.** Unsloth Q8_0 with the BF16 PLE table, in llama.cpp on the PC. The model is larger than the RAM, so
  llama.cpp reads it through mmap from NVMe. This is slow, but the reference runs only once. Unsloth measured Q8_0
  at a mean KLD of 0.0266 against BF16, so this reference may hide small differences. If credits allow, one HF Job
  makes a BF16 reference (`h200x4`, $20 per hour).
- **Text sets.** The Strata set v1 (Strata's `tools/quant_quality.py`, 4,995 positions of code, documents, English,
  other languages and an agent transcript). Also about 30 held-out windows of chat and agent text, as in
  [R-02](../results/2026-10-10-seeds-vs-affine.md).
- **Statistics.** Paired differences for each window, with 95% bootstrap intervals (10,000 resamples).
- **Revisions.** Each baseline names the repository commit of its files. Unsloth changed its packs on 2026-11-04.
- **Speed.** The same prompts in NanoSeedLM, Strata and llama.cpp, at 4K, 32K and 128K tokens of context, with MTP on
  and off.

## Steps and gates

| Step | Work | Where | Cost | Gate |
|---|---|---|---|---|
| S0 | Access and baselines | PC | Free | None |
| S1 | Tools for Qwen3.8 | Cloud session | Free | Tests pass |
| S2 | One-layer probe | HF Job | About $0.20 | G1 |
| S3 | Seed banks, P4 and P3 | HF Jobs or the RTX 4080 | About $5 for each tier, or about 5 h on the 4080 | |
| S4 | Full-model quality with emulated packs | PC | Free | G2 |
| S5 | The allocation climb | Cloud session, PC, HF Jobs | Depends on GPTQ | |
| S6 | CPU expert path, on MoVA first | PC | Free | G3 |
| S7 | Engine port | Cloud session, PC, Mac | Free | Tests pass |
| S8 | The speed climb | PC | Free | G4 |
| S9 | Release | Cloud session | Free | |

### S0. Access and baselines

1. Correct the HF network secret of the cloud environment. Use the Bearer type with an `Authorization` header.
2. Measure the DRAM bandwidth of the PC. Four DIMMs can run at a lower speed than two.
3. Run Strata and llama.cpp with UD-Q4_K_XL, UD-IQ4_XS and ISTA IQ3_XXS. Record decode and prompt speed.
4. Run `nslm-cpu-seed-bench` on the 5800X3D.

### S1. Tools for Qwen3.8

1. `nslm-moe`: a Qwen3.8 tensor adapter for the fused, stacked expert tensors. Write the failing test first.
2. `tools/`: a converter from the Unsloth imatrix (`imatrix_unsloth.gguf_file`, 580 MB) to `actsq_moe.bin`. The
   imatrix holds the sum of x^2 for each input channel of each expert. That is the statistic of the
   activation-weighted search, so no capture run is necessary.
3. `tools/hf_seedbank.sh`: download only the 97 shards that hold experts (about 250 GB). Skip the PLE table shards.
   Write the outputs to a mounted HF storage bucket. Set the job timeout. The default is 30 minutes.
4. `tools/`: write decoded seed experts into a GGUF as Q8_0 tensors (an emulated pack), so that llama.cpp can run it.

### S2. One-layer probe

1. Search one layer (for example layer 20) at P4 and P3, activation-weighted with the imatrix.
2. Also search P4 with a block Hadamard rotation and plain L2. On Qwen3-1.7B, SeedLM+O measured a Hadamard rotation
   at 29% more KL than an activation-weighted fit.
3. Record the search speed for each weight. This speed replaces the cost estimates in this plan.
4. Compare with the tensors of the same layer in UD-Q4_K_XL, UD-IQ4_XS and ISTA IQ3_XXS. Use the error weighted by
   the imatrix, for each expert tensor.

**Gate G1.** P4 has a lower weighted error than UD-Q4_K_XL on most expert tensors of the layer. If G1 fails, the
claim "faster at the same quality" stops for 4-bit packs. Weight error does not always track KLD, so G1 only decides
whether to pay for S3.

### S3. Seed banks

1. Search all experts at P4 and at P3. Use the RTX 4080 overnight, or HF Jobs when the PC is not available.
2. Copy the outputs from the storage bucket to a dataset repository.

### S4. Full-model quality with emulated packs

1. Make a P4 pack and a P3 pack as emulated packs (S1, item 4).
2. Run both in llama.cpp on the PC with the BF16 PLE table. Run UD-Q4_K_XL and UD-IQ4_XS in the same harness.
3. The Q8_0 rounding of the emulated pack adds a small error. This error works against the seeds.

**Gate G2.** The KLD of the P4 pack is not worse than the KLD of UD-Q4_K_XL (hypothesis H1). If G2 fails, the plan
stops before the engine port.

### S5. The allocation climb

Each item is one climb step:

1. Give `down_proj` one tier more than `gate_proj` and `up_proj`, at the same mean bits. The ISTA IQ3_XXS allocation
   gives `down_proj` 3.09 bits and `gate_proj` and `up_proj` 2.70 bits.
2. Give the late layers more bits. ISTA gives layers 28 to 47 about 2.8 to 3.8 bits, and most earlier layers about
   2.1 to 3.0 bits.
3. Measure the sensitivity of each tensor as layer-output error on held-out activations. Then choose tiers for a byte
   budget.
4. Weight the sensitivity by the use frequency of each expert, from router choices on calibration text. Cold experts
   stay in RAM and cost decode time for each byte.
5. P8 for the most sensitive tensors.
6. GPTQ for the most sensitive tensors. GPTQ needs `X^T X`, which the imatrix does not have. That needs a BF16
   forward pass: about $20 per hour on `h200x4`, or layer by layer on the PC after S7.
7. The 3-bit format for cold experts. It needs a new search and new kernels. Do a one-layer probe as in S2 first.
   On Qwen3-1.7B, SeedLM+O measured the 3-bit layout (3.48 bits) at 2.1x the KL of Q3_K_M (3.85 bits).

The output is a Pareto front and the members of the family.

### S6. CPU expert path

1. In the CUDA engine, on K2-Horizon-MoVA first: the CPU computes a share of the missed experts, and PCIe carries the
   rest. Measure decode speed for a range of shares.
2. A CPU mat-vec for several tokens, for MTP verify windows. The CPU makes the states once for all the tokens.
   Done: `cpu_seed4_tiled_matmul_mt`. On a cloud Xeon (AVX2 build), 2 tokens give 1.3x and 4 tokens give 2.0x the
   one-token rate for each token. Only experts that more than one token of the window uses get this gain.
   P = 3 on the CPU (`cpu_seed3_tile`) is faster again: 13.9, 21.5 and 32.3 G weights/s for each token at 1, 2 and
   4 tokens on 4 vCPUs, against 12.4, 16.8 and 25.1 at P = 4. P = 3 also reads 11% fewer bytes. So P = 3 is the
   better tier for experts that the CPU computes.

**Gate G3.** Decode reaches at least 85% of the DRAM limit at the measured hit rate. If G3 fails, the CPU path stays
experimental, and PCIe carries all misses.

### S7. Engine port

1. Qwen3.6-35B-A3B first. It has the Gated DeltaNet layers, the gated attention, the MoE with a shared expert and the
   MTP layer. Its BF16 weights (72 GB) fit in 128 GB of RAM, so the reference and the tests run on the PC.
2. Then the parts that only Qwen3.8 has: the hyper-connections, the indexer, the PLE weights and the PLE table reader.
   The PLE table reader is done (`nslm/ple.h`). It decodes rows of the four Unsloth table files, and it matches
   gguf-py exactly on rows of each file.
3. CUDA first, then Metal. Add each feature that only one engine has to the feature matrix of
   [Metal and CUDA parity](../concepts/parity.md).
4. Each kernel gets a C reference and a test on Metal and on CUDA.
5. With MTP, greedy output must be identical to plain decode, as `nslm-mova-plcheck` checks for prompt lookup.

### S8. The speed climb

Each item is one climb step:

1. V1: all missed experts over PCIe, as the CUDA engine does now.
2. V2: the CPU share from S6.
3. V3: a static set of hot experts in VRAM, from router choices, with the adaptive cache for the rest.
4. V4: MTP drafts, and the draft length.
5. V5: the other weights in P8, to free VRAM for the expert cache.
6. V6: each PLE table type, from NVMe.

**Gate G4.** NanoSeedLM decode of the chosen pack is at least as fast as Strata decode of the current UD-Q4_K_XL.
Both run on the same PC, with the same prompts and with MTP on. The KLD of the pack must pass G2.

### S9. Release

1. Write a model card for each pack of the family.
2. Write the results pages and the log entries.
3. Check the Qwen Community License 1.0 before we publish packs.

## Hypotheses

We write these before we measure, as in R-02. "Not worse" means that the upper limit of the 95% interval of
KLD(seed pack) - KLD(baseline), paired by window, is at most 0.002.

| ID | Hypothesis | Rule |
|---|---|---|
| H1 | The P4 pack (4.5 bits) is not worse than UD-Q4_K_XL (about 5.1 bits) | Not worse |
| H2 | The P3 pack (4.0 bits) is not worse than UD-IQ4_XS (about 3.9 bits) | Not worse |
| H3 | A Hadamard rotation does not decrease the weighted error of P4 seeds on one layer | Exploratory. A probe of 4 experts supports it |
| H4 | The chosen pack decodes at least as fast as UD-Q4_K_XL in Strata on the same PC | G4 |
| H5 | The cost of each smaller PLE table, as a KLD difference against the BF16 table | Report only |

## Compute and cost

HF Jobs prices (per second billing). A job needs a positive credit balance.

| Flavor | Price | Disk | Use |
|---|---|---|---|
| `l40sx1` | $1.80 per hour | 380 GB | Seed search. The most FP32 for each dollar |
| `rtx-pro-6000` | $2.75 per hour | 475 GB | Seed search with more disk |
| `l40sx4` | $8.30 per hour | 3200 GB | Seed search with `--shard`, 4x faster |
| `h200x4` | $20.00 per hour | 12000 GB | A BF16 forward pass (564 GB of VRAM) |

The seed search uses FP32 and integer arithmetic, not tensor cores. So `a100-large` and `h200` give less search for
each dollar.

The search over the routed experts of K2-Horizon-MoVA (26.5 B weights) took 3.4 h on an M4 Max. Qwen3.8 has 4.6x as
many expert weights.

Measured in step S2 (2026-11-04, `rtx-pro-6000`, layer 20, P = 4, 2 workers): 334 s for one layer. `gate_proj` and
`up_proj` took 233 s each, side by side, and `down_proj` took 100 s. All 48 layers take about 4.5 GPU-hours for each tier.
`l40sx1` was not available, so S3 runs on `rtx-pro-6000x4`.

| Item | Cost |
|---|---|
| S2, one layer | About $0.45 (measured) |
| S3, P4, on `rtx-pro-6000x4` | About $14 (estimate from S2) |
| S3, P3, on `rtx-pro-6000x4` | About $14 (estimate from S2) |
| Packing, on a CPU flavor, for each tier | About $1 (estimate) |

A BF16 reference or a GPTQ capture costs about $20 for each hour. They need more credits.

## Where each step runs

| Place | Work |
|---|---|
| Cloud session | Code, CPU tests, the documentation, control of HF Jobs, analysis of results |
| HF Jobs | Seed searches, BF16 forward passes |
| The PC (RTX 4080, Ryzen 7 5800X3D, 128 GB DDR4) | Baselines, engine runs, speed, KLD with emulated packs |
| The Mac (M4 Max) | Metal tests for parity |

A cloud session cannot reach the PC by itself. These are the four ways to run a PC step:

1. **A Remote Control folder.** In the Claude desktop app on the PC, open Settings, Claude Code, Remote Control. Add
   the NanoSeedLM folder. A cloud session can then ask to use the folder, and you approve each request. Not all
   accounts have this feature yet.
2. **A session on the PC.** Start Claude Code on the PC, in the desktop app or with `claude remote-control` in the
   repository folder. That session reads this plan, runs the PC steps and commits the results pages.
3. **A self-hosted GitHub Actions runner on the PC.** A benchmark workflow starts only from `workflow_dispatch`. A
   cloud session starts the workflow and reads its logs. On a public repository, do not let pull requests from
   forks use the runner.
4. **By hand.** A cloud session writes a script. You run it on the PC and commit its output.

### HF machines for PC-like tests

HF Jobs can run the CUDA engine with a limit on VRAM. `NSLM_VRAM_RESERVE_MB` keeps VRAM free, so a 24 GB GPU can act
as a 16 GB or a 12 GB GPU.

| Flavor | GPU | VRAM | RAM | Price | Use |
|---|---|---|---|---|---|
| `l4x1` | L4 (Ada, the same architecture as the RTX 4080) | 24 GB | 30 GB | $0.80 per hour | MoVA engine tests |
| `a10g-largex2` | 2x A10G (Ampere). Use one | 48 GB | 92 GB | $3.00 per hour | Qwen3.8 packs in RAM |
| `l4x4` | 4x L4. Use one | 96 GB | 186 GB | $3.80 per hour | Qwen3.8 on the RTX 4080 architecture |
| `rtx-pro-6000` | RTX PRO 6000 (Blackwell) | 96 GB | 256 GB | $2.75 per hour | Qwen3.8 correctness tests |

- The T4 flavors have 16 GB, but the T4 is older than Ampere, the minimum of the CUDA engine.
- Speed on these machines does not predict speed on the PC. The CPUs, the DRAM bandwidth and the PCIe links are not
  the same. Use them for correctness tests and for A/B tests on one flavor. G3 and G4 need the PC.
- A job is a batch run, so each job pays for the build and the download. Keep packs in a mounted storage bucket.
- HF Jobs has no CUDA machine with unified memory (no GH200, GB200 or DGX Spark).

## The unified-memory track

NanoSeedLM was first written for unified memory. On a 128 GB M4 Max:

- The P4 pack (about 76 GB) and the IQ4_NL PLE table (28.8 GB) fit in memory together, about 105 GB. The BF16 table
  does not fit with the pack, so its rows come from the SSD.
- There is no split between the CPU and PCIe. The GPU computes all experts, so GPU arithmetic for seeds is the limit.
- The M4 Max runs P4 experts at about 287 G weights/s (commit `c6f9c12`). The 2.36 B routed weights then take about
  8 ms for each token. That is at most about 120 tok/s from the experts alone (an estimate).
- On MoVA, seed decode on Apple Silicon is 14% to 21% slower than affine Q4. So on the Mac the claim is a smaller
  pack at equal quality, not a faster pack.
- After S7, measure KLD and speed on the M4 Max against Strata and llama.cpp (Metal) with UD-Q4_K_XL.
- A CUDA machine with unified memory, for example a DGX Spark, needs a CUDA path like the Metal path, with no expert
  cache. Add it to the parity matrix when it exists.

## Risks

- Seeds may lose to codebook quantizations at the same bits. G1 and G2 test this before the large costs.
- The CPU kernel may not take enough of the miss traffic. Then PCIe carries the rest, and the hit rate sets the speed.
- The engine port is large: Gated DeltaNet, the indexer, the hyper-connections and the PLE table are all new.
- The RAM is tight. P4 experts take 68 GB, so the BF16 PLE table cannot stay in the page cache.
- The Unsloth and ISTA files change. Each baseline must name its file revision.
- Weight error does not always track KLD. Only KLD decides a gate after G1.

## Measurements

None yet.

## See also

- [SeedLM](../concepts/seedlm.md)
- [Seeds against affine quantization (R-02)](../results/2026-10-10-seeds-vs-affine.md)
- [Make a model folder](../guides/make-a-model.md)
- [NVIDIA GPUs](../guides/cuda.md)
- [Unsloth: Qwen3.8-Flash-Next](https://unsloth.ai/docs/models/qwen3.8-next)
- [ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- [Strata](https://github.com/Niko1221/Strata)
- [SeedLM+O](https://github.com/ropaes1/seedlm-o-results)
- [SeedLM](https://arxiv.org/abs/2410.10714), Shafipour et al., ICLR 2025
