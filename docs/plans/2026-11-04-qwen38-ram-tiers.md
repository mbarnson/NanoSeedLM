# Plan: three seed packs of Qwen3.8-Flash-Next by RAM size (2026-11-04)

**Goal.** Publish three seed packs of Qwen3.8-Flash-Next, one for each minimum PC: 64 GB, 96 GB and 128 GB of RAM.
Each PC has a 12 GB GPU and keeps the BF16 PLE table on NVMe. Each pack must match the quality of an Unsloth Dynamic
V3.1 pack of its RAM class in one harness. It must also decode faster than that pack on the same PC. Seeds encode every routed expert.
Seeds also encode other weights where a gate shows no loss.

**Status.** In progress. Matt approved the plan on 2026-11-04. The parent plan is
[SeedLM packs for Qwen3.8-Flash-Next](2026-11-04-qwen38-flash-next.md).

- T1 is done, and gate G5 passes ([the packs against BF16](../results/2026-11-04-qwen38-kld-vs-bf16.md)).
- T2 is done, and gate G6 fails on weight error ([seed probe](../results/2026-11-04-qwen38-seed-probe.md)). The S3
  search of T3 runs anyway, so that the KLD of G7 decides T64.
- T4 step 1 is done: the hot experts depend on the text domain
  ([expert use by domain](../results/2026-11-04-qwen38-routing-domains.md)).
- T3 step 1 is done: the S3 banks ([3-bit seeds](../results/2026-11-04-qwen38-s3-tier.md)).
- T5 has P3 / P4 and S3 / P3 hybrids ([hybrid tiers](../results/2026-11-04-qwen38-hybrid-tiers.md)). A static pack
  does not reach its target at 3.3 bits (0.112 against 0.0735). A tier that follows each session's hot set may: with
  the text's own hot third, S3 / P3 gives 0.0765.

## Key numbers

The fixed part of each PC:

| Part | Size | Place |
|---|---|---|
| Dense weights (attention, DeltaNet, shared experts, embeddings, head) | about 4.5 GB | VRAM |
| Activations and the prompt pass | about 1 GB | VRAM |
| Expert cache | about 6 GB | VRAM (the rest of a 12 GB GPU) |
| BF16 PLE table | 102.4 GB | NVMe (16 rows of 320 bytes for each token) |
| Operating system and programs | about 10 GB | RAM |

The routed experts hold 120.8 B weights. They stay in RAM, and the cache holds the most recent ones in VRAM.

| Pack | RAM | Expert budget | Mean expert bits | Quality target (Unsloth KLD against BF16) |
|---|---|---|---|---|
| T64 | 64 GB | 50 GB | 3.3 or fewer | UD-IQ3_XXS (0.165). Stretch: UD-Q3_K_XL (0.107) |
| T96 | 96 GB | 80 GB | 5.3 or fewer | UD-IQ4_XS (0.084). Stretch: UD-Q4_K_XL (0.047) |
| T128 | 128 GB | 110 GB | 7.3 or fewer | UD-Q4_K_XL (0.047). Stretch: UD-Q5_K_XL (0.030) |

Unsloth measured its KLD in its own harness. The gates of this plan measure every pack again in one harness.

The expert bits also set the decode traffic. From the routing trace of 1,024 P3 tokens, with LRU and a 6 GB cache
([cache policies](../results/2026-11-04-qwen38-cache-policies.md)):

| Mean expert bits | Cache hits | PCIe traffic for each token | Time on PCIe 4 |
|---|---|---|---|
| 3.0 | 75.4% | 217 MB | about 9 ms |
| 3.4 | 72.6% | 274 MB | about 11 ms |
| 4.0 | 69.1% | 365 MB | about 15 ms |
| 4.5 | 66.5% | 445 MB | about 18 ms |
| 5.0 | 64.4% | 526 MB | about 21 ms |
| 6.5 | 59.0% | 787 MB | about 32 ms |

## How it works

### Seed tiers

| Tier | Layout | Bits for each weight | State |
|---|---|---|---|
| S3 | C = 12 weights, P = 4 coefficients, a 4-bit exponent, a 16-bit seed (the 3-bit layout of the SeedLM paper) | 3.0 | New: searcher, format and kernels |
| P3 (`SEED4`) | C = 8, P = 3 | 4.0 | In the packs |
| P4 (`SEED4P4`) | C = 8, P = 4 | 4.5 | In the packs |
| P8 (`SEED6P8`) | C = 8, P = 8 | 6.5 | Searcher and reader. Not in the CUDA engine |

Every search weights each squared error by the energy of its input channel: the sum of x^2 in Unsloth's imatrix
(`imatrix_unsloth.gguf_file`). So the search spends its accuracy where the activations are large.

### The allocation

Each (expert, tensor) gets one tier. Its cost is its bytes. Its value is the error that a higher tier removes,
weighted by two things:

- The sensitivity of the tensor: its output error on held-out activations (parent plan S5, item 3).
- The use frequency of the expert: its share of router choices on calibration text (`Q4E_ROUTES` traces). A cold
  expert costs little quality and many bytes.

A greedy pass upgrades the tensors with the most value for each byte until the pack's expert budget is full.

### Decode

The GPU computes the cached experts from their seeds. The CPU computes a share of the missed experts from their seeds in
RAM (parent plan S6). So fewer expert bytes cross PCIe.

## Steps and gates

| Step | Work | Cost | Gate |
|---|---|---|---|
| T1 | One quality harness against BF16 | About $20 | G5 |
| T2 | The S3 probe on one layer | About $1 | G6 |
| T3 | Full searches: S3, and P8 for the candidate tensors | About $40 | |
| T4 | Sensitivity and use frequency | About $10 | |
| T5 | The allocation, and emulated packs for each RAM size | About $15 | G7 |
| T6 | Dense weights in P4 seeds | About $5 | G8 |
| T7 | Speed on the RTX 4080 with a 6 GB cache | Free | G9 |
| T8 | Release | Free | Matt's go |

### T1. One quality harness against BF16

1. Write decoded seed experts into a GGUF as Q8_0 tensors: an emulated pack (parent plan S1, item 4).
2. Make the reference: BF16 logits on the parent plan's text sets, with the BF16 PLE table. Use `h200x4` (564 GB of
   VRAM), or Unsloth Q8_0 in llama.cpp on the PC as a cheaper reference.
3. Measure the KLD of P3, P4, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS and UD-Q4_K_XL in llama.cpp, against that reference.

**Gate G5.** The harness gives the six packs the same order as Unsloth's table. If it fails, find the cause before T5.

**Result.** G5 passes. Against BF16 on wikitext-2, P4 has a KLD of 0.0408 at 4.50 bits, on Unsloth's curve. P3 has
0.0600 at 4.00 bits, 18% more than UD-IQ4_XS at 3.97 bits.

### T2. The S3 probe

1. Write the S3 searcher: the C reference first, with a failing test, then the GPU search (CUDA and Metal).
2. Search layer 20 at S3 with the imatrix. Compare the weighted error with P3 and P4 of the same layer.
3. Decode Unsloth's IQ3_XXS tensors of layer 20, and measure their weighted error in the same way.

**Gate G6.** On layer 20, the weighted error of S3 is at most the weighted error of IQ3_XXS. If G6 fails, T64 uses P3
for the hot experts and streams the rest from the page cache.

**Result.** G6 fails. On layer 20, S3 has a weighted error of 0.213 and IQ3_XXS has 0.184. The C reference measured
it on the CPU. The GPU search of S3 is not written. Unsloth gives `down_proj` 4.5 to 6 bits in all four packs.

### T3. Full searches

1. Search all experts at S3 (24,576 experts, on `rtx-pro-6000x4`).
2. Search P8 for `down_proj` and for the late layers (28 to 47), the tensors that ISTA's allocation gives more bits.
3. Copy the seed banks to a dataset repository.

### T4. Sensitivity and use frequency

1. Record routing traces (`Q4E_ROUTES`) on three calibration texts: chat, code and documents.
2. Measure the output error of each (layer, tensor) at each tier on held-out activations.

### T5. The allocation and the emulated packs

1. Allocate each of T64, T96 and T128 within its expert budget.
2. Write each as an emulated pack, and measure its KLD in the T1 harness.

**Gate G7.** Each pack is not worse than its target pack, by the rule of the parent plan. The upper limit of the 95%
interval of the paired KLD difference is at most 0.002. If a pack fails, move bytes to its most sensitive tensors
and measure again.

### T6. Dense weights in seeds

1. Search the dense weights at P4 with the imatrix. They now use 4-bit affine weights, 5.0 bits for each weight.
2. Measure the KLD of T96 with P4 dense weights against T96 with affine dense weights.

**Gate G8.** The P4 dense weights are not worse. Then all three packs use them, and the VRAM that they free goes to
the expert cache.

### T7. Speed

1. On the RTX 4080, set the cache so that it leaves the VRAM that a 12 GB GPU leaves.
2. Measure decode and prompt speed of each pack. Also measure Strata and llama.cpp with the target Unsloth pack on the
   same PC.

**Gate G9.** Each pack decodes at least as fast as its target pack in Strata on the same PC.

### T8. Release

1. Pack T64, T96 and T128 with the CUDA engine and the MLX loader.
2. Write a model card for each. Publish after Matt's go.

## Hypotheses

| ID | Hypothesis | Rule |
|---|---|---|
| H6 | S3 seeds have at most the weighted error of IQ3_XXS on one layer | G6 |
| H7 | T64 (3.3 bits or fewer) is not worse than UD-IQ3_XXS | G7 |
| H8 | T96 is not worse than UD-IQ4_XS | G7 |
| H9 | T128 is not worse than UD-Q4_K_XL | G7 |
| H10 | P4 dense weights are not worse than the 4-bit affine dense weights | G8 |

## Risks

- S3 quality. On Qwen3-1.7B, SeedLM+O measured the 3-bit layout (3.48 bits) at 2.1x the KL of Q3_K_M (3.85 bits).
- The KL divergences of P3 and P4 so far are against the 4-bit MLX model, not BF16. They may not reach the targets.
- T64 pins 50 GB of a 64 GB PC. The operating system may limit pinned memory.
- The CPU seed kernel needs Clang. The Windows build uses MSVC.

## See also

- [SeedLM packs for Qwen3.8-Flash-Next](2026-11-04-qwen38-flash-next.md), the parent plan
- [The seed search over the Qwen3.8-Flash-Next experts](../results/2026-11-04-qwen38-seed-search.md)
- [Qwen3.8-Flash-Next decode with affine, P4 and P3 experts](../results/2026-11-04-qwen38-decode-by-format.md)
- [Expert cache policies](../results/2026-11-04-qwen38-cache-policies.md)
