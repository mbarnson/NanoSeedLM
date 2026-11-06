# Plan: trellis-coded experts for Qwen3.8-Flash-Next at 2 to 3 bits (2026-11-05)

**Goal.** Find whether trellis-coded experts beat Unsloth's quality curve at the same bits for each weight. If they do,
NanoSeedLM gets better quality and speed at a given VRAM or RAM footprint. On the RTX 4080, 2-bit experts would also
double the expert cache's slots.

**Status.** T1 done ([weight error](../results/2026-11-05-trellis-probe.md)). T2 done at L = 12
([KL against BF16](../results/2026-11-05-trellis-kld.md)). Gate TB passes. K = 4 (4.125 bits) has a KL of 0.0398,
16% below Unsloth's curve and below P4's 0.0408. K = 3 (3.125 bits) has 0.0765, 4% below an extrapolated curve. The
CUDA encoder on 8 GPUs encodes one K in about 5 minutes, so encodes run on HF. Next: T3 (decode in the engine) and T4
(packs mixed by layer or by use). A Q8_0 PLE table passes gate TP in llama.cpp (T5).

## The method

A trellis code stores a bit stream, not one code for each weight. Weight t of a block reads the stream's bits
[k t, k t + L) as an L-bit state s. Its value is the block's scale times c(s). Each weight adds k new bits, so a
weight costs k bits, but its value depends on L bits. The function c is computed, not looked up: QTIP's "1MAD" code
multiplies s by a constant, adds a constant (mod 2^32), and sums the 4 bytes. The sum of 4 uniform bytes is close to
Gaussian. The encoder runs a Viterbi search over the 2^L states for each block. Its cost for each weight is the imatrix
weight of the weight's input column times the squared error.

- No rotation: the experts are already near-Gaussian, and an imatrix-weighted fit without rotation is better
  ([rotation probe](../results/2026-11-04-qwen38-rotation-probe.md)).
- A block is T = 256 weights of one row, with a 16-bit scale. The stream has k T + L - k bits, so a block costs
  k + (L - k + 16) / T bits for each weight: 2.12 at k = 2 and 3.12 at k = 3 for L = 16.
- The decoder needs a shift, a multiply-add and a byte sum for each weight: less work than a P = 3 seed block.

## Steps and gates

| Step | Work | Where | Gate |
|---|---|---|---|
| T1 | Weight error of layer 20's experts (4 experts, gate, up, down): the trellis at k = 2 and 3 against ggml's IQ2_XXS, IQ2_XS, IQ2_S, Q2_K, IQ3_XXS, IQ3_S, Q3_K and IQ4_XS with the same imatrix | cloud (CPU) | TA |
| T2 | A packer and a trellis search of every expert; KL against BF16 as in [the packs against BF16](../results/2026-11-04-qwen38-kld-vs-bf16.md), and Unsloth's 2-bit packs measured the same way | pc-cuda or mac-metal (search), HF (KL) | TB |
| T3 | Decode kernels for trellis experts (CUDA, Metal) and decode speed at the automatic cache on the RTX 4080 | pc-cuda, mac-metal | TC |

- **TA.** At about 2.1 bits, the trellis's imatrix-weighted error is at most IQ2_S's (2.5 bits). At about 3.1 bits it
  is at most IQ3_S's (3.44 bits). Else the plan stops.
- **TB.** The model's mean KL against BF16 lies below Unsloth's curve at the same expert bits.
- **TC.** On the RTX 4080, decode is faster than P3's at the automatic cache, with the same text from run to run.

## A family of packs

The experts hold 121 G weights, so each bit for each expert weight costs 15.1 GB. The dense weights take about 4 GB
at 4 bits. The PLE table (32 GB at 4 bits) stays a separate file on the SSD. A token reads only 16 of its rows, and the
engine already reads Unsloth's engram GGUFs (`--ple-gguf`).

| Pack | Expert bits | Experts | Machine it is for |
|---|---|---|---|
| K2 | 2.1 | 32 GB | A 12 GB GPU with 32 GB of RAM (needs T6), or a 48 GB Mac |
| K2 + K3 by use | about 2.6 | 39 GB | A 16 GB GPU with 48 GB of RAM, or a 64 GB Mac |
| K3 | 3.1 | 47 GB | A 16 GB GPU with 64 GB of RAM, or a 64 GB Mac |
| K3 + K4 by use | about 3.6 | 54 GB | A 24 GB GPU with 64 GB of RAM, or a 96 GB Mac |
| K4 | 4.1 | 62 GB | A 128 GB Mac, or a 24 GB GPU with 96 GB of RAM |
| K6 | 6.1 | 92 GB | 128 GB of VRAM or more |

- "By use" gives the most used experts the larger K, from a routing trace (as the [hybrid tiers](../results/2026-11-04-qwen38-hybrid-tiers.md)).
- Each pack gets its own KL row against Unsloth's curve before it is published.
- The PLE table's own quantization is not measured yet: every KL so far used the BF16 table.

## More steps

| Step | Work | Where | Gate |
|---|---|---|---|
| T4 | The family above, each pack with its KL | pc-cuda or mac-metal (search), HF (KL) | TB for each pack |
| T5 | Packs without the PLE table, a PLE file in several formats, and the KL of each PLE format | cloud, HF (KL) | TP |
| T6 | Exclusive tiers: an expert held in VRAM has no copy in host RAM, so VRAM and RAM add up | pc-cuda, cloud | T6G |

- **TP.** A PLE format becomes a download if the KL rises by at most 2% against the BF16 table.
- **T6G.** K2 decodes on a 12 GB GPU with 32 GB of RAM, with the same text from run to run.

## Hypotheses

| | Hypothesis | Gate |
|---|---|---|
| HT1 | A trellis needs about 0.4 bits fewer for each weight than ggml's types, for the same weighted error | TA |
| HT2 | At about 2.6 bits for each expert weight, the KL is at most UD-IQ3_XXS's 0.0735 (3.28 bits) | TB |
| HT3 | At 2.1 bits the RTX 4080's cache holds about 1.9 times P3's slots, and decode is 30% faster | TC |

## Costs

- T1: the cloud container's CPU, with the layer-20 tensors and the imatrix from the P3 work. ggml builds from source.
- T2: the search of 121 G expert weights. Each weight costs about 2^L x 6 operations (3 passes of a minimum and an
  update for each state): 0.025 M at L = 12, 0.4 M at L = 16. On one CPU core at L = 12, the full search is about
  6 days. On the 16 cores of an M4 Max, it is about half a day (an estimate, not measured).
  KL jobs on HF: about $2 each, for the trellis pack and Unsloth's 2-bit packs.
- T3: kernel work on both engines.

## Risks

- QTIP's gains come with Hessian-aware rounding (BlockLDLQ). The imatrix is only the Hessian's diagonal.
- A 2-bit model may lose more quality than its weight error shows.
- The Viterbi search is slow on a CPU: T1 measures a sample of rows.

## See also

- [QTIP](https://arxiv.org/abs/2406.11235), [QuIP#](https://arxiv.org/abs/2402.04396), and TCQ's origin
  ([Marcellin and Fischer, 1990](https://doi.org/10.1109/26.46532))
- [Related work](../reference/related-work.md), section "Trellis-coded quantization"
- [SeedLM packs for Qwen3.8-Flash-Next](2026-11-04-qwen38-flash-next.md)
