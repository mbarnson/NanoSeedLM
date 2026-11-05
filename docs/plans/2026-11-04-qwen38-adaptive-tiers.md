# Plan: a high seed tier that follows the routing of Qwen3.8-Flash-Next (2026-11-04)

**Goal.** The engine keeps a low seed tier of every expert in RAM. It also keeps a high tier of the experts that the
current session uses most. The 64 GB pack must come near UD-IQ3_XXS in KLD, and the 96 GB pack must beat UD-IQ4_XS. Neither
may decode slower than its static pack.

**Status.** Draft, for Matt's go. It changes step T5 of
[three seed packs by RAM size](2026-11-04-qwen38-ram-tiers.md): a static allocation does not reach T64's target.

## Key numbers

| Pack | Low tier in RAM | High tier (RAM + 6 GB VRAM) | Router share of the high tier | KLD estimate | Target |
|---|---|---|---|---|---|
| T64 | S3, 45.6 GB | 10.4 GB of P4 | 64% to 71% | 0.075 to 0.081 | UD-IQ3_XXS 0.0735 |
| T96 | P3, 60.4 GB | 20 GB of P4 | 83% to 87% | 0.043 to 0.044 | UD-IQ4_XS 0.0509 |

The shares and the estimates come from [the tier simulation](../results/2026-11-04-qwen38-tier-sim.md). A static
pack of T64 at 3.35 bits measures 0.112 to 0.118 ([3-bit seeds](../results/2026-11-04-qwen38-s3-tier.md)).

## How it works

1. The pack holds both tiers of each expert. The low tier is pinned in RAM. The high tier is a second file on NVMe.
2. The engine counts the router's choices on the GPU. Each count halves after 500 tokens.
3. Every 16 tokens, the top N experts by count form the high set. A worker thread reads the new members from NVMe into
   a RAM pool, 3 to 5 experts for each token.
4. A routed expert in the high set runs at the high tier: from the VRAM cache, or copied from the RAM pool. Any other
   expert runs at the low tier, as now.
5. Misses copy the low tier, so S3 moves 25% fewer bytes over PCIe than P3.

## Steps and gates

| Step | Work | Owner | Cost | Gate |
|---|---|---|---|---|
| A1 | S3 in the pack format (packed exponent codes) and the packer | cloud | Free | Tests pass |
| A2 | S3 matvec and grouped GEMM in the CUDA engine, with C references | cloud, pc-cuda | Free | GA1 |
| A3 | The same on Metal | mac-metal | Free | GA1 |
| A4 | `nslm-qwen4-kld` against a llama.cpp KLD base file (BF16 logits) | cloud | About $5 | GA2 |
| A5 | Two tiers, the use counts and the high set in the CUDA engine | cloud, pc-cuda | About $10 | GA3 |
| A6 | Speed on the RTX 4080 against Strata and the static packs | pc-cuda | Free | GA4 |

- **GA1.** The S3 kernels match their C references bit for bit on CUDA and on Metal.
- **GA2.** The engine's KLD of static P3 agrees with llama.cpp's (0.0600) within 5%.
- **GA3.** T64 with the moving high tier: KLD at most 0.0735 + 0.002 (the rule of gate G7). T96: below UD-IQ4_XS.
- **GA4.** Each adaptive pack decodes at least as fast as its static pack on the 4080.

If GA3 fails for T64, T64 keeps P3 for the hot experts and streams the rest, as gate G6 of the parent plan says.

## Hypotheses

| ID | Hypothesis | Rule |
|---|---|---|
| H11 | In the engine, the high tier serves at least 64% of router choices at 10.4 GB | Report |
| H12 | T64 adaptive is not worse than UD-IQ3_XXS | GA3 |
| H13 | T96 adaptive is better than UD-IQ4_XS | GA3 |

## Risks

- The linear fits come from static maps. A tier that changes during a text may do worse.
- S3 decode costs as much as P4 for each weight. The S3 matvec may be slower than P3's.
- NVMe reads compete with the PLE table reads. Both are small, but random.
- The engine needs per-expert formats in one projection. That changes the grouped GEMM and the cache.
