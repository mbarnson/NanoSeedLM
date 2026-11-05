# A high seed tier that follows the routing of Qwen3.8-Flash-Next: a simulation (2026-11-04)

**Result.** For S3 / P3 mixes, the KL divergence falls almost linearly with the share of router choices that P3
serves. A high tier holds the experts with the largest decayed use count. At 10.4 GB of P4 it serves 64% to 71% of the
choices, and at 20 GB 83% to 87%. It loads 3 to 5 experts for each token. By the linear fits, a 64 GB pack may reach a KLD
of about 0.077 (UD-IQ3_XXS: 0.0735), and a 96 GB pack about 0.044 (UD-IQ4_XS: 0.0509).

**Status.** Current. A simulation on routing traces, and estimates from linear fits. No engine runs a tier like this.

## Setup

- Traces: wt2 (the start of the KLD text, 14,883 tokens), chat (11,894) and code (21,974), from
  [expert use by domain](2026-11-04-qwen38-routing-domains.md). The first 2,000 tokens of each warm the tier.
- The tier: every 16 tokens, the N experts with the largest use count. Each count halves after 500 or 2,000 tokens.
  An insertion is an expert that enters the tier. N is the budget over the expert size (P4 2.76 MB, P3 2.46 MB).
- KLD points: [3-bit seeds](2026-11-04-qwen38-s3-tier.md) and [hybrid tiers](2026-11-04-qwen38-hybrid-tiers.md).

```bash
python3 tools/qwen4_tier_sim.py p3-wt2.bin p3-chat.bin p3-code.bin --gb 10.4 20 --half 500 2000
```

## Results

KLD against the share of router choices in the high tier (wikitext):

| Pack | High-tier share | Mean KLD | Linear fit |
|---|---|---|---|
| S3 | 0% | 0.142 | 0.142 |
| S3, a random third in P3 | 33.5% | 0.116 | 0.116 |
| S3, the text's own hot third in P3 | 83% | 0.0765 | 0.076 |
| P3 | 100% | 0.0600 | 0.063 |
| S3, the text's own hot fifth in P4 | 68% | 0.0776 | 0.142 - 0.095 x share |

The share of router choices that the tier serves (half-life 500 tokens):

| High tier | Experts | wt2 | chat | code | Insertions for each token |
|---|---|---|---|---|---|
| 10.4 GB of P4 | 3,768 (15%) | 70.7% | 63.7% | 69.8% | 3.3 to 4.3 (9 to 12 MB) |
| 10.4 GB of P3 | 4,227 (17%) | 73.7% | 67.1% | 72.8% | 3.7 to 4.6 (9 to 11 MB) |
| 20 GB of P4 | 7,246 (29%) | 87.1% | 83.2% | 85.7% | 5.1 to 5.5 (14 to 15 MB) |

A half-life of 2,000 tokens loads 3 times fewer experts and serves 3 to 7 points less. A plain LRU tier serves 73% to
79% at 10.4 GB, but it loads 100 to 130 experts for each token.

## Notes

- 64 GB: S3 for all experts (45.6 GB) leaves 4.4 GB of RAM. With the 6 GB VRAM cache, 10.4 GB can hold P4 copies.
  The fit then gives 0.075 to 0.081.
- 96 GB: P3 for all experts (60.4 GB) and 20 GB of P4 copies give 0.043 to 0.044 by the P3 / P4 fit
  (0.060 - 0.019 x share).
- The insertions read P4 experts from a file on NVMe: about 0.3 GB/s at 30 tokens per second.
- These are estimates. The KLD of a tier that changes during the text is not measured.
