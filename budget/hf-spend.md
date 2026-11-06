# HF spend

Matt's cap: $25 a week (set 2026-11-05 17:00 Pacific). A week runs Thursday 17:00 to Thursday 17:00 Pacific.
No HF job or bucket upload starts without Matt's OK, with its flavor, minutes and cost stated first.
Linux CI: cloud only, once for each commit that changes code (about $0.07 on rtx-pro-6000).
Cost = running seconds x the flavor's price (hf jobs hardware). Scheduling time is free.

Before the cap (not counted): mac-metal's Linux CI 6aed21b3 (00:39Z, about 3 min) and 6aed2643 (00:58Z, cancelled at
about 1 min), rtx-pro-6000, about $0.15 together.

## Week 1: 2026-11-05 17:00 to 2026-11-12 17:00 Pacific

| Job | Flavor | Minutes | Cost | Running total |
|---|---|---|---|---|
| nslm-kld-tq-mix (6aed21aec78863f9c633bdd0): TQ2, mixes, UD-Q2_K_XL against BF16 | rtx-pro-6000x4 ($11/h) | 29.3 | $5.37 | $5.37 |

| Linux CI fb31f8c (6aed3b7a0e0338d1bb80d9cb): nslm-qwen4-gen --prompt-file | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | about $5.45 |

| Linux CI e752917 (6aed4fefc78863f9c633d4db): the VRAM reserve | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | about $5.53 |

| Linux CI 1d0bb37 (6aed588b0e0338d1bb80e1a6): ns_add_banks, the MTP head's banks | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | about $5.61 |

| Linux CI 7e8f43b (6aed5eb3c78863f9c633da18): nslm-qwen4-emulate --dense | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | about $5.69 |

| Linux CI 7eb5d9b (6aed60c8c78863f9c633dadc): the DeltaNet head order | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | about $5.77 |

Left this week: about $19.23.
