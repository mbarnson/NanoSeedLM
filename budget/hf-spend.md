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

| nslm-kld-d0 (6aed6158c78863f9c633db32): Q8_0, the mix /mlx, P3 /mlx against BF16 (saved the reference to /b/kld-ref) | rtx-pro-6000x4 ($11/h) | 24.9 | $4.56 | about $10.41 |

| Linux CI 99cf18a (6aed64300e0338d1bb80e49a): nslm-qwen4-pack --tq | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | |
| nslm-kld-r3 (6aed64480e0338d1bb80e4a3): Q8_0, TQ3 /mlx, TQ4 /mlx with the saved reference | rtx-pro-6000x2 ($5.50/h) | about 23 | about $2.10 | about $12.59 |
| Linux CI a58d0f5 (6aed69190e0338d1bb80e5b4): --dense8 | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | about $12.67 |

| Linux CI d2a249b (6aed69a9c78863f9c633de07): emulator packed folders | rtx-pro-6000 ($2.75/h) | about 1.7 | about $0.08 | about $12.75 |

| nslm-release-r4 (6aed6e410e0338d1bb80e70b): failed at the oQ8e download (my --include bug) | rtx-pro-6000x2 ($5.50/h) | 2.4 | about $0.22 | about $12.97 |
| nslm-release-r4 again (6aed6f19c78863f9c633e011): failed at the first write (no /b/release; my bug) | rtx-pro-6000x2 ($5.50/h) | 3.0 | about $0.27 | about $13.24 |
| nslm-release-r4, third (6aed6ff30e0338d1bb80e764), running | rtx-pro-6000x2 ($5.50/h) | expected about 70, at most 110 | expected about $6.40, at most $10.08 | |

Left this week: about $11.76 before the third nslm-release-r4.
