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
| nslm-release-r4, third (6aed6ff30e0338d1bb80e764): packed TQ3 into /b/release (86.5 GB), then bus errors on the bucket mount; cancelled | rtx-pro-6000x2 ($5.50/h) | about 23 | about $2.11 | about $15.35 |
| nslm-release-r4, fourth (6aed75a20e0338d1bb80e84b): packed TQ3-D4 and TQ4, KL of all three as shipped; runs of TQ3 only (script bug) | rtx-pro-6000x2 ($5.50/h) | 47.9 | about $4.39 | about $19.74 |

| nslm release upload (6aed9354c78863f9c633ec80, by pc-cuda): failed at once (Git Bash turned /b/... into B:/...) | cpu-xl ($1.00/h) | under 1 | about $0.01 | about $19.75 |

| nslm release upload again (6aed93be0e0338d1bb80efb1, by pc-cuda, MSYS_NO_PATHCONV=1): no bytes moved, then EIO (hf upload read the bucket mount; my script bug) | cpu-xl ($1.00/h) | 56.4 | about $0.94 | about $20.69 |

| Linux CI 8856cc4 (6aeda44f0e0338d1bb80fac7): --dense-only, 44 PASS | rtx-pro-6000 ($2.75/h) | 1.8 | about $0.08 | about $20.77 |

| Linux CI f4d997d (6aeda8020e0338d1bb80fdc8): the release workflow, NSLM_STATIC_ICU, 44 PASS | rtx-pro-6000 ($2.75/h) | about 1.8 | about $0.08 | about $20.85 |

| Linux CI 418224b (6aedac030e0338d1bb81010f): the packer writes the MLX loader, 44 PASS | rtx-pro-6000 ($2.75/h) | about 1.8 | about $0.08 | about $20.93 |

| Linux CI a499194 (6aedacb2c78863f9c6340172): the loader packer and mac-metal's resource_dir, 44 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $21.01 |

| Linux CI 00918f0 (6aedb2170e0338d1bb810657): the embedding in host memory, 44 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $21.58 |

| nslm-rel-linux (6aedb3e40e0338d1bb810784): cancelled at start (my wrong archive path) | rtx-pro-6000 ($2.75/h) | under 1 | about $0.02 | about $21.60 |

| nslm-rel-linux (6aedb3f3c78863f9c634082f): the v0.1.0 Linux archive on Ubuntu 22.04 with TQ3: chat and serve pass | rtx-pro-6000 ($2.75/h) | 8.7 | about $0.40 | about $22.00 |

| Linux CI 1a55b64 (6aedc9fc0e0338d1bb814084): Q4EOpts.pass_rows, 44 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $22.08 |

| Linux CI 354e2e1 (6aedcf6ec78863f9c6345152): passes of 8,192 rows under 32 GB, 44 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $22.16 |

| Linux CI 20666ef (6aedd91d0e0338d1bb8160ac): the BF16 KV cache (plan K2), 44 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $22.24 |

| Linux CI 98491b7 (6aede5ec0e0338d1bb81795c): nslm-qwen4-kld --save / --ref-logprobs, 44 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $22.32 |

| nslm release upload, third (6aeda68a0e0338d1bb80fcb4, by pc-cuda): TQ3 5f2a8599, TQ3-D4 7e5edb2b, TQ4 25339937 (each copy about 8 min, each upload about 1 min: Xet found the chunks) | cpu-xl ($1.00/h) | 29.1 | about $0.49 | about $21.50 |

| nslm-kld-ple4 (6aee1d9dc78863f9c634fd46): Q8_0 0.01991; Q8_0 with P3's 4-bit MLX PLE table 0.02588 (the saved BF16 reference) | rtx-pro-6000x2 ($5.50/h) | 7.3 | about $0.66 | about $22.98 |

| Linux CI f9353b3 (6aee28540e0338d1bb820865): serve-drafts S1-S3, build failed (test_serve_splitter) | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.06 |

| Linux CI 765538e (6aee2912c78863f9c63517fd): serve-drafts S1-S3, 45 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.14 |

| Linux CI e94f859 (6aee2a01c78863f9c6351a4c): nslm-serve drafts (S4), 46 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.22 |

| Linux CI 69c7238 (6aee305dc78863f9c6352521): serve draft log line, 46 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.30 |

| Linux CI 73abfdf (6aee31f80e0338d1bb821943): nslm-chat drafts at any temperature, 46 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.38 |

| Linux CI 304d415 (6aee38a8c78863f9c635314f): draft confidence gate (A1), 46 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.46 |

| Linux CI da4e1ce (6aee3b960e0338d1bb822577): drafts sampled from the head's q (A2), 46 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.54 |

| Linux CI b911cb5 (6aee509cc78863f9c6355c1b): v0.1.1 candidate (main, A1m/A2m, merged projections), 46 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.62 |

| Linux CI a193c5f (6aee63780e0338d1bb826ba5): server hardening H1-H3, 47 PASS | rtx-pro-6000 ($2.75/h) | about 2 | about $0.08 | about $23.70 |

Left this week: about $1.30.
