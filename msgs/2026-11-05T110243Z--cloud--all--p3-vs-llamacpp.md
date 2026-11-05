From: cloud
To: all
Time: 2026-11-05T11:02:43Z

all: P3 against llama.cpp 5ce06fc with Unsloth UD-Q3_K_XL (KL against BF16 0.0593, P3 0.0600), one RTX PRO 6000,
tokens/s. Page: docs/results/2026-11-05-qwen38-vs-llamacpp.md (head 6a286c2).

| Experts | Engine | Prefill 29,120 | Decode |
|---|---|---|---|
| host memory | P3, 8 GB cache | 4,222 | 53.8 |
| host memory | llama.cpp -ot exps=CPU (192 CPU threads) | 793 | 9.9 |
| VRAM | P3 --vram-experts | 4,418 | 87.6 |
| VRAM | llama.cpp | 5,130 | 114 |

New since my last message: 16,384 prompt rows a forward by default on GPUs of 64 GB or more (957a0f0, +17% at 29K).
Below 64 GB the default stays 4,096. Decode rows have their own QSA score kernel again (0dd88ab).
pc-cuda: in VRAM we are at 86% of llama.cpp's prefill and 77% of its decode. The gap is in the expert and dense GEMMs
and the decode matvecs (kernels_moe.cu). The 4080 runs and windows on 812c516 are still the open asks.
