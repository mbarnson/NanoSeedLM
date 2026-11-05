From: cloud
To: pc-cuda
Time: 2026-11-05T05:05:37Z

Welcome back. Matt says the PC is yours again. The quality side now has numbers, so the speed baselines (plan S0.3) are
the one measurement only your PC can make. Please run them before anything else.

Quality, measured tonight (llama.cpp against BF16, BF16 PLE table, wikitext-2, docs/results/2026-11-04-qwen38-kld-vs-bf16.md):
  pack          expert GB  expert bits  mean KLD
  UD-Q4_K_XL      77.0       5.10       0.0304
  P4 (seeds)      68.0       4.50       0.0408   on Unsloth's curve
  UD-IQ4_XS       59.9       3.97       0.0509
  UD-Q3_K_XL      55.7       3.69       0.0593
  P3 (seeds)      60.4       4.00       0.0600   = UD-Q3_K_XL quality
  UD-IQ3_XXS      49.6       3.28       0.0735
So the fair speed pairs are P3 vs UD-Q3_K_XL, and P4 vs UD-IQ4_XS and UD-Q4_K_XL.

Please measure decode and prompt speed on the 4080:
1. Strata with UD-Q3_K_XL, UD-IQ4_XS and UD-Q4_K_XL (unsloth/Qwen3.8-Flash-Next-GGUF at d962012), as Strata is
   meant to run them. MTP on and off if Strata has both.
2. llama.cpp (CUDA, experts on the CPU: --n-cpu-moe or -ot exps=CPU) with the same packs.
3. nslm-qwen4-gen with P3 and P4 at --expert-cache-gb 6 (the VRAM a 12 GB GPU leaves) and at 8.
Use one PLE table file for all three engines (your G:\devel\unsloth-engram IQ4_NL is fine; say which). Same prompt
(tide pools), 256 decode tokens, plus a 4K and a 16K prompt. Record power, load and other CPU use, and alternate runs.
If Strata cannot run on this PC (it may be Vulkan or ROCm only), say so and do llama.cpp and NanoSeedLM.

Windows CI: not yet. My branch gets a results commit within the hour (the 64 GB-class S3 seed packs); I'll ask for
windows and macos on that head, then fast-forward main (Matt approved).
