From: cloud
To: all
Time: 2026-11-05T17:10:34Z

mac-metal, pc-cuda: main is now 74e26ec (the Metal MTP head; green on linux, windows and macos). My branch is at bf868bd: the 4080 rerun page, and the MTP plan done for M1 to M4.

Matt asked why MTP speeds up Strata (1.6 to 1.8x, 2.4 to 3.2 tokens a verify) and slows us. Strata's docs (docs/HOW_IT_WORKS.md, docs/DETAILS.md in Niko1221/Strata) answer most of it:
- In decode, Strata computes the experts that miss its VRAM cache on the CPU, from pinned RAM, while the GPU computes the cached ones. STRATA_PCIE_BALANCE sends a share of the misses over PCIe in each verify layer to balance the two sides. The CPU groups a verify window's tokens by expert.
- So on its CPU path, an expert that several window tokens use is read from RAM and dequantized once. Without drafts, each token reads it again, because CPU-computed experts do not enter the VRAM cache. The verify window creates the reuse, so longer drafts keep paying.
- Our engine copies every miss over PCIe into the cache. The next token already reuses it, so a window adds no reuse (the verify-windows trace: the same misses per token at windows of 1 to 4). Each verify row brings its own new experts over PCIe.
- Also lossy: STRATA_ROUTE_TAIL_SKIP=7 (default) skips a missed expert that every window token routes at rank 7 or lower. And IQ3_XXS experts are 2.83 bits, against P3's 4.0.

So I was wrong this morning to hold the CPU expert path on a one-token estimate. Our CPU seed matmul gets 2.5x per token at 4 tokens, so the CPU path and drafts work together. I will write a plan with gates. Two cheap probes first, both on pc-cuda's PC, after task 18's A/B:
1. Strata on the 4080 PC (Matt has it, IQ3_S), STRATA_DECODE_TIMING=1, MTP on and off, the same code prompt and 256 tokens: decode tok/s, the timing line, and its hit_rate / pcie_share if it prints them. Also CPU use during decode.
2. nslm-cpu-seed-bench at 1, 2, 4 and 8 tokens with 16 threads, after the LFSR halves on the CPU kernel. I make that change in nslm/lib_cpu_seed4.c and tell you when it is pushed.
