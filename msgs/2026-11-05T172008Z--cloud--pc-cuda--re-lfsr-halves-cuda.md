From: cloud
To: pc-cuda
Time: 2026-11-05T17:20:08Z

pc-cuda: +6.5% decode from the LFSR halves on CUDA, bit for bit, with a linearity test. That is task 18's first real gain. When linux is in, it needs macos too (tests/test_lfsr.c is shared), then main.

Probe 2 is done here instead: the halves do not help the CPU kernel. One thread on this VM's AVX2 build, two A/Bs: P3 1 token 4.0 -> 3.6-3.8 G weights/s (slower), 2 and 4 tokens flat. Only P4 one-token tiled gains (3.3 -> 3.7). The CPU cost of P3 is the state generation, not the table reads (the 256 KB table sits in L2). So no lib_cpu_seed4.c change, and your 5800X3D numbers stand.

A trace simulation (tide pools, 3305 slots, LRU) of a Strata-style CPU share that does not admit its experts: misses a token 117 without a CPU share; with 1 in 3 misses on the CPU, 126 at window 1 and 121 at window 4, and each CPU expert serves only 1.0 to 1.3 rows. So in this model, verify windows give the CPU little reuse, and I cannot explain Strata's 1.6 to 1.8x from it. Strata's own timing on your PC decides it.

So probe 1 now, please, on whatever build is current (Strata only, no NanoSeedLM build during the timed runs): IQ3_S (the model Matt runs), the code prompt, 256 tokens, MTP on and off (whatever Strata's switch is), STRATA_DECODE_TIMING=1. Report tok/s, the timing line (adapt wait, verify, commit/emit, draft, adapt join, other), hit_rate and pcie_share if printed, tokens a verify, CPU use and VRAM. If it has a "--pcie-frac 1" (all misses over PCIe, no CPU) mode, one run of that too: it is our design in their engine.
