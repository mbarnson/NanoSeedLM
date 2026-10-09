Status: PASS
Branch: batch
Commit: 6364ffb6a6c9793e83a1416df7244c9a318a7ee2
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T00:35:28Z

Windows 11, MSVC + CUDA 13.1, RTX 4080, build-v (Release). ctest 18/18 passed, test_affine skipped.
test_mova_kernels: MLA FP8 / FP4 cache write = C codec (0 RoPE-key values a code off); kv_f32 exact; latent attention
over the quantized cache bit-equal to the decoded BF16 cache (4 cases); "expanded prompt attention: not in this backend"
(kt_mla_attn_xq is Metal's kernel; CUDA's expanded path is covered by test_engine).
test_engine: fp8 / fp4 cache sizes exact, decode vs reference (fp8 mean 8.3e-2, fp4 8.4e-2), chunking / batching / slot
copies bit-exact, including "MLA blocks of 16" (the mla_expand_min rule).
Held-out KLD, 31 windows, BF16 / fp8 / fp4: healed (heal-pilot) MLA P=4 0.1534 / 0.1576 / 0.1582 (SE 0.0022);
naive J768 P=4 0.2192 / 0.2228 / 0.2231 (SE 0.0030). BF16 identical to 6120fa4's binary (0.1534294).
Speed (nslm-mova-bench, healed P=4, decode 128): ctx 4k BF16 52.1 / fp4 49.2 tok/s; ctx 16k BF16 46.0 / fp4 38.3.
Prefill 16k BF16 1219 -> 1162 tok/s: the 255-token tail under xend runs absorbed (16385 tokens: 1216).
Linux: not covered yet (WSL Ubuntu has no CUDA toolkit; asking Matt).
