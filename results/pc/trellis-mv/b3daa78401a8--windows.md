Status: PASS
Branch: pc/trellis-mv
Commit: b3daa78401a855b4317499c759ed5a7e6307b4ef
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T22:56:16Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at b3daa78 (clean tree; on main 09adc5a).
- cl (MSVC 14.44): ctest 42/42 PASS, test_tq_mv_gpu 0.29 s (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 43/43 PASS, test_tq_mv_gpu 0.32 s (same 3 skipped).
- No new warnings in engine/kernels_moe.cu, tests/test_tq_mv_gpu.c or harness/nslm-mova-kbench.c.
Log: G:\devel\build-b3daa78401a8.log.
