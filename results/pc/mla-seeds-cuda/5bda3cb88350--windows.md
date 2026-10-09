Status: PASS
Branch: pc/mla-seeds-cuda
Commit: 5bda3cb8835066c1959610ce958c91f235c23e08
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T03:16:47Z

Windows 11, MSVC + CUDA 13.1, RTX 4080: ctest 18/18 passed (test_affine skipped), no build warnings. test_mova_kernels: per-head Q8 / Q4 (Q4 decode in the qmv form) and SEED4P4 (decode, prompt GEMMs, transposed reads) all run on CUDA; test_engine: "MLA capture: 39 rows; sums in one call vs chunks of 7: worst relative difference 7.35e-07".
