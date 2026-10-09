Status: PASS
Branch: pc/fp4-decode
Commit: bf9dc696e4d4dfc5d657a9941c20cfa3e25d9911
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T06:15:09Z

Windows 11, MSVC + CUDA 13.1, RTX 4080: ctest 18/18 passed (test_affine skipped), no warnings. FP8 / FP4 attention still bit-equal to the decoded BF16 cache (test_mova_kernels); fp4 decode at 16k 38.0 -> 40.2 tok/s vs main.
