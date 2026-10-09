Status: PASS
Branch: pc/gptq-cuda
Commit: c39e480b33f4f1d93918d3b9f2f54f9b2f158c81
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T05:34:05Z

Windows 11, MSVC + CUDA 13.1, RTX 4080: ctest 19/19 passed (test_affine skipped), no C warnings. test_engine: "MLA capture X^T X: diagonal vs sums 0.00e+00, one call vs chunks of 7 2.70e-07, 0 asymmetric"; test_mova_kernels X^T X runs on CUDA.
