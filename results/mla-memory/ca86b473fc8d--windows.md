Status: PASS
Branch: mla-memory
Commit: ca86b473fc8d0d11e4ff113f20e656d91c98567f
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T02:43:30Z

Windows 11, MSVC + CUDA 13.1, RTX 4080: ctest 18/18 passed (test_affine skipped), no build warnings. test_mova_kernels runs the new mixed BF16 / Q8 / Q4 decompression cases (kt_mla_decomp_q) on CUDA; test_kv_disk passes with the SetFileTime / GetFileAttributesEx file times.
