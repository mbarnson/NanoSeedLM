Status: PASS
Branch: mac/mla-pack
Commit: 0a3d437cae6f6b9c2c0124aa61b491feb6830878
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T01:26:48Z

Windows 11, MSVC + CUDA 13.1, RTX 4080: ctest 18/18 passed (test_affine skipped). test_mova_kernels: "per-head maps in Q8 / Q4: not in this backend" (the kt_heads_q stub) and "per-head maps (transposed weights): not in this backend". test_model_st passes.
