Status: FAIL
Branch: mac/mla-seeds
Commit: 1aa0fa2bcc06e1d4b5a01ab3f15e907acd5d8eb1
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T03:07:03Z

test_mova_kernels FAIL (everything else passes, 17/18):
  per-head maps Q4 (O 96, I 128, T 3): 98 mismatches
  per-head maps Q4 (O 128, I 192, T 5, gate): 290 mismatches
Cause: this branch changed the Q4 decode reference to the qmv form (s * q + b unrounded); CUDA k_heads_mv still decodes Q4 as bf16(s * q + b). The fix is CUDA-side (pc-cuda: k_heads_mv through mv_lane, as the dense matvec), on pc/mla-seeds-cuda.
Windows 11, MSVC + CUDA 13.1, RTX 4080, no build warnings.
