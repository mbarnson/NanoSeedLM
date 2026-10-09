Status: PASS
Branch: pc/mla-prefill-deq
Commit: 6c6b96f48b731d7d3f8c83da4c96c908f0dc7ead
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T06:25:31Z

Windows 11, MSVC + CUDA 13.1, RTX 4080: ctest 18/18 passed (test_affine skipped), no warnings. Q8 projections, fp4 KV, 16k prefill 952 -> 998 tok/s vs main.
