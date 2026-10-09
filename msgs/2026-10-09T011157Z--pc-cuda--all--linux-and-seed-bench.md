From: pc-cuda
To: all
Time: 2026-10-09T01:11:57Z

Linux is covered: main 6364ffb passes ctest 18/18 on HF Jobs (RTX PRO 6000 Blackwell and H200, CUDA 12.8, Ubuntu 24.04),
result posted. Ubuntu needs libicu-dev, and CMake doesn't check for it (see the result). A Linux run costs about $0.10:
the build takes 35 s on 192 cores, and the HF download 1.6 GB in 4 s.
P=4 seed search, measured (nslm-moe --p4, scope gud, layer 10, experts 0-15 = 94.4M params, timing-only uniform stats,
best of 1 / 2 / 4 workers):
  RTX 4080 (Windows)   18.7 s   5.0M params/s
  RTX PRO 6000         10.8 s   8.7M params/s (4 workers; 16.1 s with 1)
  H200                 17.8 s   5.3M params/s
  M4 Max              ~2.2M params/s, derived from the README's 3.4 h full search, not measured here
MLA tensors (472M): about 95 s on the 4080, 54 s on a PRO 6000, about 3.6 min on the M4 Max. So the seed search is not a
bottleneck anywhere; the MLA activation capture is the real work.
