# pc-cuda claims (updated 2026-11-05T22:30Z)

- **Online** (Claude Code CLI session).
- Now: T2, the CUDA trellis encoder, on pc/trellis-cuda 82882e2 (from 09adc5a): bit for bit nslm_tq_encode_row; windows and linux CI running.
  Files: nslm/trellis.cu, nslm/trellis_gpu.h, tests/test_trellis_gpu.c, harness/nslm-trellis-bench.c (new); CMakeLists.txt (their targets); Makefile (test_trellis_gpu.c in the CUDA-only filter-out list).
- Downloading for T5: the BF16, Q8_0 and Q5_1 engram tables (d962012) to G:\devel\unsloth-engram.
- Done this session: windows PASS on cb3d6b6, e816fe8, bc50f41, 7dd456a, 09adc5a; D4 on the 4080 (--cpu-tail 1 and 2 lose, -14% and -27%).
