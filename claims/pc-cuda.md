# pc-cuda claims (updated 2026-11-05T22:17Z)

- **Online** (Claude Code CLI session).
- Now: T2, the CUDA Viterbi encoder for nslm/trellis.h, on pc/trellis-cuda (from 09adc5a). Files: nslm/trellis_cuda.cu (new), tests/test_trellis_cuda.c (new), harness/nslm-trellis-bench.c (new), CMakeLists.txt (targets for those only).
- Waiting: T5 (KL of each PLE format), for disk (Matt is freeing ~200 GB for the BF16, Q8_0 and Q5_1 engram tables).
- Done this session: windows PASS on cb3d6b6, e816fe8, bc50f41, 7dd456a, 09adc5a; D4 on the 4080 (--cpu-tail 1 and 2 lose, -14% and -27%).
