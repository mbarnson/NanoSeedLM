# pc-cuda claims (updated 2026-11-05T22:56Z)

- **Online** (Claude Code CLI session). Matt gave the PC for now; no run over about an hour.
- Now: T5 on the 4080 (KL of the Q8_0 / Q5_1 / IQ4_NL / pack PLE tables against BF16, P3; then 256-token decodes), G:\devel\qwen4-runs\t5.sh.
- T3 CUDA: pc/trellis-mv b3daa78 (k_tq_mv_gu / k_tq_mv_sel in engine/kernels_moe.cu, kc_tq_* in engine/kernels_cuda.h, tests/test_tq_mv_gpu.c, nslm-mova-kbench rows; CMakeLists.txt and one Makefile line). windows PASS, linux running. Waiting on the MF_TQ id (msgs ...--pc-cuda--all--t3-cuda-plan.md).
- Parked: pc/trellis-cuda 82882e2 (the CUDA Viterbi encoder; windows and linux PASS).
- Done this session: windows PASS on cb3d6b6, e816fe8, bc50f41, 7dd456a, 09adc5a, 41a8fe6; D4 on the 4080 (--cpu-tail 1 and 2 lose).
