Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: eab620d3766d2039c656dc87febf60c065af5f1f
Platform: macos
Agent: mac-metal
Time: 2026-11-04T22:31:34Z

macOS 27.2, M4 Max 128 GB, AC power. make build: PASS (the same old warning, harness/nslm-mova-score.c:157).
make test with MOVA_DIR set: FAIL, 28 of 29. Only test_qwen4_kernels fails, as on 79cc53b:
  FAIL tests/test_qwen4_kernels.c:117: q4 group 32 GEMM differs   (6.8e-01)
eab620d changes only CUDA files, CUDA-only tests and docs. The Metal kt_mm still needs its group guard.
