Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: 79cc53b67b935352a0d807861005ffc2a5b607f8
Platform: macos
Agent: mac-metal
Time: 2026-11-04T22:30:07Z

macOS 27.2, M4 Max 128 GB, AC power, memory 96% free. Spotlight indexing and Podcasts used about one core each.
Clean build (rm -rf out) of 79cc53b. make build: PASS. One old warning (harness/nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: FAIL. 28 of 29 test programs pass, and none skip. The failure:
  q4 group 32 GEMM 96 x 2560, 16 tokens: 6.8e-01
  FAIL tests/test_qwen4_kernels.c:117: q4 group 32 GEMM differs
tests/kernel_backend_metal.m is the same as in ebddfb2 and 384bc45. kt_mm has no group guard, so the Metal GEMM reads
group-32 scales as group 64.
With one line at the top of kt_mm, `if (w->group && w->group != 64) return 1;`, make test passes: 29 of 29.
tools/ is unchanged since ebddfb2, where tools/test_nanoseedlm_k2.py passes (28 of 28, OMLX_K2_MODEL set).
