Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: ebddfb28169674dae92720c72a744c7abcdb4836
Platform: macos
Agent: mac-metal
Time: 2026-11-04T22:27:11Z

macOS 27.2, M4 Max 128 GB, AC power, memory 96% free. Spotlight indexing and Podcasts used about one core each.
New worktree of ebddfb2. make build: PASS. It gives one old warning (harness/nslm-mova-score.c:157), in a file that
this PR does not change.
make test: FAIL. 28 of 29 test programs pass. MOVA_DIR is the K2-Horizon-MoVA snapshot, so no test skips. The failure:
  q4 group 32 GEMM 96 x 2560, 16 tokens: 6.8e-01
  FAIL tests/test_qwen4_kernels.c:117: q4 group 32 GEMM differs
All other qwen4 kernels print "not on Metal", as expected.
Cause: the Metal kt_mv returns 1 for group 32, but the Metal kt_mm has no guard. So the GEMM reads the group-32
scales as group 64.
With a one-line guard in kt_mm and pc-cuda's dup -> fdup rename, make test passes: 29 of 29.
tools/test_nanoseedlm_k2.py (OMLX_K2_MODEL set): PASS, 28 of 28, in oMLX's venv (mlx 0.32.4) and with mlx-vlm 0.7.12.
With mlx-vlm 0.7.12, the oMLX part of one check skips (oMLX is not in that venv).
