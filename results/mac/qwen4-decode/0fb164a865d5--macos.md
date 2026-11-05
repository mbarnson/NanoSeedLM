Status: PASS
Branch: mac/qwen4-decode
Commit: 0fb164a865d5969a74d95c4a260fd9b79d19c5f6
Platform: macos
Agent: mac-metal
Time: 2026-11-05T18:08:43Z

macOS, M4 Max, clean worktree at 0fb164a865d5969a74d95c4a260fd9b79d19c5f6 (mac/qwen4-decode on main 6732b3f): make build, then make test with MOVA_DIR. 35 of 35 pass, 0 skipped (test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal and test_qwen4_mtp PASS). test_nanoseedlm_k2 PASS. One warning, as before (harness/nslm-mova-score.c:157).
