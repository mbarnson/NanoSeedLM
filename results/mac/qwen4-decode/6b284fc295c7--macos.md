Status: PASS
Branch: mac/qwen4-decode
Commit: 6b284fc295c7402b663afe86b9b8694ff5ffa59e
Platform: macos
Agent: mac-metal
Time: 2026-11-05T19:19:47Z

macOS, M4 Max, clean worktree at 6b284fc295c7402b663afe86b9b8694ff5ffa59e (mac/qwen4-decode on main 764cd13): make build, then make test with MOVA_DIR. 35 of 35 pass, 0 skipped (test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal and test_qwen4_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
