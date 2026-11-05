Status: PASS
Branch: mac/qwen4-decode
Commit: b910034b4fcef00099c37fa339d68871d17e4308
Platform: macos
Agent: mac-metal
Time: 2026-11-05T19:53:16Z

macOS, M4 Max, clean worktree at b910034b4fcef00099c37fa339d68871d17e4308 (mac/qwen4-decode on main 764cd13): make build, then make test with MOVA_DIR. 35 of 35 pass, 0 skipped (test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal and test_qwen4_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
