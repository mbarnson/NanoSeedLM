Status: PASS
Branch: mac/qwen4-decode
Commit: 764cd13cabc0c31c8428dc3745696998766400aa
Platform: macos
Agent: mac-metal
Time: 2026-11-05T18:58:11Z

macOS, M4 Max, clean worktree at 764cd13cabc0c31c8428dc3745696998766400aa (mac/qwen4-decode on main 0fb164a): make build, then make test with MOVA_DIR. 35 of 35 pass, 0 skipped (test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal and test_qwen4_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (no_thinking; not mine). The tail_skip warning is gone.
