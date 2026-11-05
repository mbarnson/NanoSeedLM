Status: PASS
Branch: mac/qwen4-chat-drafts
Commit: e816fe88008e72c184a63a21cd6a55bb5466ca94
Platform: macos
Agent: mac-metal
Time: 2026-11-05T21:44:59Z

macOS, M4 Max, clean worktree at e816fe88008e72c184a63a21cd6a55bb5466ca94 (mac/qwen4-chat-drafts on main cb3d6b6): make build, then make test with MOVA_DIR. 36 of 36 pass, 0 skipped (test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal, test_qwen4_mtp and the new test_qwen4_api_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
