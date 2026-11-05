Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 7dd456aff5d1a848e429f4217364debe25ac579d
Platform: macos
Agent: mac-metal
Time: 2026-11-05T22:02:16Z

macOS, M4 Max, clean worktree at 7dd456aff5d1a848e429f4217364debe25ac579d (claude/sleepy-brown-k67z9u, 6 docs-only commits on main e816fe8): make build, then make test with MOVA_DIR. 36 of 36 pass, 0 skipped (test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal, test_qwen4_mtp and test_qwen4_api_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
