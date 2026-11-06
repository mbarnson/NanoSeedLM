Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 354e2e1b614d5bfd3336c52ba8d984a38245586a
Platform: macos
Agent: mac-metal
Time: 2026-11-06T13:18:36Z

macOS, M4 Max: clean worktree at 354e2e1b614d5bfd3336c52ba8d984a38245586a (claude/sleepy-brown-k67z9u: passes of at most 8,192 rows by default; engine/qwen4_eng.h builds on the Mac too). Run unasked, as the Mac is free.
- make build, then make test with MOVA_DIR: 38 of 38 pass, 0 skipped ("all tests passed"). test_qwen4_metal PASS.
- test_nanoseedlm_k2 PASS.
- Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
