Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 73abfdf0d9aeb5a8874493ded05150572930c124
Platform: macos
Agent: mac-metal
Time: 2026-11-06T20:50:55Z

macOS, M4 Max: clean worktree at 73abfdf0d9aeb5a8874493ded05150572930c124 (claude/sleepy-brown-k67z9u: serve drafts through eng_generate_pick, q4e_draft_from, nslm-chat drafts at any temperature).
- make build, then make test with MOVA_DIR: 40 of 40 pass, 0 skipped ("all tests passed").
- New PASS: test_qwen4_pick and test_sample. test_qwen4_mtp, test_qwen4_api_mtp and test_qwen4_metal PASS (the trellis head unchanged).
- test_nanoseedlm_k2 PASS.
- Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
