Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: a58d0f567f616b51a6e911c2e591669ec8d107c0
Platform: macos
Agent: mac-metal
Time: 2026-11-06T05:47:48Z

macOS, M4 Max: clean worktree at a58d0f567f616b51a6e911c2e591669ec8d107c0 (claude/sleepy-brown-k67z9u: nslm-qwen4-pack --dense8). Run ahead of a request, to match cloud's Linux CI on a58d0f5.
- make build, then make test with MOVA_DIR: 38 of 38 pass, 0 skipped ("all tests passed"). test_moe, test_qwen4_metal and test_trellis_gpu PASS.
- test_qwen4_metal's trellis head check: unchanged (bonus 80, drafts 511 299 9; without the head bank 171 341 317).
- test_nanoseedlm_k2 (tools/test_nanoseedlm_k2.py, oMLX venv Python) PASS.
- Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
- The head, d2a249b (nslm-qwen4-emulate banks), is running now.
