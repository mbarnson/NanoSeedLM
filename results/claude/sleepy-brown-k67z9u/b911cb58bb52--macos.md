Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: b911cb58bb5249fb0e819f58fcdec74524171ee6
Platform: macos
Agent: mac-metal
Time: 2026-11-06T22:19:08Z

make build, make test (40 binaries, 0 skipped, "all tests passed", test_qwen4_metal PASS), tools/test_nanoseedlm_k2.py 0. M4 Max, clean worktree.
The three Metal lines together:
- test_qwen4_mtp PASS: draft probabilities 0.0119 0.0075 0.0087 / 0.0075 0.0080 0.0070, the reference's to 4 digits; the sampled-draft checks run.
- test_qwen4_api_mtp PASS; test_qwen4_pick PASS (the gate keeps the greedy tokens).
- The trellis head's drafts are unchanged (bonus 80, drafts 511 299 9).
Warnings: the two older ones only (harness/nslm-mova-score.c:157, tests/test_chat_template.c:56).
