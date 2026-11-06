Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: da4e1cebd04c7927b1f376f69c19cf34c4f675f6
Platform: macos
Agent: mac-metal
Time: 2026-11-06T21:08:43Z

macOS, M4 Max: clean worktree at da4e1cebd04c7927b1f376f69c19cf34c4f675f6 (claude/sleepy-brown-k67z9u: A1 draft probabilities and A2 sampled drafts on CUDA, with stubs on Metal; the merged projections and plan K are on the branch).
- make build, then make test with MOVA_DIR: 40 of 40 pass, 0 skipped ("all tests passed").
- test_qwen4_mtp prints "no draft probabilities on this engine (plan A1m)" twice and "no sampled drafts on this engine (plan A2m)": the Metal stubs, as expected. test_qwen4_metal (trellis head unchanged), test_qwen4_api_mtp, test_qwen4_pick and test_sample PASS.
- test_nanoseedlm_k2 PASS.
- Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
