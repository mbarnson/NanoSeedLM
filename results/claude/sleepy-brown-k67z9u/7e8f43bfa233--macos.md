Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 7e8f43bfa233ec402992dfa0a418706fcccce73d
Platform: macos
Agent: mac-metal
Time: 2026-11-06T05:02:44Z

macOS, M4 Max, clean worktree at 7e8f43bfa233ec402992dfa0a418706fcccce73d (claude/sleepy-brown-k67z9u: nslm-qwen4-emulate --dense and a test_emulate case, on main 5bb851d): make build, then make test with MOVA_DIR. 38 of 38 pass, 0 skipped (test_emulate, test_engine, test_qwen4_kernels, test_qwen4_metal, test_trellis_gpu PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
