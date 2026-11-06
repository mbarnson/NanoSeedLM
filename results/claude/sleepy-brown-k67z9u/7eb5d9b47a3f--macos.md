Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 7eb5d9b47a3f4b6f6baacb8d9512c6ed70f78e50
Platform: macos
Agent: mac-metal
Time: 2026-11-06T05:17:35Z

macOS, M4 Max, clean worktree at 7eb5d9b47a3f4b6f6baacb8d9512c6ed70f78e50 (claude/sleepy-brown-k67z9u: nslm-qwen4-emulate --dense with the DeltaNet head map, a test_emulate case): make build, then make test with MOVA_DIR. 38 of 38 pass, 0 skipped (test_emulate, test_engine, test_qwen4_kernels, test_qwen4_metal, test_trellis_gpu PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
