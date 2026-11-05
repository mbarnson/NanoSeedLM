Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 41a8fe6c9f7d9ba8d1d009e7aab0cc9fadc6c4a8
Platform: macos
Agent: mac-metal
Time: 2026-11-05T22:47:35Z

macOS, M4 Max, clean worktree at 41a8fe6c9f7d9ba8d1d009e7aab0cc9fadc6c4a8 (claude/sleepy-brown-k67z9u on main 09adc5a: the faster trellis encoder, nslm-moe --codec tq): make build, then make test with MOVA_DIR. 37 of 37 pass, 0 skipped (test_trellis with its reference-Viterbi check, test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal, test_qwen4_mtp and test_qwen4_api_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
