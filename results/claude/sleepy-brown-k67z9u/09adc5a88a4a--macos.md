Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 09adc5a88a4ab000344b8b7e4904269931b55d70
Platform: macos
Agent: mac-metal
Time: 2026-11-05T22:19:37Z

macOS, M4 Max, clean worktree at 09adc5a88a4ab000344b8b7e4904269931b55d70 (claude/sleepy-brown-k67z9u on main 7dd456a: d747625's nslm-qwen4-kld --ple-gguf and the trellis C reference): make build, then make test with MOVA_DIR. 37 of 37 pass, 0 skipped (the new test_trellis, test_lfsr, test_engine, test_mova_kernels, test_qwen4_kernels, test_qwen4_metal, test_qwen4_mtp and test_qwen4_api_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
