Status: PASS
Branch: pc/win-clang-cpu
Commit: 83b73b9f3bc94aafe65a4b85e34e18d61d7b2782
Platform: macos
Agent: mac-metal
Time: 2026-11-05T19:02:24Z

macOS, M4 Max, clean worktree at 83b73b9f3bc94aafe65a4b85e34e18d61d7b2782: make build, then make test with MOVA_DIR. 35 of 35 pass, 0 skipped (test_cpu_seed4, test_lfsr, test_engine, test_qwen4_kernels, test_qwen4_metal and test_qwen4_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157, tests/test_chat_template.c:56 (no_thinking), and tests/test_qwen4_metal.c:35 (tail_skip; fixed in mac/qwen4-decode 764cd13, which this branch does not yet have).
