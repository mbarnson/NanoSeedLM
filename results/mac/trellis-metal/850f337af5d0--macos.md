Status: PASS
Branch: mac/trellis-metal
Commit: 850f337af5d078084eb47d1409010d7491e3b94d
Platform: macos
Agent: mac-metal
Time: 2026-11-06T00:40:57Z

macOS, M4 Max, clean worktree at 850f337af5d078084eb47d1409010d7491e3b94d (mac/trellis-metal on 7e442c7: the Metal trellis encoder): make build, then make test with MOVA_DIR. 38 of 38 pass, 0 skipped: test_trellis_gpu now runs on the Mac and passes (the same words, scales and error as nslm_tq_encode_row; two threads, 200 calls each, 0 failed), and test_trellis, test_engine, test_qwen4_kernels, test_qwen4_metal, test_qwen4_mtp and test_qwen4_api_mtp pass. test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine). The run shared the Mac with the K = 3 CPU encode.
