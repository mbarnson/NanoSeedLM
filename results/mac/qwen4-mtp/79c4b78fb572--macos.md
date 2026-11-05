Status: PASS
Branch: mac/qwen4-mtp
Commit: 79c4b78fb572216c3838eec7545a00074ed700f7
Platform: macos
Agent: mac-metal
Time: 2026-11-05T16:17:40Z

macOS, M4 Max, clean worktree at 79c4b78fb572216c3838eec7545a00074ed700f7: make build, then make test with MOVA_DIR. 34 of 34 pass, 0 skipped. test_qwen4_metal PASS (rows of up to 8 now bit for bit for any split; the forced split path against the C reference). test_qwen4_kernels PASS. test_nanoseedlm_k2 (MLX loader) PASS.
Also, not in make test: tests/test_qwen4_mtp.c built against the Metal engine with a local stub of cuda_runtime_api.h. PASS: drafts equal to the C reference at prompt lengths 16 and 45, keep/accept bit for bit, and greedy decode with 1, 2 and 3 drafts equal to decode without.
Warnings: only the two already there (engine/qwen4_api.c:56 mtp initializer, harness/nslm-mova-score.c:157).
