Status: PASS
Branch: mac/qwen4-mtp
Commit: 74e26ec88fc510b62f006c23d7b649e047512437
Platform: macos
Agent: mac-metal
Time: 2026-11-05T17:01:51Z

macOS, M4 Max, clean worktree at 74e26ec88fc510b62f006c23d7b649e047512437 (mac/qwen4-mtp on 3f570f3): make build, then make test with MOVA_DIR. 35 of 35 pass, 0 skipped. That count includes test_qwen4_mtp, now in the Metal make test, and test_engine, test_mova_kernels and test_qwen4_metal. test_nanoseedlm_k2 PASS. One warning, as before (harness/nslm-mova-score.c:157).
