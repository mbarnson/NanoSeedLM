Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 5ed8d1bb3433bf159ca28816a4a39e60a1c3aeee
Platform: windows
Agent: pc-cuda
Time: 2026-11-04T23:37:25Z

win\build.bat test, clean build in a fresh worktree of 5ed8d1b (MSVC 14.44, CUDA 13.1, RTX 4080, driver 610.88):
ctest 31/31 passed (test_qwen4_ref, test_qwen4_kernels, test_qwen4_engine, test_mova_kernels). Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir). Q4E_PROFILE=1 smoke run on P3 (cache 8): works, exit 0.
