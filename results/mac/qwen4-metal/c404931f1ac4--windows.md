Status: PASS
Branch: mac/qwen4-metal
Commit: c404931f1ac4cf42ce029f96de613827a04e8637
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T02:31:57Z

win\build.bat test, clean build in a fresh worktree of c404931 (MSVC 14.44, CUDA 13.1, RTX 4080, driver 610.88):
ctest 31/31 passed (test_qwen4_ref, test_qwen4_kernels, test_qwen4_engine, test_mova_kernels). Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir).
