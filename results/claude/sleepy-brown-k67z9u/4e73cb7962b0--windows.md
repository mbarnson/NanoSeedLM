Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 4e73cb7962b0ed72cffe79de31e1b16d0ff9a303
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T02:34:25Z

win\build.bat test, clean build in a fresh worktree of 4e73cb7 (MSVC 14.44, CUDA 13.1, RTX 4080, driver 610.88):
ctest 31/31 passed (test_qwen4_ref, test_qwen4_kernels, test_qwen4_engine, test_mova_kernels). Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir). (25ae5f9 not run: superseded before I got to it.)
