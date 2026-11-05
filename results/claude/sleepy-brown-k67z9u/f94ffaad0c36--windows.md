Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: f94ffaad0c3682cde7c6724cc597bf5b6e6e5bf2
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T02:43:16Z

win\build.bat test, clean build in a fresh worktree of f94ffaa (MSVC 14.44, CUDA 13.1, RTX 4080, driver 610.88):
ctest 31/31 passed (test_qwen4_ref, test_qwen4_kernels, test_qwen4_engine, test_mova_kernels). Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir).
