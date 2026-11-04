Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 79cc53b67b935352a0d807861005ffc2a5b607f8
Platform: windows
Agent: pc-cuda
Time: 2026-11-04T22:30:00Z

win\build.bat test, unmodified 79cc53b (MSVC 2022, CUDA 13.3, RTX 4080, driver 610.88), fresh out/; incremental ninja
build over the 384bc45 build dir (all changed files rebuilt). ctest 31/31 passed: test_qwen4_ref, test_qwen4_kernels,
test_qwen4_engine, test_mova_kernels included. Skipped: test_affine, test_mova_cfg, test_tokenizer (no model dir).
