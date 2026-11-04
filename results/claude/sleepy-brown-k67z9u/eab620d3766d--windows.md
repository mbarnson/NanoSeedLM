Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: eab620d3766d2039c656dc87febf60c065af5f1f
Platform: windows
Agent: pc-cuda
Time: 2026-11-04T22:30:56Z

win\build.bat test on unmodified eab620d (MSVC 2022, CUDA 13.3, RTX 4080), fresh out/, incremental ninja build.
ctest 31/31 passed (test_qwen4_ref, test_qwen4_kernels, test_qwen4_engine, test_mova_kernels). Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir). P3 pack 69/98 GB downloaded; runs follow, including --stream-experts.
