Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: df37ba8baff38c5f7f81cff776f6df8a97518dab
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T03:27:25Z

win\build.bat test on df37ba8 (on main c67e351; MSVC 14.44, CUDA 13.1, RTX 4080), fresh out/, incremental build:
ctest 33/33 passed (test_emulate, test_searchc, test_qwen4_*, test_mova_kernels included). Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir).
