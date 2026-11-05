Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 047ad9ffe63e210775472744b620ee31799e6987
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T05:07:58Z

cmake --build build + ctest on 047ad9f (main df37ba8 + 11; MSVC 14.44, CUDA 13.1, RTX 4080), incremental build:
ctest 34/34 passed (test_emulate, test_searchc, test_searchc_gpu, test_qwen4_*, test_mova_kernels included).
Skipped: test_affine, test_mova_cfg, test_tokenizer (no model dir), as on df37ba8. Time: 2026-11-05T05:07:58Z
