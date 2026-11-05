Status: PASS
Branch: pc/fp4-decode-gap
Commit: a085dce8072473b07a5026b47d2a46e9407e090d
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T02:45:04Z

win\build.bat test on a085dce (pc/fp4-decode-gap, on main c404931; MSVC 14.44, CUDA 13.1, RTX 4080): ctest 31/31 passed,
incl. test_mova_kernels' two new 40-split latent-attention cases with long rows (0 mismatches, FP8 / FP4 bit-equal to
BF16). Skipped: test_affine, test_mova_cfg, test_tokenizer (no model dir).
