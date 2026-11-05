Status: PASS
Branch: pc/q4-splitk
Commit: b400cf7d26c5439718c6d29bbaf03eec54d112a3
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T14:54:54Z

cmake --build build + ctest on b400cf7 (pc/q4-splitk on main 22675d8; MSVC 14.44, CUDA 13.1, RTX 4080): 38/38 passed.
test_mv K = 4096 (k_mv_q4sk) and T = 1 vs 2..8 bit equality pass for every format. Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir). 2026-11-05T14:54:54Z
