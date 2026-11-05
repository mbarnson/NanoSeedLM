Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 350481b85e7d77f5b0654bc5594247c98183a375
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T02:52:19Z

win\build.bat test on 350481b (PR #17 rebased on main a085dce; MSVC 14.44, CUDA 13.1, RTX 4080), fresh out/, incremental
build over the f94ffaa worktree's build dir (every changed file rebuilt): ctest 31/31 passed (test_qwen4_ref,
test_qwen4_kernels, test_qwen4_engine, test_mova_kernels incl. the long-row MLA cases). Skipped: test_affine,
test_mova_cfg, test_tokenizer (no model dir).
