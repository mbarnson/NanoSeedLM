Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 1930db3e2f1509e33a4e6164e638646e3e043c37
Platform: macos
Agent: mac-metal
Time: 2026-11-05T06:33:13Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (fresh worktree): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 31 of 31 test programs, none skipped;
test_qwen4_kernels on Metal: 15 of 15 (the new 12-row attention case 2.1e-7).
tools/test_nanoseedlm_k2.py (oMLX venv, OMLX_K2_MODEL set): PASS, 28 tests ok (test_load_mla and _affine included).
1930db3 differs from the asked 18be2bf only in engine/qwen4_cuda.c (Q4E_PROFILE stages; not built on macOS).
