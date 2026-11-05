Status: PASS
Branch: pc/p4-gather
Commit: c67e3511d7e3a94d68a51926acde5fb90247a99c
Platform: macos
Agent: mac-metal
Time: 2026-11-05T03:17:07Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (fresh worktree): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 29 of 29 test programs, none skipped.
test_qwen4_kernels on Metal: all 14 checks pass. tools/test_nanoseedlm_k2.py: PASS. The CUDA change (engine/kernels_moe.cu)
is not built on macOS.
