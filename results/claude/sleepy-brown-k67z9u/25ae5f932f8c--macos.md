Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 25ae5f932f8c168ebd0b7d3e71db21f5470e1b15
Platform: macos
Agent: mac-metal
Time: 2026-11-05T02:11:08Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 29 of 29 test programs, none skipped.
test_qwen4_kernels prints "not on Metal" for the qwen4 ops (this branch has no Metal qwen4 kernels) and passes.
tools/test_nanoseedlm_k2.py (oMLX venv, OMLX_K2_MODEL set): PASS. The CUDA changes are not built on macOS.
