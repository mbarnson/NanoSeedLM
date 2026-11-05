Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 047ad9ffe63e210775472744b620ee31799e6987
Platform: macos
Agent: mac-metal
Time: 2026-11-05T05:14:32Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 31 of 31 test programs, none skipped.
test_qwen4_kernels on Metal: all 14 checks pass. tools/test_nanoseedlm_k2.py: PASS.
