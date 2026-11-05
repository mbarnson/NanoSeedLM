Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: f94ffaad0c3682cde7c6724cc597bf5b6e6e5bf2
Platform: macos
Agent: mac-metal
Time: 2026-11-05T02:47:10Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 29 of 29 test programs, none skipped.
test_qwen4_kernels runs on Metal (main c404931's kernels) against this branch's test: all 14 checks pass.
tools/test_nanoseedlm_k2.py (oMLX venv, OMLX_K2_MODEL set): PASS. The tree differs from my trial rebase 72e90d9 in
docs only. The CUDA changes are not built on macOS.
