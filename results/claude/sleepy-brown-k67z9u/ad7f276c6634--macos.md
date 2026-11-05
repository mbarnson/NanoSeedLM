Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: ad7f276c6634c70c9d63c97a9c3a3048e3d0b259
Platform: macos
Agent: mac-metal
Time: 2026-11-05T03:18:18Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157);
nslm/lib_searchc.c and nslm/moe.c's --probe build without warnings. make test with MOVA_DIR = the K2-Horizon-MoVA
snapshot: all tests passed, 30 of 30 test programs (the new test_searchc: PASS), none skipped. test_qwen4_kernels on
Metal: all 14 checks pass. tools/test_nanoseedlm_k2.py: PASS.
