Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: f2caa8b4b63806e7d7b252618662536029d4f890
Platform: macos
Agent: mac-metal
Time: 2026-11-05T07:32:58Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (fresh worktree): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 31 of 31 test programs, none skipped;
test_qwen4_kernels on Metal 15 of 15. tools/test_nanoseedlm_k2.py: PASS. f2caa8b is 825dd76 plus two commits in
engine/kernels_qwen4.cu only (not built on macOS).
