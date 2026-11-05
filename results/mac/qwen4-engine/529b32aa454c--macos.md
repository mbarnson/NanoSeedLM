Status: PASS
Branch: mac/qwen4-engine
Commit: 529b32aa454ce578392effcc2ef209f1bb037549
Platform: macos
Agent: mac-metal
Time: 2026-11-05T13:04:04Z

macOS 27.2, M4 Max 128 GB, AC power. make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 34 of 34
test programs, none skipped (the code tree was also clean-built: rm -rf out, the one old warning). test_qwen4_kernels
with KT_ATTEND_SPLIT=2 / 5 / 8 / 16 (the split decode attention) against nslm/qwen4_ops.h: PASS. tools/test_nanoseedlm_k2.py: PASS.
