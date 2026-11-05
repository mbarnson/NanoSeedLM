Status: PASS
Branch: mac/qwen4-engine
Commit: 48718c6abb15a8876dfbe1a99a4bb0e43182f57b
Platform: macos
Agent: mac-metal
Time: 2026-11-05T07:14:05Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 32 of 32 test programs (new: test_qwen4_metal),
none skipped. test_qwen4_kernels on Metal 14 of 14 (main's test). tools/test_nanoseedlm_k2.py: PASS.
