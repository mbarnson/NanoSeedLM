Status: PASS
Branch: mac/qwen4-engine
Commit: 89093ce18bcf3fddabb8ca6a87b96f475e472edc
Platform: macos
Agent: mac-metal
Time: 2026-11-05T11:57:26Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 34 of 34 test programs, none skipped
(test_engine through engine/engine.c; test_qwen4_metal with qwen4_tiny_attn). tools/test_nanoseedlm_k2.py: PASS.
