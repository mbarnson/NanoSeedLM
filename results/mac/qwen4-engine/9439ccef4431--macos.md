Status: PASS
Branch: mac/qwen4-engine
Commit: 9439ccef44311207ee487f23c6bdb7e5fa08408e
Platform: macos
Agent: mac-metal
Time: 2026-11-05T14:05:35Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 34 of 34 test programs, none skipped;
test_qwen4_kernels with KT_ATTEND_SPLIT=8: PASS. tools/test_nanoseedlm_k2.py: PASS.
nslm-serve on the P3 pack, two chat turns (greedy, reasoning_effort low, reasoning_content sent back): both stop; turn 2
119 prompt tokens with 98 cached, ttft 0.40 s (turn 1: 1.23 s), decode 38.8 tok/s.
