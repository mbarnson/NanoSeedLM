Status: PASS
Branch: mac/qwen4-engine
Commit: 3fefabcdedc894218842b6b456c0bbd21bc9730e
Platform: macos
Agent: mac-metal
Time: 2026-11-05T13:35:25Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 34 of 34 test programs, none skipped;
test_qwen4_kernels with KT_ATTEND_SPLIT=8: PASS. tools/test_nanoseedlm_k2.py: PASS. nslm-chat on the P3 pack (--no-think):
"Common tide pool animals include sea stars, sea anemones, and crabs." (17 tokens, stopped at the turn's end).
