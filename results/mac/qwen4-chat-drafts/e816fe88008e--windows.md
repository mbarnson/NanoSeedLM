Status: PASS
Branch: mac/qwen4-chat-drafts
Commit: e816fe88008ecc
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T21:47:53Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at e816fe8 (clean tree).
- cl (MSVC 14.44): ctest 40/40 PASS, test_qwen4_api_mtp included (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 41/41 PASS (same 3 skipped).
- No warnings in engine/qwen4_api.c, harness/nslm-chat.c or tests/test_qwen4_api_mtp.c.
Log: G:\devel\build-e816fe88008e.log.
