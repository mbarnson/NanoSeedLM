Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 09adc5a88a4ab000344b8b7e4904269931b55d70
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T22:16:45Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at 09adc5a (clean tree).
- cl (MSVC 14.44): ctest 41/41 PASS, test_trellis 2.64 s (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 42/42 PASS, test_trellis 1.69 s (same 3 skipped).
- No warnings in nslm/lib_trellis.c or tests/test_trellis.c.
Log: G:\devel\build-09adc5a88a4a.log.
