Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 41a8fe6c9f7d9ba8d1d009e7aab0cc9fadc6c4a8
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T22:47:11Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at 41a8fe6 (clean tree).
- cl (MSVC 14.44): ctest 41/41 PASS, test_trellis 1.32 s (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 42/42 PASS, test_trellis 0.97 s (same 3 skipped).
- No warnings in nslm/lib_trellis.c, tests/test_trellis.c or nslm/moe.c.
Log: G:\devel\build-41a8fe6c9f7d.log.
