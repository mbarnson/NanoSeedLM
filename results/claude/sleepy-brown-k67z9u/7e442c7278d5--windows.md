Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 7e442c7278d5c23e2a8662d1bffc02d070df6a85
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T23:48:40Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at 7e442c7 (clean tree; contains ece9722 and 7dbac2b).
- cl (MSVC 14.44): ctest 42/42 PASS, test_trellis_gpu (two threads, T 256 and 320) 2.51 s (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 43/43 PASS, test_trellis_gpu 2.37 s (same 3 skipped).
Log: G:\devel\build-7e442c7278d5.log.
