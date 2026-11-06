Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 354e2e1b614d5bfd3336c52ba8d984a38245586a
Platform: windows
Agent: pc-cuda
Time: 2026-11-06T13:18:15Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at 354e2e1 (clean tree, reconfigured; contains 3150d4b).
- cl (MSVC 14.44): ctest 43/43 PASS (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 44/44 PASS (same 3 skipped).
- The new default, Q4E_PASS_ROWS unset, release TQ3-D4, --ctx 32768, automatic cache: 3004 slots, decode 38.23 / 38.19 tok/s, md5 16994974. The same as setting (b) of the grid.
Log: G:\devel\build-354e2e1b614d.log.
