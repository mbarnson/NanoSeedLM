Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: a499194e3ca57f8306c181319b36e493429a1c03
Platform: windows
Agent: pc-cuda
Time: 2026-11-06T10:47:25Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at a499194 (fetched by SHA after the rebase; clean tree, reconfigured).
- cl (MSVC 14.44): ctest 43/43 PASS (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 44/44 PASS (same 3 skipped). No warnings in engine/qwen4_api.c or engine/qwen4_eng.h.
Log: G:\devel\build-a499194e3ca5.log.
