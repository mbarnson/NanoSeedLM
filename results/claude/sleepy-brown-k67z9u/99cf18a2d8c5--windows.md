Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 99cf18a2d8c56fb93736ffe07c3ab96b9b4bab33
Platform: windows
Agent: pc-cuda
Time: 2026-11-06T05:47:11Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at 99cf18a (clean tree, reconfigured).
- cl (MSVC 14.44): ctest 43/43 PASS, test_moe 0.15 s, test_qwen4_engine (the head by the routers) 3.23 s (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 44/44 PASS (same 3 skipped).
- A first run failed to link test_moe (nslm_qwen4_pack_tq unresolved). That was my CI's fault, not the code's: my incremental builds did not re-run CMake, so the GLOB of nslm/lib_*.c missed the new lib_qwen4_pack.c. My CI now reconfigures every run.
Log: G:\devel\build-99cf18a.log.
