Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: f4d997d8ad509631c524838c37b1a76636414356
Platform: windows
Agent: pc-cuda
Time: 2026-11-06T10:21:29Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release.
- G:\devel\NanoSeedLM-lfsr at f4d997d (reconfigured): cl ctest 43/43 PASS, clang-cl 44/44 PASS (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- The release path, as release.yml runs it: a clean worktree, NSLM_CMAKE_ARGS=-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded, win\build.bat test. It configured with the static runtime (CMakeCache: MultiThreaded), and ctest gave 43/43 PASS.
- dumpbin /dependents on that build: nslm-chat.exe needs icu.dll, SHELL32.dll and KERNEL32.dll; nslm-serve.exe adds bcrypt.dll and WS2_32.dll. Neither needs VCRUNTIME, MSVCP or a cudart DLL. icu.dll is Windows' own (Windows 10 1903 and later), so the zip runs on a clean PC with only the NVIDIA driver.
Logs: G:\devel\build-f4d997d.log, G:\devel\build-rel-f4d997d.log.
