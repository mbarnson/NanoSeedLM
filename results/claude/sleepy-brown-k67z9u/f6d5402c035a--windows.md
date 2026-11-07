Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: f6d5402c035a7824f219105b720ee8cf92292dff
Platform: windows
Agent: pc-cuda
Time: 2026-11-07T02:06:05Z

Windows 11, RTX 4080 16 GB, driver 610.88, CUDA 13.1, MSVC 14.44, clang-cl. Worktree G:/devel/NanoSeedLM-rel at f6d5402c035a7824f219105b720ee8cf92292dff; cmake reconfigured, Release, Ninja.
- cl (build): 100% tests passed out of 46.
- clang-cl (build-clang): 100% tests passed out of 47 (test_cpu_seed4 is built only by clang).
- The GPU tests ran alongside a compute-sanitizer racecheck of b911cb5's test_engine. They passed anyway.
- Gate HC: nslm-serve.exe from both builds shows Control Flow Guard in dumpbin /headers, and CF instrumented in /loadconfig.
