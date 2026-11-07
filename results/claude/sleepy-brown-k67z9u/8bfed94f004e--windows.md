Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 8bfed94f004ec0798bc7fe6faf5695eeb00c7afa
Platform: windows
Agent: pc-cuda
Time: 2026-11-07T00:55:17Z

Windows 11, RTX 4080 16 GB, driver 610.88, CUDA 13.1, MSVC 14.44, clang-cl (LLVM). Worktree G:/devel/NanoSeedLM-rel at 8bfed94f004ec0798bc7fe6faf5695eeb00c7afa; cmake reconfigured, Release, Ninja.

- cl (build): 100% tests passed out of 46.
- clang-cl (build-clang): 100% tests passed out of 47. The extra one is test_cpu_seed4, which is built only by clang. Both lists gained test_serve_guard over b911cb5 (45 and 46).
- The GPU tests ran alongside a compute-sanitizer racecheck of b911cb5's test_engine, which had the GPU at 100%. They passed anyway.
- Gate HC (dumpbin on nslm-serve.exe): PASS for both builds.
  - `/headers` DLL characteristics: Control Flow Guard, High Entropy Virtual Addresses, Dynamic base, NX compatible.
  - `/loadconfig`: CF instrumented, FID table present. Guard Flags are 10017500 for cl and 00010500 for clang-cl; Guard CF function counts are 0x558 and 0x51B.
