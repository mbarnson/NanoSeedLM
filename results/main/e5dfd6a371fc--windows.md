Status: PASS
Branch: main
Commit: e5dfd6a371fc6a3dbcbe72303a419ec41f8abd2c
Platform: windows
Agent: pc-cuda
Time: 2026-11-07T03:56:29Z

Windows 11, RTX 4080 16 GB, driver 610.88, CUDA 13.1. Cloud's short checks of the final v0.1.1 draft (optional; they don't hold the release).

**Tests at e5dfd6a371fc6a3dbcbe72303a419ec41f8abd2c** (G:/devel/NanoSeedLM-rel; cmake reconfigured, Release, Ninja):
- cl: 100% tests passed out of 46.
- clang-cl: 100% tests passed out of 47.

**The draft's nslm-v0.1.1-windows-x64-cuda12.8.zip** (assets of 03:53:45Z, release run 42335523960):
- SHA-256 f2c18fb83bec2f17be5b72d2ac9b42fe73083d8bf055fbc6b79d3045246659b8, the same as SHA256SUMS.
- start-here.ps1 equals git e5dfd6a's start/start-here.ps1, apart from CRLF. It is also byte-identical to the a8a3416 zip's, which passed the full upgrade test (results/main/a8a34166c434--windows.md).
- README.txt line 1 names e5dfd6a371fc6a3dbcbe72303a419ec41f8abd2c. Line 39 says "about 61 GB for a TCQ pack".
- dumpbin on the shipped build/bin/nslm-serve.exe and nslm-chat.exe:
  - /headers: Control Flow Guard, High Entropy Virtual Addresses, Dynamic base, NX compatible.
  - /loadconfig: CF instrumented.
