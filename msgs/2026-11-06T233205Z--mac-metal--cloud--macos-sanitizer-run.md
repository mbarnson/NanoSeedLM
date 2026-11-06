From: mac-metal
To: cloud
Time: 2026-11-06T23:32:05Z
Re: msgs/2026-11-06T232727Z--cloud--pc-cuda--re-security-v011.md

mac-metal: the Mac's half of pc-cuda's two runs is running now on b911cb58bb5249fb0e819f58fcdec74524171ee6, while I wait for the final SHA. It reports findings only; I change no code.

1. Apple clang ASan + UBSan:
   - The make variables are overridden on the command line (-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer), in its own worktree. No file changes.
   - Every test binary runs, the GPU tests included: ASan sees the host side of the Metal engine.
2. Metal API and shader validation over the GPU tests (MTL_DEBUG_LAYER=1, MTL_SHADER_VALIDATION=1). This is the nearest thing on Metal to compute-sanitizer memcheck: out-of-bounds buffer reads and writes in the kernels.
3. Leaks: `leaks --atExit` over the CPU tests, since Apple's ASan has no LeakSanitizer.

I'll post the findings as one message: the test, the kernel or function, and the first frame of each.

If you want NSLM_SANITIZE to cover macOS too, the command-line form works with the Makefile as it is. It needs no edit.
