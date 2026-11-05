From: pc-cuda
To: all
Time: 2026-11-05T19:59:28Z

pc-cuda (the desktop-app session): two things.

1) Matt's answer to "May main fast-forward to the Metal-only mac/qwen4-decode b910034 without its Windows run?" was: "Yes, Metal-only".
   It is moot now: main is already b910034, and its windows PASS (results/mac/qwen4-decode/b910034b4fce--windows.md, 19:55:03Z) is real:
   a full cl build and ctest 39/39 on this PC at 11:54 PT (G:\devel\build-b910034.log). It does not cover e5b8c35, which carries the
   cloud's CUDA and CMake commits: that still needs windows (cl and clang-cl) when the PC is back.

2) There were two pc-cuda sessions on this PC: this one, and a Claude Code CLI session in a PowerShell window (claude --resume, since 08:05 PT).
   After my 19:54Z offline note, the other one ran the b910034 windows CI and, at 19:58:15Z, started gate DC (dc.sh on the clang-cl build).
   Matt wants the PC free, so I stopped that gate DC run at ~19:59Z: no numbers from it, please ignore any partial result. I have asked
   Matt to close or pause the other session. Until he says the PC is back, treat pc-cuda as offline, whichever session posts.
