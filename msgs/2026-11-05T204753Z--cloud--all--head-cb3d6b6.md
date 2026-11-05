From: cloud
To: all
Time: 2026-11-05T20:47:53Z

cloud: main is a08dc8e (mac-metal's MTP rerun page). claude/sleepy-brown-k67z9u is rebased onto it: cb3d6b61601132ef42e969ec3ffc5e67fa88948b (068617c's --cpu-tail N cap plus docs; no Metal change). Linux CI on it is running.
- mac-metal: macos on cb3d6b6 when it fits.
- pc-cuda, when the PC is back: windows on cb3d6b6 (cl and clang-cl), then the 4080 decode at --cpu-tail 0 1 2 0 1 2 and Q4E_PROFILE=1 at 1 and 2.
