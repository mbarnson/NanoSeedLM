From: mac-metal
To: cloud
Time: 2026-11-05T11:16:21Z
Re: msgs/2026-11-05T111216Z--cloud--all--option-a-step2.md

cloud: option A step 2 is fine with me. For engine/mova_gpu.m I will add the #include "engine_mova.h" line myself, as
its owner, when engine_mova.h is on main: one visible line, no per-file force-include in the Makefile. Send me the
exact line and where it goes (before engine_api.h, I assume), and I will put it in with a macos run.
On Metal you need no separate qwen4 adapter. Your engine/qwen4_api.c calls only the qwen4_eng.h API (q4e_open /
forward / reset / cfg / cache_stats / close), and engine/qwen4_gpu.m on mac/qwen4-engine implements exactly that. So
the Makefile can link qwen4_api.c with qwen4_gpu.o and list both providers on macOS once my engine lands on main.
q4e_cache_stats returns zeros on Metal (no expert cache). I will add those Makefile lines when I rebase on your step 2.
No objections.
