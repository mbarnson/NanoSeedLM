# pc-cuda claims (updated 2026-11-05T19:54:18Z)

# pc-cuda claims (updated 2026-11-05T19:58Z)

- **Offline** for a few hours from ~19:55Z: Matt uses the PC. Nothing runs on it. Back when Matt says so.
- Owed on return: windows on mac/qwen4-decode b910034 and claude/sleepy-brown-k67z9u c18e7ac (or newer; cl and clang-cl);
  gate DC on the 4080 (--cpu-tail 0 7 0 7, KL at 0 and 7, one Q4E_PROFILE=1 run).
- Done today: windows CI on everything posted; LFSR halves on CUDA (main); pc/win-clang-cpu (in cloud's branch); gates DL, DC (pre-fix),
  DA, D1 check, M3 rerun, D2 probe (fails DB), Strata probe, CPU seed probe, P3H search (byte-identical to mac-metal's).
