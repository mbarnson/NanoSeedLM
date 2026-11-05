# pc-cuda claims (updated 2026-11-05T03:27:45Z)

- **Offline** (2026-11-05 ~03:30Z, Matt needs the PC). Nothing open; no pc-cuda branch in flight. main = c67e351.
- Pending for others: none from pc-cuda. windows PASS posted on df37ba8 (fast-forward of main is open to whoever has
  Matt's go once macos passes).
- Queue (CUDA files): Windows VRAM reserve default (WDDM over-commit); SEED6P8 reader for the CUDA engine; P4 gate+up
  gather (k_mv_gu_seed4p4 101-110 us vs P3 67); KV VRAM rows by active slot; plan S0.3 baselines and S0.2 STREAM triad;
  full-model GPTQ capture when Matt is around.
