# pc-cuda claims (updated 2026-11-05T03:18:02Z)

- Nothing open. Done: k_mv1_* on main c67e351.
- Next for others: windows on cloud's rebased claude/sleepy-brown-k67z9u head (ad7f276 + rebase).
- Queue (CUDA files): SEED6P8 reader for the CUDA engine; Windows VRAM reserve default (WDDM over-commit); the gather's
  gate+up for P4 (k_mv_gu_seed4p4 101-110 us vs P3 67); KV VRAM rows by active slot; plan S0.3 baselines and S0.2 STREAM
  triad; full-model GPTQ capture when Matt is around.
