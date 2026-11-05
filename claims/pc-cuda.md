# pc-cuda claims (updated 2026-11-05T02:49:10Z)

- Nothing open. Done: the fp4 MLA decode gap is on main a085dce (rows >= 12288 positions take 32-head blocks).
- Next for others: windows on cloud's rebased claude/sleepy-brown-k67z9u head when posted.
- Queue (CUDA files): P4 down-projection gather (K = 640 misses mv_lane's paired SEED4P4 path; 75.5 GB/s on the 4080);
  SEED6P8 reader for the CUDA engine; Windows VRAM reserve default (WDDM over-commit); KV VRAM rows by active slot;
  plan S0.3 baselines (Strata, llama.cpp on UD-Q4_K_XL) and S0.2 STREAM triad; full-model GPTQ capture when Matt is around.
