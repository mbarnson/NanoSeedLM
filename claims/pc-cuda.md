# pc-cuda claims (updated 2026-10-08T23:30:00Z)

- **Waiting on mac-metal's push to `batch`** (kvq.h, EngOpts.mla_expand_min, the server's cold-cache save-race fix).
  I will not touch shared files until it lands.
- **Next, after rebasing on that push** (CUDA files only unless noted):
  1. Port `nslm/kvq.h` latent formats to CUDA (`engine/mova_cuda.c`, `engine/kernels_moe.cu`), check them against the
     shared C reference, and measure KLD on the healed MLA P=4 folder.
  2. `mla_expand_min` on CUDA: stay absorbed for short prompt passes.
  3. CUDA KV pool: give VRAM rows to the active slots, not all to slot 0 (today slot 1 of `--max-seqs 2` sits entirely
     in mapped host memory).
- Linux: setting up WSL Ubuntu so I can post `linux` results.
