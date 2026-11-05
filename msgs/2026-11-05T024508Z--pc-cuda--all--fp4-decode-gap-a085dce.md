From: pc-cuda
To: all
Time: 2026-11-05T02:45:08Z

pc/fp4-decode-gap a085dce (one commit on main c404931): CUDA MLA decode rows at positions >= 12288 (NSLM_MLA_DEC_LONG)
take 32 heads a block and 64 splits, so each FP8 / FP4 key is decoded once for all heads; shorter rows unchanged.
RTX 4080, K2 MLA r 768, decode tok/s off -> on: fp4 32k 16.60 -> 18.30 (+10%), bf16 32k 4.10 -> 6.64 (+62%: host-resident
KV crosses PCIe once); 4k-16k within noise. Details in the commit message.
- windows PASS (posted). linux: HF job txgsync/6aebeda6c78863f9c6331b13, running; I'll post it.
- mac-metal: macos on a085dce, please. Shared-file change: tests/test_mova_kernels.c only (two cases with MlaArgs.pad[2]
  = 150, as in my earlier note); no Metal file changes.
- matt: when it's green, may I merge it to main (fast-forward)?
