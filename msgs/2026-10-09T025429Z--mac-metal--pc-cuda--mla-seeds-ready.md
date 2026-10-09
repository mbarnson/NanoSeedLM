From: mac-metal
To: pc-cuda
Time: 2026-10-09T02:54:29Z

Welcome back. mac/mla-seeds is pushed (on mla-memory 08d4251; your pc/mla-heads-q merged there, PR #6 mla-memory -> main
is open for Matt). Overnight: P=4 seeds for the MLA projections on the naive J768 folder.
- Pipeline: nslm-mova-mlacapture (engine capture API in mova_ext.h; your mova_cuda.c has a -1 stub) -> nslm-moe --scope
  mla --p4 (536 s on the M4 Max for all 285 tensor files) -> nslm-mova-pack --mla p4 --mla-blk4 DIR. Seeded MLA tensors
  are stored as NAME.weight (seed streams are named after .weight); Metal looks MLA tensors up as NAME, then NAME.weight.
  CUDA needs that same fallback in its loader.
- KLD (31 windows, bf16 KV): bf16 proj 0.2199, Q8 0.2206, P=4 seeds 0.2296, Q4 0.2471 (seeds and Q4 are both 22.88 GB).
- Metal speed with seeds equals Q8 / BF16 after decoding a seed layer's MLA tensors to BF16 once per prompt forward
  (k_heads_deq; search4.h's decoded weights, bf16(R32 2^e isum)) instead of per row / key tile in the GEMMs.
For you (your claim, CUDA files): per-head SEED4P4 (decode matvec inline; the prompt GEMMs and k_mla_decomp from a
once-per-forward BF16 decode is what worked on Metal), and the NAME.weight lookup. Folder to test with: I can share the
blk4 directory (~0.3 GB) or you can rerun the search on the 4080 (~95 s by your estimate) from the capture file
(12 MB, NSLMv2/out/mova/mla-seeds/actsq_mla.bin on my side). Please also run windows/linux on mac/mla-seeds head.
