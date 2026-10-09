# pc-cuda claims (updated 2026-10-09T01:45:51Z)

- **pc/mla-heads-q (08d4251, on mla-memory f612209), pushed:** CUDA per-head Q8 / Q4. Windows PASS. Linux job txgsync/6ac84748fee2c900701722ed started
  2026-10-09 ~01:50Z (check and post). Speed benchmark (BF16 vs Q8 projections, fp4 KV, 4k / 16k) still to run. Then
  ask Matt to merge it into mla-memory.
- **Next, CUDA files only:** fp4 / fp8 decode speed in k_mla_attn_tc (load the next tile's codes into registers before
  computing the current tile, decode into shared memory at the end); KV VRAM rows by active slot; then CUDA per-head
  SEED4P4 after mac/mla-seeds.
- Paused overnight: Matt is using the GPU.
