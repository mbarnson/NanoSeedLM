# pc-cuda claims (updated 2026-10-09T00:56:48Z)

- **Next: CUDA per-head Q8 / Q4 for MLA HEADS tensors** (q_rope_mix, q_lat, v_up) on `pc/mla-heads-q` from mla-memory,
  after mac-metal's packer lands (mac/mla-pack). Files: engine/kernels_moe.cu, engine/kernels_cuda.h, engine/mova_cuda.c,
  tests/kernel_backend_cuda.c. KLD on the healed folder.
- Then: fp4 / fp8 decode speed (k_mla_attn_tc staging codes with cp.async); CUDA KV VRAM rows by active slot.
- Re-run windows for mac/kv-disk-age once it is rebased.
