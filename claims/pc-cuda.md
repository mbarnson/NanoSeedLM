# pc-cuda claims (updated 2026-10-09T00:52:56Z)

- **Done:** CUDA fp8 / fp4 + mla_expand_min (on main at 6364ffb).
- **Next (CUDA files only):**
  1. fp4/fp8 decode speed on CUDA: k_mla_attn_tc decodes the cache with plain stores (16k: 38.3 vs BF16 46.0 tok/s);
     stage the codes with cp.async and decode in shared memory.
  2. CUDA KV pool: give the VRAM rows to the active slots, not all to slot 0.
- Re-run windows for mac/kv-disk-age once it is rebased on batch.
- Linux: no toolkit in WSL yet; Matt's call.
