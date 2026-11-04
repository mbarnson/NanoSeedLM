# pc-cuda claims (updated 2026-11-04T23:41:15Z)

- **Now (2026-11-04, Matt's go):** the fp4 MLA decode gap vs BF16 on CUDA, branch `pc/fp4-decode-gap` from main db0ea15.
  Files: engine/kernels_moe.cu (k_mla_attn_tc decode path), maybe engine/kernels_cuda.h, tests/kernel_backend_cuda.c
  (all mine). No shared files planned; I'll message first if that changes. Model: txgsync/K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4.
- Queue (CUDA files): KV VRAM rows by active slot; full-model GPTQ capture when Matt is around.
