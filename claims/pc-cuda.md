# pc-cuda claims (updated 2026-10-09T00:02:13Z)

- **Now, on `pc/mla-kvq` (from batch 1551e9e):** CUDA MLA `--kv fp8` / `--kv fp4` per nslm/kvq.h, and CUDA
  `mla_expand_min` (expand only for prefills of 256+ new tokens, up to the last multiple). Files: `engine/mova_cuda.c`,
  `engine/kernels_moe.cu`, `engine/kernels_cuda.h`, `tests/kernel_backend_cuda.c`. Shared files: none planned.
- **Next:** CUDA KV pool VRAM rows by active slot (not all to slot 0); WSL Ubuntu for `linux` results.
