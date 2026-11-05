# pc-cuda claims (updated 2026-11-05T02:52:55Z)

- **Now:** the P4 (SEED4P4) decode gather at K = 640 (qwen4 down 2560 x 640: 122 us, 75.5 GB/s on the 4080; mv_lane's
  paired path needs nbk % 64 == 0). Branch `pc/p4-gather` from main a085dce. Files: engine/kernels_moe.cu (mine).
- Waiting: linux and macos on PR #17 head 350481b, then I fast-forward main to it.
- Queue (CUDA files): SEED6P8 reader for the CUDA engine; Windows VRAM reserve default (WDDM over-commit); KV VRAM rows by
  active slot; plan S0.3 baselines and S0.2 STREAM triad; full-model GPTQ capture when Matt is around.
