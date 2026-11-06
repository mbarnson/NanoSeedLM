# mac-metal claims (updated 2026-11-06T03:54:28Z)


- **Done:** gate TC on the M4 Max (msgs 03:54Z): P3, TQ3L12, TQ4L12 and the down mix. At K = 0 the trellis packs decode 1.7 to 2.3% faster than P3, and prompts run at P3's speed warm. The memory line is the pack's files (P3 98.2, TQ3 84.9, mix 90.0, TQ4 100.1 GB).
- **Kept on the Mac:** ~/models/Qwen3.8-Flash-Next-tq-k3-l12 and -tq-k4-l12 (my Metal-GPU encodes, byte-equal to the bucket's), with the TQ3L12, TQ4L12 and TQ3TQ4down pack folders of links.
- **Offered:** the PLE table's working set (resident pages after a long run), which decides whether TQ3 fits a 64 GB Mac. Also the L = 16 Metal encoder speed (a layout with 8x fewer threadgroup reads).
- **Done (main):** trellis experts on Metal; the Metal trellis encoder; MTP drafts on Metal and in nslm-chat; the day's Metal decode and prompt kernels.
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m.
- **Rules (Matt):** I push only mac/* branches and post them here; main merges are cloud's. Nothing on HF without a message to cloud first.
