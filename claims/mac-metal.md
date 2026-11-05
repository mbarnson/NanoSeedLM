# mac-metal claims (updated 2026-11-05T23:19:22Z)


- **Running (cloud's go, 22:36Z):** the full trellis encode, K = 3, L = 12, all 48 layers (gate, up, down), on the Mac CPU. nslm-moe from 41a8fe6 in ~/devel/nslm-tq, --workers 12, under caffeinate, pid 98044. Started 2026-11-05T22:51:38Z, estimate 08:40Z. Output in ~/models/Qwen3.8-Flash-Next-tq-k3-l12. The bucket sync waits for Matt's OK in this session (script ~/models/sync-tq.sh).
- **Now:** the Metal trellis encoder, mac/trellis-metal acc4090 on pc/trellis-cuda 82882e2 (local; I push after the rebase onto main once pc/trellis-cuda lands). Exact: test_trellis_gpu passes unchanged, and an L 6..16 sweep shows no differences. L 12 at 11 to 13 M weights a second; L 16 still slow, so its layout is next. Files: nslm/trellis.metal, nslm/trellis_metal.m; shared lines in trellis_gpu.h, test_trellis_gpu.c, the Makefile and parity.md (cloud's go).
- **Proposed:** nslm-moe --gpu for the tq codec, then K = 4 on the GPU (about 3 h).
- **Done (on main):** MTP drafts on Metal and in nslm-chat (Q4E_DRAFTS); the day's Metal decode and prompt kernels; the T3 preview (trellis matvec as fast as P3's seeds, exact).
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m.
