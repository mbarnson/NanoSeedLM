# mac-metal claims (updated 2026-11-05T22:52:45Z)


- **Running (cloud's go, 22:36Z):** the full trellis encode, K = 3, L = 12, all 48 layers (gate, up, down), on the Mac CPU. nslm-moe from 41a8fe6 in ~/devel/nslm-tq, --workers 12, under caffeinate. Started 2026-11-05T22:51:38Z, about 9.8 h (estimate 08:40Z). Output in ~/models/Qwen3.8-Flash-Next-tq-k3-l12. It resumes: a finished .blkt is skipped. No Mac benchmarks while it runs.
- **Next:** a Metal Viterbi encoder for L = 16, exactly nslm_tq_encode_row (new files, a test against the C encoder, a Makefile rule).
- **Done (on main):** MTP drafts on Metal and in nslm-chat (Q4E_DRAFTS); the day's Metal decode and prompt kernels; the T3 preview (trellis matvec on Metal as fast as P3's seed matvec, exact).
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c.
- **Later:** fuse the hc inject rows into the down GEMM; prompt attention; a chunked DeltaNet for prompt rows.
