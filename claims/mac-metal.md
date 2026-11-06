# mac-metal claims (updated 2026-11-06T01:31:39Z)


- **Running (cloud's go):** the trellis encodes for gate TC on the Mac, on the Metal GPU (nslm-moe b301be5, 2 workers). K = 3, L = 12 is at 68 of 144 at 01:31Z, ETA about 02:25Z (~/models/Qwen3.8-Flash-Next-tq-k3-l12). Then K = 4 starts by itself, ETA about 04:10Z (~/models/Qwen3.8-Flash-Next-tq-k4-l12). No Mac benchmarks while they run.
- **Next:** gate TC on the M4 Max, decode and prompt against P3 for TQ3L12, TQ4L12 and the "down" mix (TQ3 gate/up + TQ4 down, as cloud's 01:16Z message to pc-cuda). Code and chat, 256 tokens, two rounds, K = 0 and drafts, Q4E_PROFILE=3 on the experts, plus prompt runs.
- **Done (main 6f90957):** trellis experts on Metal (gathers, prompt GEMM, qwen4_gpu.m); the Metal trellis encoder; MTP drafts on Metal and in nslm-chat; the day's Metal decode and prompt kernels.
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m.
- **Rule (Matt):** I push only mac/* branches and post them here; main merges are cloud's. Nothing on HF without a message to cloud first.
