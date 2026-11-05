# mac-metal claims (updated 2026-11-05T14:55:20Z)

- **Done (2026-11-05 14:52Z):** the Metal qwen4 engine is on main 22675d8 (pc-cuda's fast-forward, Matt's go; green on
  all three). P3 on the M4 Max: decode 42-43 tok/s short, 36-38 after 19.8K tokens; prompt 400-440 tok/s. nslm-chat
  and nslm-serve run Qwen3.8-Flash-Next on Metal.
- **Done:** MTP step M1's golden: mac/qwen4-mtp-golden 0936d38 (tests/data/qwen4_tiny/model-mtp.safetensors and
  tests/data/qwen4_tiny_mtp_golden.safetensors on cloud's b92e85d), for cloud to take.
- **Now (Matt: MTP go):** the Metal side of MTP. First the k_mv Q4 path at 40 blocks a row equal for T = 1 and T > 1.
  Then per-row recurrent state in k_q4_gdn_step / gdn_conv / ple_conv (for an accept-m verify API, as proposed to
  cloud), then the draft head on Metal once qwen4_ref_mtp is on main. Files: engine/*.metal, engine/qwen4_gpu.m (mine).
- Later: QSA indexer at long context; prompt GEMM tile loads; llama.cpp (Metal) on UD-Q4_K_XL as the Mac baseline.
