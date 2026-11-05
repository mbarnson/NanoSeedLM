# mac-metal claims (updated 2026-11-05T21:45:55Z)

# mac-metal claims (updated 2026-11-05T21:46Z)

- **Done (on main):**
  - MTP drafts on Metal (plan step M4): the head, q4e_draft / q4e_forward_keep / q4e_accept, and verify rows bit for bit equal to decode rows.
  - The day's Metal decode kernels: P3 decode 41.3 -> 48.9 tok/s without drafts, 61.5 with 1 draft.
  - Prompt GEMM k_mm64: 445 -> 507 tok/s at 5.6K tokens.
  - QSA top-k: 20K-context decode 40.5 -> 45.2.
- **Waiting for windows:** MTP drafts in nslm-chat, mac/qwen4-chat-drafts e816fe88008e72c184a63a21cd6a55bb5466ca94 on main cb3d6b6 (macos and linux PASS).
  - Q4E_DRAFTS=K in engine/qwen4_api.c, nslm-chat's temp-0 drafts loop, tests/test_qwen4_api_mtp.c.
  - P3 code 47.8 -> 69.6 tok/s at K = 2 in nslm-chat, the same text.
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c.
- **Next:**
  - Fuse the hyper-connection inject rows into the down GEMM (about 1.3% of a prompt).
  - Prompt attention (latency-bound tile walk; a shared K/V buffer was slower).
  - A chunked DeltaNet for prompt rows.
