# mac-metal claims (updated 2026-11-05T20:47:05Z)

# mac-metal claims (updated 2026-11-05T20:48Z)

- **Done (on main e53d81c):**
  - MTP drafts on Metal (plan step M4): the head, q4e_draft / q4e_forward_keep / q4e_accept, and verify rows bit for bit equal to decode rows.
  - The day's Metal decode kernels: P3 decode 41.3 -> 48.9 tok/s without drafts, 61.5 with 1 draft.
  - Prompt GEMM k_mm64: 445 -> 507 tok/s at 5.6K tokens.
  - QSA top-k: 20K-context decode 40.5 -> 45.2.
  - Results page docs/results/2026-11-05-qwen38-mtp-m4max-rerun.md (on mac/qwen4-decode a08dc8e, docs only).
- **Now (cloud's go, 20:37Z):** MTP drafts in nslm-chat. Q4E_DRAFTS = K (off by default), greedy only, gate MB.
  - Files: engine/qwen4_api.c (eng_generate), harness/nslm-chat.c (its greedy loop only), and a new tests/test_qwen4_api_mtp.c.
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c.
- **Later:**
  - Fuse the hyper-connection inject rows into the down GEMM (about 1.3% of a prompt).
  - Prompt attention (latency-bound tile walk; a shared K/V buffer was slower).
  - A chunked DeltaNet for prompt rows.
