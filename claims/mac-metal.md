# mac-metal claims (updated 2026-11-06T06:01:44Z)

# mac-metal claims (updated 2026-11-06T06:02:00Z)


- **For review:** mac/tq-head-test a611a5e (on 7e42e52): test_qwen4_metal covers a trellis MTP head (bank L{n_layer}): gate MB with 1 to 3 drafts, and the head's bank in use. macos PASS.
- **Done (05:42Z and 05:55Z, to cloud):** the oMLX remote-code test on oMLX 2165e757.
  - P3 loads and chats only with the one-line model_file change.
  - The MTP garbage was the published repos' stale nanoseedlm_seeds.py (from before cccbd52: per-pair rows of mlx-vlm's verify).
  - With main's module, MTP output equals MTP off, at 85.1 tok/s.
  - The re-upload to the P3 and P4 repos waits for Matt; not mine.
  - The worktree ~/devel/omlx-2165e75 holds the one-line diff only. No oMLX PR.
- **Done today (main or cloud's head):** the Metal trellis gathers in form (a), +3.2% decode on the mix (7e42e52); tools/page_resident.py (main); the M4 Max trellis speed and PLE working-set measurements (cloud's pages); the Metal trellis encoder; trellis experts on Metal; MTP drafts in nslm-chat; macos PASS on 99cf18a, a58d0f5 and d2a249b.
- **Measured for plan L:** on Metal, table codes and LFSR-style linear tables are slower than the computed 1MAD (msgs 04:56Z).
- **Kept on the Mac:** the K = 3 and K = 4 L = 12 banks; the TQ3L12 / TQ4L12 / TQ3TQ4down pack folders; ~/models P3 and P4 as they are (cloud, 05:56Z).
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m, tools/page_resident.py.
- **Rules (Matt):** I push only mac/* branches and post them here; main merges are cloud's. Nothing on HF without a message to cloud first.
