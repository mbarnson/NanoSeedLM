# mac-metal claims (updated 2026-11-06T06:45:24Z)

# mac-metal claims (updated 2026-11-06T06:47:00Z)


- **For review:** mac/tq-head-test a611a5e (on 7e42e52): test_qwen4_metal covers a trellis MTP head (bank L{n_layer}): gate MB with 1 to 3 drafts, and the head's bank in use. macos PASS.
- **Done (06:44Z, to cloud):** validated mbarnson/omlx fix/qwen4-exp-custom-model-file at 29ee79c on the M4 Max.
  - The regression test fails before the fix and passes after; the two relevant files give 212 passed; the default suite gives 19,296 passed.
  - End to end: P3 at cba0d3e with MTP is byte-identical to MTP off at about 85 tok/s, and stock oQ4e-mtp is unchanged.
  - The worktrees and scratch folders are removed (cloud, 06:36Z). The PR is Matt's call.
- **Done today (main or cloud's head):** the Metal trellis gathers in form (a), +3.2% decode on the mix (7e42e52); tools/page_resident.py (main); the M4 Max trellis speed and PLE working-set measurements (cloud's pages); the Metal trellis encoder; trellis experts on Metal; MTP drafts in nslm-chat; macos PASS on 99cf18a, a58d0f5 and d2a249b; the cause of the oMLX MTP garbage (the repos' stale nanoseedlm_seeds.py, re-uploaded by cloud).
- **Measured for plan L:** on Metal, table codes and LFSR-style linear tables are slower than the computed 1MAD (msgs 04:56Z).
- **Kept on the Mac:** the K = 3 and K = 4 L = 12 banks; the TQ3L12 / TQ4L12 / TQ3TQ4down pack folders; ~/models P3 and P4 as they are (cloud, 05:56Z).
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m, tools/page_resident.py.
- **Rules (Matt):** I push only mac/* branches and post them here; main merges are cloud's. Nothing on HF without a message to cloud first.
