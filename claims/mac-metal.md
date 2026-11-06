# mac-metal claims (updated 2026-11-06T04:56:25Z)


- **For review:** mac/tq-head-test a611a5e (on 7e42e52): test_qwen4_metal covers a trellis MTP head (bank L{n_layer}): gate MB with 1 to 3 drafts, and the head's bank in use. macos PASS.
- **Done today (main or cloud's head):** the Metal trellis gathers in form (a), +3.2% decode on the mix (7e42e52); tools/page_resident.py (main); the M4 Max trellis speed and PLE working-set measurements (cloud's pages); the Metal trellis encoder; trellis experts on Metal; MTP drafts in nslm-chat.
- **Measured for plan L:** on Metal, table codes and LFSR-style linear tables are slower than the computed 1MAD (msgs 04:56Z).
- **Kept on the Mac:** the K = 3 and K = 4 L = 12 banks and the TQ3L12 / TQ4L12 / TQ3TQ4down pack folders.
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m, tools/page_resident.py.
- **Rules (Matt):** I push only mac/* branches and post them here; main merges are cloud's. Nothing on HF without a message to cloud first.
