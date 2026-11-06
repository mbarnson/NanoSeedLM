# mac-metal claims (updated 2026-11-06T13:47:34Z)

- **Working on (cloud 10:01Z): R7, trellis experts in the MLX loader**, on mac/mlx-trellis (from main 5cd113a):
  - new tools/nanoseedlm_trellis.py (mine): TrellisSwitchLinear, a gather mat-vec and a sorted tile GEMM from the MF_TCQ kernels;
  - a small sanitize branch in tools/nanoseedlm_qwen4.py (cloud's file; cloud said I may change it);
  - tests first (rows equal nslm_tq_decode_row's; kernels against a dequantized matmul); then gate RF on TQ4L12.
  - Done: 7f4b9e8 (local): nanoseedlm_trellis.py, the loader branch, tools/test_nanoseedlm_trellis.py in make test-mlx (all pass). Next: gate RF on TQ4L12.
  - RD (TQ4 download test) goes first when pc-cuda's upload lands.
- **Working on (cloud 13:45Z): plan K3**, the qwen4 Metal engine's KV cache in BF16 (qwen4_gpu.m, kernels_qwen4.metal, test_qwen4_metal against qwen4_ref_kv_bf16), on a mac/* branch from 20666ef. RD on TQ4 stays first when the Wi-Fi is back.
- **Done:** qwen4_gpu.m takes the engine's resource_dir (nslm-serve --res and its default), on its own branch. It needs a Q4EOpts field (qwen4_eng.h) and one line in qwen4_api.c.
- **Later (R6):** test the macOS release tarball (nslm-chat once, nslm-serve with one request) when the draft is up.
- **Done (06:44Z, to cloud):** validated mbarnson/omlx fix/qwen4-exp-custom-model-file at 29ee79c on the M4 Max.
  - The regression test fails before the fix and passes after; the two relevant files give 212 passed; the default suite gives 19,296 passed.
  - End to end: P3 at cba0d3e with MTP is byte-identical to MTP off at about 85 tok/s, and stock oQ4e-mtp is unchanged.
  - The worktrees and scratch folders are removed (cloud, 06:36Z). The PR is Matt's call.
- **Done today (main or cloud's head):** the Metal trellis gathers in form (a), +3.2% decode on the mix (7e42e52); tools/page_resident.py (main); the M4 Max trellis speed and PLE working-set measurements (cloud's pages); the Metal trellis encoder; trellis experts on Metal; MTP drafts in nslm-chat; macos PASS on 99cf18a, a58d0f5, d2a249b and 8856cc4; the cause of the oMLX MTP garbage (the repos' stale nanoseedlm_seeds.py, re-uploaded by cloud).
- **Measured for plan L:** on Metal, table codes and LFSR-style linear tables are slower than the computed 1MAD (msgs 04:56Z).
- **Kept on the Mac:** the K = 3 and K = 4 L = 12 banks; the TQ3L12 / TQ4L12 / TQ3TQ4down pack folders; ~/models P3 and P4 as they are (cloud, 05:56Z).
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m, tools/page_resident.py.
- **Rules (Matt):** I push only mac/* branches and post them here; main merges are cloud's. Nothing on HF without a message to cloud first.
