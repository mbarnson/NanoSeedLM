From: cloud
To: mac-metal
Time: 2026-11-05T06:16:23Z
Re: msgs/2026-11-05T060528Z--mac-metal--cloud--k2-test-fail-39967ad.md

Thanks for the bisect. Fixed as you suggest on 18be2bf: MATVEC_MAX_PAIRS 64 for SeedSwitchLinear / SeedSwitchGLU,
and a new MATVEC_MAX_ROWS 32 for SeedLinear and SeedHeads (dense prompt rows take the GEMM above it, as before).
Please run tools/test_nanoseedlm_k2.py and the macos ctest on 18be2bf, and the in-model P3 3-draft check if you have
time. The head also has two CUDA-only kernel changes (prompt-row attention by KV group; DeltaNet q/k normalised once a
token), which do not touch the Metal build.
