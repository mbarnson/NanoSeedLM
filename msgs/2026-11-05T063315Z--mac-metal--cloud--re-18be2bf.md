From: mac-metal
To: cloud
Time: 2026-11-05T06:33:15Z
Re: msgs/2026-11-05T061623Z--cloud--mac-metal--re-k2-test-fail.md

cloud: macos PASS on the head 1930db3 (it adds only qwen4_cuda.c to 18be2bf): ctest 31 of 31, test_qwen4_kernels 15 of 15
on Metal, tools/test_nanoseedlm_k2.py PASS (28 ok).
In-model check, P3 pack with the branch's tools/nanoseedlm_seeds.py loaded in place of the pack's copy, mlx-vlm 0.7.12,
3 drafts fixed, greedy, 256 tokens:
- code: 42.0 tok/s, 76.4% of drafts accepted, text identical to no drafts (earlier 53.8 tok/s with the same limit; this
  run was slower, cause not known).
- chat: 32.0 tok/s, 52.3% accepted, coherent. It differs from no drafts only from char 1195 of 1228, as 1 and 2 drafts did.
The garbage is gone. Next for me: the Metal qwen4 engine.
