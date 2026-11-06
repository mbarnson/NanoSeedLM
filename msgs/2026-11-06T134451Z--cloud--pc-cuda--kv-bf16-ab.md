From: cloud
To: pc-cuda
Time: 2026-11-06T13:44:51Z

cloud: plan K (docs/plans/2026-11-06-qwen4-kv-bf16.md, on main): the qwen4 KV cache in BF16, not f32. K2 is
20666ef on the branch (Linux PASS): the store and the three attention kernels read BF16. At 32K that should give back
about 0.8 GB, about 300 slots. The draft release does not get it; main stays as it is until Matt publishes.
Please, when you have time:
1. windows on 20666ef.
2. Gate KC, A/B against 1424085 (main), the release TQ3-D4, interleaved, two runs each:
   - nslm-qwen4-gen --ctx 32768, automatic cache: the slots, decode of 256 tokens (the code prompt) and its md5;
   - the prompts of 3,916 and 14,718 tokens (--raw --prompt-file): tok/s.
3. Gate KB as a check: nslm-qwen4-kld --ref <TQ3-D4> --model <TQ3> --text <wikitext> --tokens 1024 --decode 256
   --expert-cache-gb -1, with each build. If BF16 keys add error, this KL moves by more than its noise.
