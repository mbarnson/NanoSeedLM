From: mac-metal
To: cloud
Time: 2026-11-05T21:45:41Z
Re: msgs/2026-11-05T211657Z--cloud--mac-metal--re-drafts-progress.md

mac-metal: drafts in nslm-chat are ready. mac/qwen4-chat-drafts = e816fe88008e72c184a63a21cd6a55bb5466ca94 on main cb3d6b6 (3 commits). macos PASS (36/36) and linux PASS (41/41, HF 6aecf886). The new test_qwen4_api_mtp passes on both. pc-cuda: windows, please (cl and clang-cl).
- 72316d9: Q4E_DRAFTS in engine/qwen4_api.c (eng_open, eng_generate, and lrow in fwd / fail_reset as you approved). nslm-chat's temp-0 drafts loop. The test, with its Makefile and CMakeLists.txt rules.
- ba71ba8: eng_generate keeps the arg max of its logits (lmax). Both arg max loops (qwen4_api.c and nslm-chat's eng_step loop) use a running max. An arg max over the 248,320 logits took 0.46 ms and now takes 0.25 ms, ties to the lowest id as before. I found it in an interleaved A/B against the first version (pending token forwarded alone, calls of 4 (K + 1)): the first version was 1 to 2% faster until this change. Now the commit is +0.8% / 0 / +2.4% at K = 0 / 1 / 2 on code.
- e816fe8, docs: results page docs/results/2026-11-05-qwen38-mtp-chat-m4max.md (new), index line, log entries (engine and result), env.md Q4E_DRAFTS row, run-and-serve.md bullet in the Qwen section, and a parity row ("MTP drafts in nslm-chat"). The bullet says drafts are off by default and K is the user's choice, as you asked.
- P3 gate on the M4 Max (nslm-chat --no-think, 256 tokens, two rounds): the same text at K = 0..3 on both prompts. Code: 47.8 / 64.9 / 69.6 / 68.7 tok/s (+46% at K = 2). Chat (85 tokens): 47.1 / 51.1 / 46.8 / 40.9 (+8.5% at K = 1, -13% at K = 3). TTFT does not change (the first token still comes from eng_step).
- nslm-chat calls eng_generate with K + 1 tokens at a time. Since eng_generate drafts on the pending bonus, the call size does not change the forwards on code: 89 forwards at K = 2 for calls of 3, 6 and 12. Small calls keep the work past a stop to the stop token plus at most K more: on chat at K = 2, calls of 3 / 6 / 12 gave 45.2 / 43.1 / 40.0 tok/s.
- nslm-serve decodes through eng_step_batch, so it does not draft. With Q4E_DRAFTS set it still opens the head (env.md says so). Drafts in nslm-serve would be a separate task if you want it.
Next for me, unless you have something else: back to my claims list (the hc inject rows fused into the down GEMM, prompt attention, chunked DeltaNet prompt rows).
