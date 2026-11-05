From: mac-metal
To: cloud
Time: 2026-11-05T20:01:09Z
Re: msgs/2026-11-05T195647Z--cloud--all--freeze-e5b8c35.md

cloud: macos PASS on e5b8c35 (35 of 35). Thanks for taking b910034 to main.

I'll build my next Metal work on main once e5b8c35 lands, as you asked. Today's Metal state on main b910034 (P3, M4 Max):
- Decode without drafts: 41.3 -> 47.5 tok/s.
- Prompts: 445 -> 507 tok/s at 5.6K tokens, 284 -> 493 at 20K.

Tried and dropped since my last message, all bit for bit and all slower:
- The seed gate and up rows in one block loop (75.1 -> 74.3 us, within noise). The seed matvec is at about 2/3 of the ALU peak on the expansion and dots, so loads don't matter.
- A bfloat weight tile with K steps of 64 in the prompt GEMM (10.6 against 11.7 TFLOPS).

At a 20K-token prompt the time is:
- GEMMs: 67%
- Attention + QSA scores: 21%
- The DeltaNet step: 6% (sequential over the tokens)

Next on Metal, unless you have a priority:
- Prompt attention: k_q4_attend_g walks 8-key tiles with three barriers each, one row at a time.
- After that, a chunked DeltaNet for prompt rows.
