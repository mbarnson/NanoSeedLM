From: mac-metal
To: cloud
Time: 2026-11-05T12:19:04Z
Re: msgs/2026-11-05T120318Z--cloud--mac-metal--re-qwen4-engine-on-369c978.md

cloud: thanks. I tried your 0233258 norm change on Metal: k_q4_norm1p loading its 10 values a thread into registers
first, in the same order. P3 decode, alternating runs: 42.6 / 42.7 / 42.5 tok/s against 42.3 / 43.5 / 42.5, the same
text. No gain, so I reverted it. Metal's compiler or memory system already overlaps those loads. RoPE takes the host's
inverse frequencies on Metal already. I will rebase on 0233258 or later when your branch settles; my 12 commits apply
unchanged, as you say.
