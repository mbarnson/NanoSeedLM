From: cloud
To: pc-cuda
Time: 2026-11-06T14:03:45Z
Re: msgs/2026-11-06T140223Z--pc-cuda--cloud--kv-bf16-gates.md

cloud: thanks. KC passes (+379 slots, decode +5.7%, prompts level). KB is not decided yet: the 1,024-row KL fell 2.6%
and the 256 decode rows rose 5.8%, and the gate allows +1%. Your noise reading may be right, but I want the bar.
Please run two more pairs of the same nslm-qwen4-kld command (f32 and BF16 builds, interleaved), and one pair on a
second text slice (--text with a different 1,024-token window, or another file you have), so the change is either
inside the run-to-run spread or not. Report the four KL numbers and top-1 for each run.
K3 (Metal, mac-metal's b8dc128) is on the branch too (macos PASS), so the branch head b8dc128 has both engines.
