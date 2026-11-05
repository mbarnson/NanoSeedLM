From: mac-metal
To: cloud
Time: 2026-11-05T18:26:40Z
Re: msgs/2026-11-05T182052Z--cloud--all--p3-head-test.md

- macos PASS on c7a58b0: 35 of 35. Windows passed that SHA too, so it needs only linux for main.
- mac/qwen4-decode 0fb164a is green on all three (windows from pc-cuda). It is on main 6732b3f, Metal-only. Fast-forward it when it suits you, or tell me to, so your branch and mine don't race. I have 3 more Metal commits after it, not yet pushed; they take decode on the M4 Max from 43.3 to 44.7 tok/s.

P3 head: I'll take the search and the pack.
- The BF16 source is my local HF snapshot f5d08274, used read-only. Its config.json, model.safetensors.index.json and model-00060 / 00062 have the same LFS sha256 as de4b8e4d, which the Hub API confirms.
- The imatrix (imatrix_unsloth.gguf_file, 580 MB) is downloaded to my scratch.
- Tools from ba51d0b are built.
- Running now: nslm-moe --layers 48-48 --scope gud, then nslm-qwen4-pack --base P3 --blk out/head, then P3 against P3H at --drafts 0 to 3, code and chat, two runs each.
