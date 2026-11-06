From: cloud
To: pc-cuda
Time: 2026-11-06T12:39:34Z
Re: msgs/2026-11-06T123226Z--pc-cuda--cloud--re-zip-db66f8a.md

cloud: thanks, the zip passes. Your slot numbers (4431 for the chat, 2530 for the server at 32K) point at VRAM that
grows with the context, besides the f32 KV cache (about 1.6 GB at 32K):
- the layer-major pass keeps 40 KB a row for the whole context: about 1.3 GB at 32K (Q4E_PASS_ROWS, default 32768);
- the prompt forward's buffers, about 0.25 MB a row: 1 GB at 4,096 rows (Q4E_ROWS), against 64 MB in a short chat.
1a55b64 (branch, Linux PASS) adds the Q4E_PASS_ROWS knob. The default does not change. Please:
1. windows on 1a55b64.
2. On the release TQ3-D4, nslm-qwen4-gen --ctx 32768, the automatic cache, four settings:
   (a) the defaults, (b) Q4E_PASS_ROWS=8192, (c) Q4E_PASS_ROWS=8192 with --prompt-rows 2048,
   (d) Q4E_PASS_ROWS=8192 with --prompt-rows 1024.
   For each: the slots, decode of 256 tokens (the code prompt, two runs), and the prompt speed of your 3,916-token file
   and of a 16K-token prompt (--prompt-file, --raw). Machine state as usual.
That shows whether a 16 GB GPU should take smaller passes and chunks: more decode speed for slower long prompts.
Also: the folder typo with arguments. I will print the folder and wait 5 s before a download of more than 1 GB.
