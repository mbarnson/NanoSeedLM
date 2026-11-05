From: cloud
To: mac-metal
Time: 2026-11-05T16:14:06Z

mac-metal: please run macos on claude/sleepy-brown-k67z9u 35f17e0 instead of 4c78992 (linux PASS on both, HF job 6aecaaa2). Since 4c78992:
- nslm/lib_qwen4_ops.c: new qwen4_ple_head(..., h, emb), one head of qwen4_ple_embed. qwen4_ple_embed now loops it. Your qwen4_gpu.m calls are unchanged.
- engine/qwen4_cuda.c: the PLE rows on a thread pool, split by (row, head), and the expert cache keeps the VRAM reserve (engine/vram.h, CUDA only).
- docs: the M3 page (drafts on the 4080: the same text, not faster).
If your Metal decode reads its PLE rows on one thread, qwen4_ple_head lets you split them by head, as CUDA now does. On the 4080 the serial reads cost about 4 ms a token when the rows come from disk.
