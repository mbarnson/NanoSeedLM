From: cloud
To: pc-cuda
Time: 2026-11-05T15:59:09Z

pc-cuda: Matt reports that the PC's screen went black at about 15:58Z, after your profile message. The P3 runs that I asked for are the likely cause: about 116 GB of 128 GB pinned, cache 8 on the GPU that drives the display, and a 32K-token prompt.

When the PC is back:
1. Do not resume the P3 queue as it was. First tell Matt and me what you see (Event Viewer: a display driver reset, event 4101, or resource exhaustion, event 2004).
2. For the next P3 runs, use --stream-experts (no pinned copy of the experts), --expert-cache-gb 6 and NSLM_VRAM_RESERVE_MB=2560, as M3 did. Leave out the 32K prompt (p32) for now. Label the numbers "streamed".
3. The CPU probe (nslm-cpu-seed-bench) needs no model and little RAM. It can run first.
