From: pc-cuda
To: all
Time: 2026-10-09T05:24:57Z

mac/mla-seeds 59a0b6c is green: macos, windows and linux all PASS. It fast-forwards main ca86b47. I'm asking Matt to
merge it.
Healed model, P=4 seed MLA projections, pipeline run entirely on CUDA (capture, then search in 273 s, then a seeded delta
written with the packer's layout); 31 held-out windows, BF16 KV:
  BF16 0.1534 / Q8 0.1536 / P=4 seeds 0.1620 / Q4 0.1762   (delta 945 / 502 / 254 / 266 MB)
Seeds keep about two-thirds of Q4's loss at Q4's size, the same pattern as your naive numbers (0.2199 / 0.2206 / 0.2296 /
0.2471).
Next for me: nslm4_gpu_search_a in nslm/search4.cu for mac/mla-gptq.
