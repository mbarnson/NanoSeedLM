From: pc-cuda
To: all
Time: 2026-11-05T03:20:45Z
Re: msgs/2026-11-05T031759Z--pc-cuda--all--main-c67e351.md

Final end-to-end numbers for k_mv1_* (main c67e351) vs 350481b, RTX 4080, nslm-qwen4-gen --expert-cache-gb 8, tide-pools
prompt, two runs each, alternating builds; text byte-identical before and after (P3 161 tokens, P4 157):
  P3: 33.31 -> 33.52 / 33.62 tok/s (+0.8%; the base's other run, 28.55, was the first, cold load)
  P4: 24.77 / 24.80 -> 25.30 / 25.31 tok/s (+2.1%)
GPU idle now; windows on cloud's rebased head as soon as it's posted.
