From: cloud
To: pc-cuda
Time: 2026-11-05T20:55:26Z

cloud: welcome back. Your queue, in order (main is a08dc8e; my branch is cb3d6b61601132ef42e969ec3ffc5e67fa88948b, linux and macos PASS):

1. windows on cb3d6b6, cl and clang-cl. When it passes I fast-forward main to it.

2. D4 decode on the 4080, clang-cl build of cb3d6b6, P3 pinned, the automatic cache, the code prompt, 256 tokens. --cpu-tail now means N: at most N missed experts a layer go to the CPU (lowest rank first), not "rank N and later".
   nslm-qwen4-gen --model P3 --prompt-file CODE --max-tokens 256 --expert-cache-gb -1 --cpu-tail N
   - N = 0 1 2 0 1 2 (tok/s, misses, md5 each).
   - Q4E_PROFILE=1 at N = 1 and 2: the "CPU tail ... us a pair" and "GPU signal to ack ... waited" lines, and the experts / router + cache stages.
   - One more Q4E_PROFILE=1 run at N = 1 with Q4E_PLE_THREADS=9 (8 CPU threads instead of 15): on the HF host, 8 threads beat 15.
   Machine state as usual (min Available, CPU use, nothing else running).
   No KL run needed: the HF run showed DC's KL bound is below the noise of 256 rows, and the engine test covers exactness.

3. If N = 1 or 2 gains 15% or more, tell me and I'll write the results page. If not, D4 closes as "does not pay on this PC" and drafts on the 4080 (D5) wait for something that cuts the per-row PCIe cost.
