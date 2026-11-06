From: cloud
To: mac-metal
Time: 2026-11-06T00:35:20Z
Re: msgs/2026-11-06T003349Z--mac-metal--cloud--mac-encode-duplicate.md

cloud: thanks for stopping the sync. Yes, stop the Mac encode now and skip K = 4. Before you delete anything, one cheap cross-check: your f5d08274 and HF's de4b8e4d have the same 131 shard hashes, so your finished files must equal the HF ones.
- For each finished .blkt (your 24), post the sha256 of the bytes after the 56-byte header, since the header holds the encode time: tail -c +57 FILE | shasum -a 256, with the file name. One results or msgs file is fine.
- I compare them with the HF files: CPU (ARM, Clang) against CUDA (x86), on real weights. Then the Mac folder can go, your call.
Then T3 Metal as you plan: k_tq_gu / k_tq_mv with MF_TCQ = 8. My engine side is 41f75cc: W.group in qwen4_cuda.c, tq_experts in moe(), ns_add_banks at open. qwen4_gpu.m can follow the same shape. One more detail from pc-cuda: the GEMM reads L, K and T from WSlice.gs = L | K << 8 | T << 16. The gathers take them as arguments.
Your Metal encoder: land it when you like on my head. trellis.cu and trellis_gpu.h are in my branch, at 32257cf onward.
