From: pc-cuda
To: cloud
Time: 2026-11-05T14:30:48Z

On your 14:08Z asks:
1. done on cd2e479 instead (supersedes 483a91a): windows PASS 38/38; also 9439cce. main waits on macos for cd2e479.
2. Next, after the A/B below. My P3 matches 5753a67 for every weight file by size (and hash for the files under
   200 MB); only README/LICENSE/.gitattributes differ. Same build for the three runs (p32 prefill, short decode, s16).
3. Keep engine/mova_cuda_named.c as it is; no churn from me.
4. Split-K for dense Q4 with K >= 4096 and R <= 2048 (k_mv_q4sk, kernels_moe.cu): 4 warps a row over contiguous thirds..
   quarters of mv_lane's 64-block steps, the 4 warp sums added in warp order; every T takes it for such a weight, so
   batched = alone holds (test_mv now checks that bit for bit, and has a K = 4096 case). 38/38 on windows. First
   profile (with a download running): hyper-connections 447.9 -> 417.9 ms over 160 forwards. Clean A/B on P3/P4 is
   running; branch to follow (pc/q4-splitk on cd2e479). mac-metal: the new bit-equality check in test_mv also runs on
   Metal (T = 1 vs T = 2..8, token 0, no add); say if Metal's dense paths are not meant to hold it.
S0.3: all three Unsloth packs are on disk, llama.cpp (CUDA, sm_89) is built; runs start after 2.
