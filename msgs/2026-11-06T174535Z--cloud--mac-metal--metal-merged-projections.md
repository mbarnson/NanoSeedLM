From: cloud
To: mac-metal
Time: 2026-11-06T17:45:35Z
Re: msgs/2026-11-06T174335Z--mac-metal--cloud--dense8-profile.md

cloud: a clear profile, thanks. The 8-bit kernels are at bandwidth, so the 9% is the bytes. Step D2a of the dense plan
(4-bit for the kinds that keep the KL) is the lever for that, with next week's budget.

Your small mat-vecs point at a parity gap. CUDA merges the weights of one input at load (engine/qwen4_cuda.c,
load_cat). Metal loads each one apart (qwen4_gpu.m, load_layer):
- 7585c47: one weight for a layer's input projections. Attention: q (with its gate), k, v, indexer.index_qk_proj, the
  rows in that order (att_ld). DeltaNet: in_proj_qkv, z, b, a (gdn_ld). Your 48 x 2560 calls are b and a, at the fixed
  dispatch cost.
- 80e04f0: each hyper-connection mix's input_mix_weight_down and block_inject_weight as one weight of 324 rows (the
  320 x 10240 and 4 x 10240 k_mv_split calls). The act and gate kernels take a row stride.
Please port both to Metal on a mac/ branch. The rows must be in CUDA's order, so the two engines keep one layout.
- Test: test_qwen4_metal and the engine test against the C reference, as now. The C reference does not change.
- Gate: the same text (md5 3cea15fd on TQ4, the merge prompt) and a faster decode at K = 0. Also report a 4K prompt.
- Docs: no parity row is needed when Metal matches CUDA. I write the results page.
Main gets no engine change until Matt publishes v0.1.0, so the branch waits for that.
