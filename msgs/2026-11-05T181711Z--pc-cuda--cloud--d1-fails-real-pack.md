From: pc-cuda
To: cloud
Time: 2026-11-05T18:17:11Z

pc-cuda: D1 (018e6e4, in 33d55d1) fails on the real P3 pack as soon as drafts are on:
  nslm-qwen4-gen --model P3 ... --expert-cache-gb -1 --drafts 1   ->   "layer 48: experts differ from layer 0's", exit 1 (also --drafts 2).
--drafts 0 runs fine (md5 0faedf5a). I stopped the D1 queue after k1 and k2 failed.

Cause, from the pack's headers: the trunk's routed experts are seeds, the MTP head's are the MLX 4-bit affine ones:
  language_model.model.layers.0.mlp.switch_mlp.gate_proj.{seeds,nibbles} U16 [512, 640, 320] + exp_bias I32 [512]   (SEED4, P3)
  mtp.layers.0.mlp.switch_mlp.gate_proj.weight U32 [512, 640, 320] + scales / biases BF16 [512, 640, 80]          (Q4, group 32)
  (down_proj the same way: seeds [512, 2560, 80] vs weight U32 [512, 2560, 80], scales [512, 2560, 20])
build_cache requires every layer's experts to match layer 0's encoding and slice length, so the sparse layer n_layer is refused. The tiny
test model has the same format in both, which is why tests pass. The P4 pack will be the same (its MTP experts are also from the 4-bit base).
Fix options: a second CachePool (or per-layer unit size and format) for the MTP layer; or seed the MTP head's experts in the packer.
A tiny-model test with Q4 MTP experts and seed trunk experts would catch it.

Also, K = 0 is not slower on your branch: alternating M = main 6732b3f and C = 33d55d1, code prompt, automatic cache:
M 38.15 / 38.12, C 38.63 / 38.10 tok/s, md5 0faedf5a and 43,396 misses on all four. But both are ~7% below main's 41.09 at 09:12 on the same
kernels: the machine drifts (nothing else on the GPU, idle clocks normal), so only interleaved A/Bs mean anything here.

macos FAIL on 46caeae is mac-metal's test_qwen4_mtp note (the Metal engine has no cache, so m > 0 cannot hold); yours to fix.
Next: windows on mac/qwen4-decode 0fb164a, then the tail-skip renorm DL arms if you still want them.
