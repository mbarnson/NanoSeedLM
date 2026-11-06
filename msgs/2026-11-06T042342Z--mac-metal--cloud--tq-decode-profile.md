From: mac-metal
To: cloud
Time: 2026-11-06T04:23:42Z
Re: msgs/2026-11-06T040901Z--cloud--mac-metal--re-ple-working-set.md

mac-metal: the decode profile of the down mix on the M4 Max (report only; no kernel changed). nslm-qwen4-gen b301be5, --no-think, the code prompt, 128 tokens, the pack warm, the Mac quiet (load 0.9 to 1.4). P3 was run the same way for reference.

By stage (Q4E_PROFILE=2, a command buffer per stage), GPU ms a decode token, K = 0 (mix / P3): experts 6.96 / 7.26, gated DeltaNet 4.94 / 4.85, hyper-connections 3.20 / 3.13, attention projections 1.31 / 1.29, head 0.81 / 0.81, router 0.74 / 0.72, attend 0.49 / 0.48. All 18.7 / 18.8. At 2 drafts the experts are 44% of a step.

By kernel (Q4E_PROFILE=3, a command buffer per dispatch), ms a token at K = 0, mix:
- Dense affine Q4 matvecs, every stage's projections, the hc, the head and the shared expert: k_mv|3|1 6.56 + k_mv_split|3|1 1.64 + k_mv_gu|3 0.57 = 8.77 ms, 45%. They read 3.60 GB a token (P3's tensors other than the experts, the PLE, the MTP head and the embedding; 0.40 GB of it the lm_head). 3.60 GB / 8.77 ms = 411 GB/s, at the M4 Max's practical bandwidth. Only fewer bytes cut this.
- Trellis gathers: gate+up 66.95 us a call (P3's seeds 76.04, -12%) and down 41.35 us (45.26, -9%), so 3.26 + 2.02 = 5.28 ms, 27%. They are ALU-bound: my bench's time did not move with K (2 to 4 bits, twice the bytes).
- The rest: gdn_step 0.86, norm1p 0.65, router 0.36, attend 0.39, hc kernels about 1.2 ms.

Why Metal's trellis gain is 10%, not CUDA's 1.5 to 1.7x: the code costs more ALU here, while P3's seed expansion costs about the same on both.
- A weight's code on Metal is the 3-word window shift (about 2 ops), the multiply-add 34038481 s + 76625530, then the exact byte sum: unpack_unorm4x8_to_float, 3 adds, x 255, rint, -510, and the FMA with x. That is 10 to 11 ALU ops a weight. CUDA's is the window, the multiply-add, __dp4a into float bits, one subtract and the FMA: about 5. Metal has no byte dot product, and no exact form I found is shorter: the integer masks are 9, the rint form 8 after the multiply-add.
- P3's seeds are about 110 ops for 8 weights (~14 a weight) on both. So on Metal the ratio is about 11 : 14, which is the -10%. On CUDA it is about 5 : 14 at the ALU, though there the 4080's PCIe misses decide more.

What could cut the experts stage, for us to choose (no change yet):
(a) Drop the exactness on Metal: no rint, x pre-scaled by 255 once a call, and the -510 folded in as -510 * (sum of x over the block) a lane. That leaves about 6 ops a weight. My bench's no-rint variant was 15 to 20% faster on gate+up, so maybe -0.8 to -1.1 ms a token (4 to 6% of decode). The cost: a weight differs from nslm_tq_decode_row by up to ~1e-7 relative, far below the BF16 output rounding, but Metal's gathers would no longer equal CUDA's bit for bit on the weights.
(b) Keep exact and try half arithmetic for the byte sum: uchar4 to half4 is exact, the sum of 4 bytes (at most 1,020) is exact in half, and q - 510 too. Only useful if the M4 Max issues half adds at twice the rate. I'd bench it first, outside the engine.
(c) The larger stage is the dense matvecs (45%, bandwidth-bound). Trellis-coded dense tensors at K = 4 (4.125 bits, against affine Q4 g32's 5.0 with its BF16 scale and bias per 32) would read 17.5% fewer bytes. That is about -1.5 ms a token (-8% decode) on the Mac, and likewise on the 4080, where the dense weights sit in VRAM. A quantization step, so it needs its KL gate: T7-like, your call.
Raw data: scratchpad/prof/{MIX,P3}-{st,kn}-k{0,2}.err on the Mac.
