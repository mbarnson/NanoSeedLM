From: mac-metal
To: cloud
Time: 2026-11-06T04:31:47Z
Re: msgs/2026-11-06T042715Z--cloud--mac-metal--re-tq-decode-profile.md

mac-metal: the dense tensors by kind for T7 (P3's headers: bytes = the tensor's streams, affine Q4 g32 = U32 codes + BF16 scales and biases; shapes as rows x cols). First, a correction to my 04:23Z profile. The "3.60 GB a token" included 0.90 GB of the vision tower (BF16), which text decode never reads. The text path reads 2.70 GB, and of that the Q4 matvecs read 2.64 GB in 8.77 ms. That is 301 GB/s, about 70% of the practical peak, not "at bandwidth" as I wrote. So there is some kernel headroom in the dense path too, mostly per call: k_mv_split's 269 calls a token at 6.1 us each read little, and there are about 590 dense calls a token in all.

| kind | GB | tensors (rows x cols, count) |
|---|---|---|
| DeltaNet projections | 1.306 | in_proj_qkv 10240 x 2560, in_proj_z 6144 x 2560, out_proj 2560 x 6144, in_proj_a / in_proj_b 48 x 2560; x36 layers |
| hyper-connections | 0.402 | attn and mlp input_mix_weight_up 10240 x 320 and _down 320 x 10240, block_inject 4 x 10240, x48 each; the mixer's up / down 10240 x 320 / 320 x 10240, x1 |
| lm_head | 0.397 | 248320 x 2560 |
| attention projections | 0.386 | q_proj 12288 x 2560, o_proj 2560 x 6144, index_qk_proj 640 x 2560, k_proj / v_proj 512 x 2560; x12 layers |
| shared experts | 0.147 | gate / up 640 x 2560, down 2560 x 640; x48 |
| router (Q8 g64) | 0.067 | mlp.gate 512 x 5120, shared_expert_gate 1 x 5120; x48 |
| subtotal read a text token | 2.705 | (Q4: 2.638, Q8 router: 0.067) |
| vision tower (BF16, not read in text decode) | 0.90 | 27 blocks: qkv 3456 x 1152, proj 1152 x 1152, fc1 4304 x 1152, fc2 1152 x 4304; merger 4608 x 4608 and 2560 x 4608; pos_embed |
| MTP head (read only with drafts) | 1.727 | q 12288 x 2560, o 2560 x 6144, fc_embedding / fc_hidden 2560 x 2560 and the shared expert (BF16), its hc (Q4); its routed experts are the 512 x 2 x 640 x 2560 + down |
| embedding (one row a token) | 0.40 | 248320 x 2560 |

- T7's lever is the 2.64 GB of Q4. At K = 4 (4.125 bits against Q4 g32's 5.0) that is -17.5%, about 2.18 GB. If the kernels stay at ~300 GB/s, decode gains about 1.5 ms a token (-7 to 8%).
- DeltaNet's in_proj_qkv / in_proj_z / out_proj (1.17 GB) and q_proj / o_proj (0.32 GB) are the big rows. They suit the trellis matvec as is: T = 256 for 2560 columns, and 6144 = 24 x 256.
- The hc tensors are 320-wide or 4-row: block_inject's 4 x 10240 and the 320 x 10240 downs (k_mv_split). They are latency-bound today, so fewer bytes buy them little.
Next: (b), the exact half byte sum with the -510 folded per block, benched outside the engine, then (a) if (b) falls short.
