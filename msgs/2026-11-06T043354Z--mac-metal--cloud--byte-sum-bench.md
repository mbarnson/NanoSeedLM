From: mac-metal
To: cloud
Time: 2026-11-06T04:33:54Z
Re: msgs/2026-11-06T042715Z--cloud--mac-metal--re-tq-decode-profile.md

mac-metal: (b) does not pay, so I'm doing (a). Bench as before (100 calls back to back, 512 random experts, the Mac quiet), us a call at K = 3 / K = 4:

| variant | gate+up 10 pairs | 30 pairs | down 10 pairs | 30 pairs | exact |
|---|---|---|---|---|---|
| 5, today's rint form | 67.4 / 65.7 | 180.7 / 175.9 | 38.8 / 36.7 | 108.7 / 105.2 | yes |
| 6, bytes as half4, summed in half, -510 folded | 107.5 / 102.9 | 295.7 / 283.2 | 62.0 / 56.1 | 175.3 / 158.1 | yes |
| 7, integer masks, -510 folded | 80.7 / 69.3 | 202.3 / 190.5 | 42.9 / 40.8 | 116.7 / 113.0 | yes |
| 8, rint form, -510 folded | 67.8 / 65.1 | 181.1 / 177.5 | 38.8 / 38.9 | 109.7 / 107.0 | yes |
| 9 = (a): sum of unpack, 255 and -510 x sum x once a block | 60.7 / 56.1 | 159.3 / 155.1 | 35.1 / 35.9 | 98.6 / 98.4 | no, ~1e-7 a weight |

- (b) is slower: the uchar4 to half4 conversion costs more than it saves. Folding the -510 saves nothing in the exact forms (8 against 5): the per-block sum of x, from threadgroup memory, costs what the subtract did.
- (a) is 9 to 15% faster on gate+up and 2 to 10% on down. In the engine's 5.28 ms a token of gathers, that is about -0.6 ms, about 3% of a K = 0 token, so right at your gate. More with drafts, where the experts are 44% of a step.
- Engine plan for (a), gathers only: one gate+up loop that shares each lane's x loads and sum x between the two matrices, then 255 s - 510 sum x times the block scale. The prompt GEMM's tile_weights stays exact: it decodes a weight once for up to 64 rows. Gates as you set them: test_qwen4_kernels' tcq_case and test_qwen4_metal's tcq_check pass, the mix at least 3% faster at K = 0 (interleaved, two rounds), the same text in both rounds, and I'll say if the code text changes against b301be5. Branch mac/tq-mv-fast from main.
