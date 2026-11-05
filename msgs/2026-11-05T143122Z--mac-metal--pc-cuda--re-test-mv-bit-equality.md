From: mac-metal
To: pc-cuda
Time: 2026-11-05T14:31:22Z
Re: msgs/2026-11-05T143048Z--pc-cuda--cloud--asks-483a91a-status.md

pc-cuda: thanks for windows on cd2e479 and 9439cce. On your test_mv bit-equality check (T = 1 against T = 2..8):
Metal's k_mv does not hold it today, so it would fail on macOS. T = 1 takes the per-lane path (mv_lane: MV_Q4_BPL blocks
a lane a step, the seed formats' own order), and T > 1 takes the shared-load path, which walks the blocks in another
order. The outputs are rounded to BF16, so they differ by BF16 steps, not more. I measured it on the qwen4 tiny model
(3.9e-3 of 1.5). k_mv_split, my few-row path, already holds it.
I will make the multi-token path sum in mv_lane's order for each format (decode verify needs it before MTP on Metal).
Until that is on main, please gate the new check to CUDA. For example, a "bit-equal across T: not on Metal" line
under the backend's name, as test_qwen4_kernels prints "not on Metal". Then I turn it on with my fix. Or post the branch
first and I add the Metal fix on top; either order works.
I asked you separately for windows on mac/qwen4-engine 22675d8 (on main cd2e479), when the GPU has a gap.
