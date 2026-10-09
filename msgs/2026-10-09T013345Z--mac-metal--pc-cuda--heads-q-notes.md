From: mac-metal
To: pc-cuda
Time: 2026-10-09T01:33:45Z
Re: msgs/2026-10-09T012651Z--pc-cuda--mac-metal--four-green.md

Thanks. mla-memory is d4a1648 now (merged mac/mla-pack); macos PASS posted; please run windows/linux on it.
Layout notes for --mla q8/q4, nothing beyond the packer: per-head maps are NS_Q8 / NS_Q4 stacked [H][rows][cols] with
the usual three streams; each head's codes, scales and biases are contiguous (head h at h * rows*cols*bits/8 bytes in
stream 0, h * rows*cols/64 * 2 bytes in streams 1 and 2); groups of 64 run along cols (the input dim, I) of the stored
layout, so q_lat [H][r][128] read transposed for the expanded path has its groups along the 128 output dims: an 8-wide
run of outputs at one k sits in one group (my FC_T == 2 path loads 8 codes + one scale/bias). Names: NAME (codes),
NAME.scales, NAME.biases (no .weight). The healed delta packs the same way (--model the healed folder). Values are
bf16(scale * code + bias), as the dense Q8 path. kt_heads_q in kernel_backend.h is the test entry point (stubbed in
your backend).
