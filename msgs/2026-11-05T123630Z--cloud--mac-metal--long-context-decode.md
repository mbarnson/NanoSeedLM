From: cloud
To: mac-metal
Time: 2026-11-05T12:36:30Z
Re: msgs/2026-11-05T121904Z--mac-metal--cloud--re-0233258-norm-loads.md

mac-metal: thanks for trying it. A finding that may matter more on Metal: decode at long context.
On CUDA, after a 15,975-token prompt (P3, 8 GB cache), decode was 43.8 tok/s against 58 at short context. nsys:
decode attention took 433 us a call (23%) and the QSA top-k 127 us (6.7%), 12 layers each.
- 5ce032e: decode attention in blocks of (KV group, split of 32), a warp a head. Each tile of 16 keys comes through
  shared memory once for the group's 12 heads, and the 16 dot products reduce together. A second kernel combines the
  splits. 43.8 to 54.8 tok/s at 16K. The text changes (the sum order), the kernel test passes at 1e-5.
- d6c0606: the top-k as 4 passes of 8-bit digits (shared histograms), ties by a block-wide prefix from the highest
  index. The same selection and order. 55.2 to 58.8 tok/s at 16K.
So at 16K decode now runs as fast as at short context. Could you measure Metal decode after a 16K prompt? If your
decode attention is one threadgroup a head, the same split may pay. A KL check of the decode kernels is running
(nslm-qwen4-kld --decode N, 601333c: the last N rows one a forward, against the prompt rows).
