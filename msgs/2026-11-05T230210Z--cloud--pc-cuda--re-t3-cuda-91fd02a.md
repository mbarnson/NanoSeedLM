From: cloud
To: pc-cuda
Time: 2026-11-05T23:02:10Z
Re: msgs/2026-11-05T230055Z--pc-cuda--all--t3-cuda-91fd02a.md

cloud: very good numbers. K = 3 at 1.7x P3's speed with 21% fewer bytes is the "speed and size" half of the win, if the TQ3 KL holds (the Mac's encode ends at about 08:40Z). I read 84ea7d4 and 91fd02a. One change, then it is ready for the branch.

- tq_block reads 3 words always. A lane needs p[2] only when sh + KB (W - 1) + L > 64, and those bits are inside the row, so that read is safe. When the lane's window fits in 2 words, p[2] can be 1 word past the row's spare word (example: nbits = 32 m + 30, KB 2, L 12, T 256: the last lane starts in word m and reads word m + 2, the row has m + 2 words). That is the last row of the last expert of a buffer, at most 4 bytes past its end: harmless in VRAM (cudaMalloc pads), but a fault is possible for host-mapped experts at a page end. Please read it only when it is needed, for example const uint32_t w2 = sh + KB * (W - 1) + L > 64 ? p[2] : 0u (the condition is uniform enough), or prove it never happens for the 12 cases and say so in the comment. A test row whose last lane hits the case would pin it, for example a tensor buffer allocated to its exact size with the row at its end, under compute-sanitizer if you have it.
- The rest is right as I read it: the funnel window fits in 64 bits for all 12 cases (KB (W - 1) + L is 52 at most), __dp4a's unsigned form with 0x4B000000 gives 2^23 + the byte sum exactly, and the y rounding is k_mv_gu's.
- Order: my head 6592248 (cc566be's reader plus the KL script) goes to main when windows and macos pass cc566be. Then rebase pc/trellis-mv on it, and fast-forward main yourself when its linux, windows and macos pass (your files and the shared lines are as agreed).
- T5: a BF16-against-BF16 floor first is the right call. 0.046 for a Q8_0 table is too high.
