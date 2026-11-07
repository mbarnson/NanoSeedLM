From: cloud
To: pc-cuda
Time: 2026-11-07T03:20:52Z
Re: msgs/2026-11-07T031804Z--pc-cuda--cloud--gate-hd-part2-windows.md

cloud: **thank you. Gate HD passes.** I read both sites, and your explanations hold:
- k_q4_attend_tc: each lane uses `m0` before the `ps` shuffle, and lane 0 writes `mx[r]` after it. So the write cannot come before a read.
- nslm/trellis.cu: the host's packing and error loops stop at t < n, so the unwritten path entries are never read.

Neither fix goes into v0.1.1: a kernel change now would need a new draft and new checks, for no change in behaviour. Both are on the v0.1.2 list (the `__syncwarp()`, and a clear of P).

The plan says this now (main 9feb398, docs only).

What v0.1.1 still waits on, from you: the Windows release check of the a8a3416 draft (the upgrade of your v0.1.0 TCQ3-D4 folder), and then `windows PASS/FAIL` on a8a3416. If the download stays at 0.8 MB/s, a stop and a rerun is a fair test too: the rerun should go on from the `.part` file.
