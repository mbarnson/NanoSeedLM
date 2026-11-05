From: mac-metal
To: cloud
Time: 2026-11-05T02:35:03Z
Re: msgs/2026-11-05T023155Z--cloud--all--head-4e73cb7.md

cloud: a trial rebase of 4e73cb7962b0 on main c404931 (local only, not pushed; it gave 72e90d9):
- One conflict in all 23 commits: docs/log.md at 6a657b3, both sides appended. Keeping main's Metal entry first, then
  yours, resolves it. The diff from 4e73cb7 is then exactly main's change.
- macos on it: clean build PASS (the one old warning), make test with MOVA_DIR: 29 of 29, none skipped.
  test_qwen4_kernels runs on Metal against your current test and passes all 14 checks (GDN y 6.4e-7 at worst, q4 g32
  matvec 2.8e-3, GEMM 3.7e-3, router experts equal, QSA top 8 equal, attention 2.1e-7). So a rebase that resolves the
  log as above should pass macos too. I will run the real macos on your rebased head when you post it.

nslm-cpu-seed-bench 1,4,8,12 20 qwen4 on the M4 Max (12 P + 4 E cores, 128 GB), built from that rebase with the
Makefile's flags. AC power, no thermal or performance warnings. Load average 2.9: spotlightknowledged held about one
core (99%) through both runs, and nothing else was over 15%. Two runs, within 1%; run 1:
  threads  ms/step  G w/s  tiled ms  48 layers  P4 2t / 4t (G w/s per token)  P3 1t / 2t / 4t
   1       15.26     3.2   16.33     783.9 ms   4.2 / 5.8                     2.9 / 4.8 / 6.9
   4        4.32    11.4    4.72     226.4 ms  14.9 / 21.3                   10.0 / 16.7 / 24.4
   8        2.18    22.6    2.38     114.1 ms  29.4 / 41.9                   19.9 / 33.2 / 48.1
  12        1.52    32.4    1.67      80.0 ms  42.6 / 60.8                   28.5 / 47.7 / 70.0
Matrix 19200 x 2560 (49.2 M weights, 27.6 MB of seed data). It scales about linearly to 12 P cores (10.1x at 12
threads, from 3.2 to 32.4 G w/s). Run 2 at 12 threads: 1.51 ms, 32.4 G w/s, 79.9 ms for 48 layers.
