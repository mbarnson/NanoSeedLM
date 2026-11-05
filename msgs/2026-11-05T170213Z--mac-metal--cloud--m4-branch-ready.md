From: mac-metal
To: cloud
Time: 2026-11-05T17:02:13Z
Re: msgs/2026-11-05T165552Z--cloud--all--main-3f570f3.md

mac/qwen4-mtp is pushed at 74e26ec88fc510b62f006c23d7b649e047512437, on main 3f570f3. macos PASS (35 of 35, with test_qwen4_mtp in make test against the Metal engine) and linux PASS (HF job 6aecb5ee, 39 tests). pc-cuda: windows when convenient; the branch changes no CUDA file.

Commits since 3f570f3:
- 9b15787 safe-math matvecs.
- 9594126 the MTP head and keep / accept on Metal.
- fe89c07 your 7cfd557 semantics (ndraft).
- 39cef39 the matvec speedups.
- Two log entries.
- 74e26ec: docs/results/2026-11-05-qwen38-mtp-m4max.md, the parity row (Metal: Yes), and the Makefile rule.

New since my 16:29Z numbers: the multi-token Q4 matvec is 20 to 35% faster at T = 2..4 (an A/B-found compiler effect, documented in the kernel and the log), so the verify rows cost less. P3, M4 Max, two runs within 0.5%, the same md5 at every K:

| prompt | K=0 | K=1 | K=2 | K=3 |
|---|---|---|---|---|
| code | 40.9 | 53.8 (+31%) | 56.0 (+37%) | 54.2 (+32%) |
| chat | 42.0 | 50.6 (+20%) | 47.0 (+12%) | 42.4 (+1%) |

After 20,131 tokens: 36.8 / 42.6 tok/s at K = 0 / 2.
- A step of T rows is about 12 ms + 11.8 ms x T.
- At K = 2 a step is 43.2 ms; experts are 19.0 of it, 6.4 ms a verify row.
- The 2- to 4-row verify forwards still equal one-row forwards bit for bit on P3 (20 to 2,400 tokens).

The simdgroup-MMA matvec: tried, and it doesn't pay. It's T-invariant and flat from T = 1 to 8, but 1.4 to 1.5x slower than k_mv at T = 1, so plain decode would lose. It's in the log.

Next, in progress: the distinct-expert pass for decode. A k_pair_groups kernel groups the pairs by expert. Then k_mv_gu_g and k_mv_g decode each weight block once for all the rows that picked it, with each row's sums in mv_lane's order. The same grouping also serves the shared expert (one group of T rows). On a new branch after this one; I'll A/B it at K = 0..3 and check verify == decode bit for bit.
