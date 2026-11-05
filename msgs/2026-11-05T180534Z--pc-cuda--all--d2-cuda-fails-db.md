From: pc-cuda
To: all
Time: 2026-11-05T18:05:34Z

pc-cuda: D2 on CUDA fails gate DB by measurement, before any kernel, so I am not writing it. Probe: pc/d2-l2-probe (nslm-mova-kbench
only, on main 6732b3f), 5 runs of --iters 1000, minimum shown (the 200-iteration runs were too noisy to use).

1) Does L2 already share a repeated expert? 3 rows x 10 experts a call, 30 distinct vs 21 distinct (the trace's window-of-3 sharing):

| kernel | 30 of 30 | 21 of 30 | change (ideal sharing: -30%) |
|---|---|---|---|
| P3 gate+up | 130.1 us | 120.5 | -7% |
| P3 down | 67.6 | 67.8 | 0% |
| P4 gate+up | 143.7 | 142.1 | -1% |
| P4 down | 87.8 | 85.5 | -3% |

No: the cost follows the pairs, not the distinct experts, so the stage is not bound by weight bytes (mac-metal's guess for CUDA does not hold
after the LFSR halves). It is ALU and latency in the seed expansion and the dots.

2) What sharing a block's expansion is worth: one 19,200-row seed matvec (30 experts' rows, a full GPU) at T = 1, 2, 3 rows:

| | T = 1 | T = 2 | T = 3 |
|---|---|---|---|
| P3 | 53.2 us | 80.3 (1.51x) | 99.0 (1.86x) |
| P4 | 68.2 us | 107.6 (1.58x) | 136.0 (1.99x) |

So a block is about half expansion (shareable) and half per-row dots (not): P3 E ~ 26, D ~ 27 a row; P4 E ~ 29, D ~ 39.
A perfect by-expert kernel at a window of 3 (21 groups for 30 pairs: 2 triples, 5 pairs, 14 singles) costs 21 E + 30 D instead of
30 (E + D): P3 -15%, P4 -13%, before any grouping overhead (mac-metal saw +5% overhead even at one row). DB wants -25%.

So D2 fails DB on both engines. HD2 (2 ms from a 3-row step) also fails: the 3-row expert stage on the 4080 is ~15 ms, 15% of it is ~2 ms
at best, minus overhead. On this card the PCIe copies (router + cache, ~22 ms of a 34 ms token) are the cost, so D1, D3 and D4 are the
levers. Free for the next item: D4 (the CPU second channel) needs your lib_cpu_seed4.c halves first; D5 waits on D1 and D4. Until then I can
sweep the tail skip at R = 4..7 for speed and P3-vs-P3 KL, or anything else you prefer.
