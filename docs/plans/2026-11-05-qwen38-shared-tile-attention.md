# Plan: prompt attention that loads each key tile once for several rows (2026-11-05)

**Goal.** Make Qwen3.8-Flash-Next prompt attention on CUDA 1.5 times faster, with the same KL against the reference.
At 15,922 tokens it takes 480 ms of 3.73 s (13%). Each block takes one row and one KV group, and loads its own
selected keys. Neighbouring rows select most of the same blocks, so most of those loads repeat.

**Status.** Stopped at gate G2: the kernel was 1.7 times slower, and `c036709` reverts it
([shared key tiles](../results/2026-11-05-qwen38-shared-tile-attention.md)). Gates G1 and G3 passed.

## Key numbers

The union of the QSA selections of G consecutive prompt rows, in units of one row's k = 512 blocks. P3, 15,922 wikitext
tokens, all 12 attention layers (job `6aec7e3c`, `scratchpad/rowsjob/seldump.sh`):

| Rows a group | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Union / k, lowest layer | 1.18 | 1.41 | 1.66 | 1.94 | 2.26 | 2.61 |
| Union / k, highest layer | 1.23 | 1.54 | 1.91 | 2.34 | 2.80 | 3.25 |

For 4 rows, a block loads about 1.5 k blocks in place of 4 k: 2.7 times fewer K and V loads. The MMA rows become 4 rows
of 12 heads (48, no padding) in place of 12 heads padded to 16. So the scores grow by 1.5 x 12 / 16 = 1.13 times.

## Steps

1. A C reference of the same order: none is needed. `qwen4_attend` stays the reference, and the kernel test compares at
   1e-5 for sparse and dense rows, as now.
2. `k_q4_attend_tc4`: a block takes a KV group and 4 rows. It builds the union of their selected blocks and tails in
   block order, in shared memory. Each tile of 16 keys comes in once. Each row masks the keys that it did not select.
3. Rows with fewer than 4 rows left, and decode rows, keep their kernels.
4. Kernel test: add 4-row groups with mixed sparse and dense rows, and a group at the start of the prompt.
5. Measure the 15.9K and 29K prompts (A/B on one node), and the KL of 4,096 tokens against the 4-bit MLX model.

## Gates

- G1: the kernel test passes at 1e-5 against `qwen4_attend`.
- G2: prompt attention at 15.9K takes at most 320 ms (from 480).
- G3: the KL against the 4-bit MLX model on 4,096 tokens stays within 2% of 0.161.

## Hypotheses

- H1: the kernel is bound by its K and V loads (about 3 TB/s from L2 now), not by its MMAs. The pass-0 TF32 change
  removed 22% of the MMAs and gained 6%, which supports this.
- H2: 12 warps a block (3 MMA row tiles x 4 dimension slices) keep enough blocks on each SM.

## Costs

- About a day of kernel work, and GPU jobs of about 20 minutes each.
- The key order of each row changes, so greedy text can change at near ties.
- Metal has its own attention kernel. mac-metal decides whether it needs the same change.
