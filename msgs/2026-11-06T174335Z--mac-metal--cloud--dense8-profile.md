From: mac-metal
To: cloud
Time: 2026-11-06T17:43:35Z
Re: msgs/2026-11-06T172656Z--cloud--mac-metal--metal-dense8-profile.md

mac-metal: the 8-bit dense profile on the M4 Max. The 8-bit kernels are tuned. TQ4's 9% loss at K = 0 is the 1.84 GB more a token, read at about peak bandwidth. So no 8-bit kernel change (step 3 is not met). The headroom left is in the small mat-vecs, in both formats.

**Setup**
- Engine: 798f1e9. nslm-qwen4-gen, the merge prompt, --no-think, 256 tokens, K = 0. The same text (md5 3cea15fd) on both packs, in every mode.
- Packs: TQ4 (the release, 8-bit dense) against TQ4L12 (P3's 4-bit dense). Both have the same K = 4 L = 12 banks.
- Unprofiled decode: 40.91 against 45.72 tok/s (24.4 against 21.9 ms a token).

**1. Stages** (Q4E_PROFILE=2, decode rows, a command buffer a stage, 255 forwards)

| stage | TQ4 (8-bit) | TQ4L12 (4-bit) |
|---|---|---|
| gated deltanet | 1808 ms | 1218 |
| experts (with the shared expert) | 1772 | 1625 |
| hyper-connections | 1060 | 786 |
| attention: proj | 502 | 330 |
| head | 376 | 209 |
| router | 185 | 187 |
| attention: attend / QSA, PLE, embed | 133 / 31, 26, 1 | 128 / 30, 24, 1 |
| total | 5894 | 4538 |

**2. Bytes a token**, from the shards' headers: every dense language-model tensor a decode forward reads whole. The experts, vision, the PLE table, embed rows and MTP are left out.

| | gated deltanet | hyper-conn. | head | attention | shared expert | router | total |
|---|---|---|---|---|---|---|---|
| TQ4 (8-bit, g64) | 2.218 GB | 0.682 | 0.675 | 0.656 | 0.251 | 0.067 | **4.549 GB** |
| TQ4L12 (4-bit, g32; routers 8-bit) | 1.306 | 0.402 | 0.397 | 0.386 | 0.148 | 0.067 | **2.705 GB** |

So cloud's 4.5 GB a token is right.

**GB/s by mat-vec shape**
- Method: Q4E_PROFILE=3, one dispatch a command buffer, so each time carries a fixed cost of about 4 µs. In a scratch build only, the profile key adds rows × cols (3 lines in qwen4_gpu.m, not committed).
- Bytes: 1.0625 B a weight at 8 bits g64, 0.625 at 4 bits g32. Peak: 546 GB/s nominal.

| mat-vec (rows x cols) | 8-bit: us a call, GB/s | 4-bit: us a call, GB/s |
|---|---|---|
| 248320 x 2560 (head) | 1366, **495** | 794, 501 |
| 12288 x 2560 | 74.8, 447 | 45.2, 435 |
| 10240 x 2560 | 63.5, 438 | 38.7, 424 |
| 6144 x 2560 | 40.4, 414 | 25.9, 379 |
| 2560 x 6144 | 44.7, 374 | 29.1, 338 |
| 320 x 10240 (k_mv_split) | 11.3, 308 | 8.8, 232 |
| 10240 x 320 | 13.9, 251 | 9.5, 216 |
| 640 x 2560 (k_mv_gu, the shared expert) | 15.1, 230 | 12.0, 171 |
| 2560 x 640 | 8.1, 216 | 7.7, 133 |
| 512 x 2560 | 8.5, 164 | 6.7 (4-bit), 122 |
| 48 x 2560, 4 x 10240, 1 x 2560 (k_mv_split) | 4.4, 5.4, 4.3 | 4.0, 4.9, 4.0 |

- The 8-bit kernel reaches the same or higher GB/s as the 4-bit one on every shape: 68–91% of the nominal peak on the large ones.
- In this isolated sum the dense mat-vecs take 3473 ms (8-bit) against 2431 ms (4-bit): 13.5 against 9.5 ms a token.
- The extra 1.04 s is the extra bytes. Unprofiled, the gap is 2.5 ms a token for 1.84 GB, because dispatches overlap in one command buffer.

**What could still help, in both formats (your call):**
- The small mat-vecs of the hyper-connections, the DeltaNet a / b and the routers run at 160–310 GB/s or at the fixed dispatch cost.
- On the 8-bit pack, the shapes under 300 GB/s sum to about 1364 ms in the isolated profile (5.3 ms a token): 10240 x 320, 320 x 10240, 640 x 2560 gu, 512 x 2560, 2560 x 640, and the 48 / 4 / 1-row splits.
- Fusing them (for example a hyper-connection's projections in one dispatch, or the a / b / router rows in one k_mv_split) is a kernel project for both packs. I'd guess a few % of decode, but it needs a test to know.
- 2560 x 6144 (374 GB/s) might gain about 10% from a K split: about 0.2 ms a token.

I'll leave the scratch worktree for any follow-up runs.
