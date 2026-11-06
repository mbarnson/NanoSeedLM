# MTP drafts with trellis experts on the RTX 4080, with and without a trellis head (2026-11-05)

**Result.** With the MTP head's experts in trellis banks, the head joins the expert cache. Drafts then cost 236 to
301 cache slots, not 976 to 1,120, and MTP drafts break even on the RTX 4080. The mix decodes 5% faster with one
draft and 1% slower with two, and K = 3 is 2% slower with two. With the head in P3 seeds, drafts were 6% to 12%
slower. The round-to-round noise of this session was about 10%, so drafts stay off by default on CUDA.

**Status.** Current.

## Setup

- Machine: pc-cuda's Windows 11 PC, RTX 4080, the default VRAM reserve (1,024 MB on this display GPU). The PC ran
  nothing else.
- Builds: clang-cl Release. Without the head banks: `48094f2`, 04:15 to 04:28 UTC. With them: `7e42e52`, 04:52 to
  05:05 UTC.
- Packs: the trellis mix (K = 3, `down_proj` at K = 4) and trellis K = 3, as in
  [Trellis experts on the RTX 4080](2026-11-05-trellis-speed-4080.md).
  - Without the head banks, the head's experts are P3 seeds. They stay fixed in VRAM.
  - With them, the head's experts are banks `L48_*.blkt`, in the K of each projection of the pack. `ns_add_banks`
    (`1d0bb37`) takes a bank of layer 48 as the head's.
  - pc-cuda encoded these banks on the RTX 4080 in 139 s with the CUDA encoder. The source was the BF16 head, with
    uniform weights.
- Runs: `nslm-qwen4-gen`, the code prompt, 256 tokens, the automatic cache, `--drafts 0`, 1 or 2, two rounds each.

## Results

Without the head banks (the head's experts in P3 seeds, in VRAM):

| Pack, drafts | Tokens per second (2 rounds) | Slots | Misses / hits | Tokens a forward |
|---|---|---|---|---|
| Mix, 0 | 54.43 / 48.37 | 4,581 | 35,189 / 95,955 | 1 |
| Mix, 1 | 49.69 / 46.73 | 3,605 | 44,134 / 74,931 | 1.86 |
| Mix, 2 | 48.17 / 48.21 | 3,569 | 45,537 / 70,907 | 2.59 |
| K = 3, 0 | 61.59 / 62.55 | 5,070 | 30,339 / 100,817 | 1 |
| K = 3, 2 | 53.41 / 56.02 | 3,950 | 41,280 / 72,162 | 2.67 |

With the head banks (the head's experts in the cache):

| Pack, drafts | Tokens per second (2 rounds) | Slots | Misses / hits | Tokens a forward |
|---|---|---|---|---|
| Mix, 0 | 48.50 / 53.98 | 4,581 | 35,189 / 95,955 | 1 |
| Mix, 1 | 53.91 / 53.91 | 4,345 | 39,235 / 82,100 | 1.84 |
| Mix, 2 | 53.09 / 48.78 | 4,310 | 40,191 / 76,472 | 2.61 |
| K = 3, 0 | 61.99 / 58.18 | 5,070 | 30,339 / 100,817 | 1 |
| K = 3, 2 | 58.84 / 58.51 | 4,769 | 37,054 / 81,484 | 2.59 |

Against the mean of each pack at 0 drafts:

| Pack, drafts | Without the head banks | With the head banks |
|---|---|---|
| Mix, 1 | -6% | +5% |
| Mix, 2 | -6% | -1% |
| K = 3, 2 | -12% | -2% |

Each drafted text was the text of 0 drafts (gate MB). The mix's md5 was 88187647, and K = 3's was 95a5e359.

## Notes

- The head banks take the misses of a draft run from 9,000 to 11,000 above 0 drafts down to 4,000 to 7,000.
- The rounds of one setting varied from 48.5 to 54.0 tokens per second. More rounds, or `Q4E_PROFILE`, can show
  whether one draft on the mix is a real gain.
- The banks of the head add 0.98 GB (K = 3) and 1.09 GB (the mix) to a pack's files. The P3 head's experts stay in the
  folder, but the engine does not read them.
