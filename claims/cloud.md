# cloud claims (updated 2026-11-06T20:59:34Z)

# cloud claims (updated 2026-11-06T21:00Z)

**Hand-off.** This cloud session ends here. A new cloud session takes over with a new goal. Keep posting results and messages to `cloud`: the new session reads this branch.

- **Released:** v0.1.0 (tag `47714b8`, a pre-release, published 2026-11-06 20:47Z). Main is `0c20dd9`: only docs since the tag.
- **Hugging Face cards:** TCQ4 `163d3d5f`, TCQ3-D4 `9be4e986`, TCQ3 `5da7791d`, from `tools/qwen4_release_card.py` on main and the logs in `kld/`. The P3 and P4 cards still need the lead about the LFSR seeds' benefits, with long-run numbers. P3's long runs are in `results/2026-11-06-long-bench-m4max.md` on main. P4 waits for mac-metal.
- **Branch `claude/sleepy-brown-k67z9u` `da4e1ce`, for v0.1.1:** linux PASS, windows PASS, macos PASS at `73abfdf` (not yet at `da4e1ce`).
  - It has: plan K (the BF16 KV cache), the PLE emulator (`ple:TIER`), one sampler (`harness/sample.c`), and drafts in nslm-serve and nslm-chat at any temperature (`eng_generate_pick`).
  - It also has: the draft gate (`Q4E_DRAFT_MIN_P`, A1), drafts sampled from the head's q (A2), and `nslm/rng.h`.
  - Merge to main when macos passes `da4e1ce`, with `mac/qwen4-merged-proj` `b80abc4`. Then set VERSION 0.1.1 to start the next draft.
- **Open gates:**
  - pc-cuda: GA and GB on the RTX 4080 (`docs/plans/2026-11-06-draft-acceptance.md`), then Strata at temperature 1.0.
  - mac-metal: macos at `da4e1ce`, gate SC on the M4 Max, the P4 numbers, the Metal PLE row fix, A1m and A2m.
- **Pending:**
  - R8: the signing secrets go into a `release` environment (`docs/plans/2026-11-05-release-family.md`).
  - Pin the new HF revisions in the start scripts, and show GB in decimal (pc-cuda's nit), with v0.1.1. The scripts must also drop the empty line in pack mode.
  - A results page for gates SC, GA and GB.
- **Proposed next goal (Matt to confirm):**
  - On an RTX 4080 with a TCQ pack, more than 60 tok/s for one request at the card's defaults, and more than 60 in aggregate in nslm-serve.
  - KL as shipped unchanged, and Metal at parity.
  - The pieces: drafts that pay (A1, A2), a CPU kernel for the trellis experts of verify rows (a plan to write), and batched Qwen decode in nslm-serve (a plan to write).
- **HF budget:** about $1.46 left this week (`budget/hf-spend.md`). The week resets Thursday 17:00 Pacific. Linux CI: `bin/hf-linux-ci.sh REV`, about $0.08.
