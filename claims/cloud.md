# cloud claims (updated 2026-11-06T21:19:11Z)

# cloud claims (updated 2026-11-06T21:20Z)

**Hand-off.** This cloud session ends here. A new cloud session takes over with a new goal. Keep posting results and messages to `cloud`: the new session reads this branch.

- **Released:** v0.1.0 (tag `47714b8`, a pre-release, published 2026-11-06 20:47Z). Main is `fb1504f`: only docs and tools since the tag.
- **Hugging Face:** all five cards come from `tools/qwen4_release_card.py` on main and the logs in `kld/`. Do not edit a README on the Hub: change the generator, then upload all five.
  - Each card has the same sections: which pack for which machine, the family tables, the trade-offs, long-run decode and "Why NanoSeedLM". Each table fits the Hub's 600-pixel card column (checked with Playwright).
  - The TCQ repositories now carry their MLX model files, `model_file` and LICENSE (TCQ4 `e53ffdd2`, TCQ3-D4 `30012655`, TCQ3 `d1353e4a`). `tools/hf_release_publish.sh` refuses a folder without them.
  - Open: mac-metal checks mlx-vlm and oMLX on the three TCQ repositories (msgs/...tcq-mlx-files). Then the cards' "tested" notes change. P4's long runs and KL as shipped. TCQ3 on a 12 GB GPU (pc-cuda, emulated).
- **Branch `claude/sleepy-brown-k67z9u` `da4e1ce`, for v0.1.1:** linux PASS, windows PASS, macos PASS at `73abfdf` (not yet at `da4e1ce`).
  - It has: plan K (the BF16 KV cache), the PLE emulator (`ple:TIER`), one sampler (`harness/sample.c`), and drafts in nslm-serve and nslm-chat at any temperature (`eng_generate_pick`).
  - It also has: the draft gate (`Q4E_DRAFT_MIN_P`, A1), drafts sampled from the head's q (A2), and `nslm/rng.h`.
  - Merge to main when macos passes `da4e1ce`, with `mac/qwen4-merged-proj` `b80abc4`. Then set VERSION 0.1.1 to start the next draft.
- **Open gates:**
  - pc-cuda: GA and GB on the RTX 4080 (`docs/plans/2026-11-06-draft-acceptance.md`), then Strata at temperature 1.0, then TCQ3 on an emulated 12 GB GPU.
  - mac-metal: macos at `da4e1ce`, gate SC on the M4 Max, the P4 numbers, the Metal PLE row fix, A1m and A2m.
- **Pending:**
  - R8: the signing secrets go into a `release` environment (`docs/plans/2026-11-05-release-family.md`).
  - With v0.1.1, in the start scripts: pin the new HF revisions, show GB in decimal (pc-cuda's nit), drop the empty line in pack mode, and say "a 12 or 16 GB GPU" for TCQ3.
  - A results page for gates SC, GA and GB.
- **Proposed next goal (Matt to confirm):**
  - On an RTX 4080 with a TCQ pack, more than 60 tok/s for one request at the card's defaults, and more than 60 in aggregate in nslm-serve.
  - KL as shipped unchanged, and Metal at parity.
  - The pieces: drafts that pay (A1, A2), a CPU kernel for the trellis experts of verify rows (a plan to write), and batched Qwen decode in nslm-serve (a plan to write).
- **HF budget:** about $1.46 left this week (`budget/hf-spend.md`). The week resets Thursday 17:00 Pacific. Linux CI: `bin/hf-linux-ci.sh REV`, about $0.08.
