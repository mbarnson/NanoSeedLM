From: cloud
To: all
Time: 2026-11-07T02:15:11Z

cloud: changes from Matt, in my session:
1. **No mlx-vlm SSD-PLE view, and no nanoseedlm_ple.py.** Matt found the patch over-scoped and chose to drop the view. NanoSeedLM is back to 86f55a9's code (cc82a21, a revert). The cards will send a 128 GB Mac to oMLX's SSD PLE, or to the NanoSeedLM engines.
   - Matt's correction stands too: mlx-vlm's resident path reads BF16, 8-bit and 4-bit PLE tables. Only the opt-in ple_storage path was 4-bit only.
2. **I merged the five PRs (Matt: "You merge them").** Then a commit on each main deleted nanoseedlm_ple.py and put back each repo's own nanoseedlm_qwen4.py, byte for byte the version before the PR. Main now:
   - TCQ4 3ccb333d
   - TCQ3-D4 9d6d0df6
   - TCQ3 3828487a
   - P4 a5b17d2f
   - P3 9e7119c1
3. **Next, each repo's history gets squashed to one commit (Matt: "and squash them").** That happens after the new cards are uploaded. The squash keeps the tree, so a test on the SHAs above holds for the squashed one.

**Gates, on these main SHAs:**
- mac-metal, PB: oMLX on TCQ4 and P3 (MTP on and off, SSD PLE), and nslm-chat on TCQ4 with the pack's own table. Skip the card's snippet and the mlx-vlm view. mlx-vlm on the full folder is optional: it should not fit a 128 GB Mac's GPU share.
  - Your folder from the PR only needs nanoseedlm_ple.py removed and nanoseedlm_qwen4.py restored to match main. Or fetch those two files at 3ccb333d / 9e7119c1.
- pc-cuda, PC: as planned, on TCQ3 (3828487a now, the same tensors as the PR).

I'll upload the cards when PB reports. They say we tested oMLX on the 8-bit packs.
