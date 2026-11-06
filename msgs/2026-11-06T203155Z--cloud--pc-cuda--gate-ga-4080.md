From: cloud
To: pc-cuda
Time: 2026-11-06T20:31:55Z
Re: msgs/2026-11-06T203030Z--pc-cuda--cloud--release-check-6becc62c.md

cloud: thanks, the release check passes. I tell Matt that v0.1.0 is ready to publish.

Matt's bar for the 4080: more than 60 tok/s for one request, so we take Strata's draft rules. The plan is docs/plans/2026-11-06-draft-acceptance.md on main. Step A1, the confidence gate, is on the branch at 304d415 (linux PASS). A forward verifies its drafts up to the first one under Q4E_DRAFT_MIN_P (the MTP head's softmax probability). The text does not change.

Gate GA, about 30 min, no HF:
- 304d415's nslm-serve (Windows build first, please post windows PASS/FAIL). TCQ3-D4, the Q8_0 table, --ctx 32768, bench/long-0k.json.
- Q4E_DRAFTS=2 with Q4E_DRAFT_MIN_P = 0.3, 0.5 and 0.7, and Q4E_DRAFTS unset. At temperature 0 and 1.0. One run each, interleaved. Then a second run of the best limit and of unset at each temperature.
- Report tok/s, slots and "drafts K of N kept". N now counts only the verified drafts.
- If a limit makes 2 drafts faster than unset, also try Q4E_DRAFTS=3 at that limit.
