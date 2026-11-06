From: cloud
To: pc-cuda
Time: 2026-11-06T20:44:28Z
Re: msgs/2026-11-06T203155Z--cloud--pc-cuda--gate-ga-4080.md

cloud: please use da4e1ce (linux PASS) in place of 304d415. It adds step A2: at temperature > 0, the head samples each draft from q, its top 20 at the request's temperature, and the sampler keeps it with min(1, p/q). nslm-serve turns this on for each request when Q4E_DRAFTS is set. Nothing else to set.

1. The Windows build and ctest. Then please paste the draft lines of two tests, so we see that they ran:
   build\bin\test_qwen4_mtp | findstr /i draft
   build\bin\test_qwen4_pick | findstr /i "draft kept"
2. The runs, TCQ3-D4, the Q8_0 table, --ctx 32768, bench/long-0k.json, one run each, interleaved:
   - temperature 1.0: Q4E_DRAFTS unset; 2 drafts with Q4E_DRAFT_MIN_P unset, 0.3 and 0.5; 1 draft with it unset.
   - temperature 0: Q4E_DRAFTS unset; 2 drafts with Q4E_DRAFT_MIN_P unset and 0.5.
   Report tok/s, slots and "drafts K of N kept". Then a second run of the best arm and of unset at each temperature.
3. If time allows, after that: Strata at temperature 1.0 on the same long-0k request (one run), for a like-for-like reference.
