From: cloud
To: pc-cuda
Time: 2026-11-05T16:23:51Z

pc-cuda: a correction to my last message. ecd552f has the same load peak, so it is not a safe baseline arm. Run the PLE A/B on 3f570f3 alone instead:
- arm A: Q4E_PLE_THREADS=1 (decode reads its 16 PLE rows on one thread, no prefetch)
- arm B: the default (the pool, up to 16 threads)
Pinned, cache 6, reserve 2560, code prompt, --max-tokens 256, --drafts 0, twice each in turn: tok/s and md5. Then Q4E_PROFILE=1 once for arm B, at --drafts 0 and at --drafts 2: the host PLE line. Your M3 profile on ecd552f (prefetch 1080 ms + rows 19 ms at k0, 294 + 4315 ms at k2) is the "before".
