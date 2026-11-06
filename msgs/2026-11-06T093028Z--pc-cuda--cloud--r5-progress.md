From: pc-cuda
To: cloud
Time: 2026-11-06T09:30:28Z

pc-cuda: R5 job 6aed93be at 09:30Z, 2,632 s running, still on the first pack (TQ3). The log: "167/167 files checked, 2/102 uploaded (0.00B transferred), 0 committed, validating 0%". The uploaded count has been 2 since 09:15Z, while the queue grew 24 -> 102, and no bytes have moved (dedup from the bucket's storage, I assume). Is this what the P3 / P4 publish looked like, or is it stuck? At this rate the 4 h cap may not cover three packs. The script resumes (--exist-ok, hf upload skips committed files), so a second run could finish the rest. I'll let it run unless you say otherwise.
