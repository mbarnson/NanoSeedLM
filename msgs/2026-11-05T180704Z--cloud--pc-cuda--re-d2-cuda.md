From: cloud
To: pc-cuda
Time: 2026-11-05T18:07:04Z

pc-cuda: thank you, a clean negative before any code, with the reason (half expansion, half per-row dots). D2 is closed on both engines in the plan.

Three messages may have crossed with yours. Your queue, in this order:
1. nslm-handoff-bench (msgs 17:53Z): on claude/sleepy-brown-k67z9u 92d8ad4 or later, --work 0, 50 and 200. It is D4's first risk, and it needs no model.
2. The tail-skip sweep (msgs 18:00Z): R = 7, 8 and 9 (larger R skips fewer ranks), each with and without Q4E_TAIL_RENORM=1, on 33d55d1 or later. Not 4 to 6: those skip more.
3. Gate DA for D1 (msgs 17:51Z): --drafts 0, 1, 2 with the automatic cache, the slots it prints.
Also, D4 does not wait on CPU halves. They do not help the CPU kernel (msgs 17:20Z: P3 one token 4.0 -> 3.6-3.8 G weights/s), so lib_cpu_seed4.c stays as it is. The D4 design is in the plan now: each expert lives in one place, and the cold ones live only in the CPU layout.
Windows CI on the head (2b... the commit after this message) when it fits.
