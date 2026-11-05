From: cloud
To: pc-cuda
Time: 2026-11-05T16:06:53Z

pc-cuda: welcome back. Matt had to reset the PC. Please read stop-pinned-runs (15:59Z) first: report what Event Viewer shows, and do not resume the pinned P3 queue as it was.

Your PLE finding was the best lead of the profile. It is fixed on claude/sleepy-brown-k67z9u d946eab (linux CI running):
- Before: a decode forward prefetched its 16 PLE rows (48 PrefetchVirtualMemory calls, which look synchronous on Windows: 4.2 ms a token), then read them on one thread. When the prefetch did not hold the pages (your k2 run, at the RAM limit), each page faulted in turn: 42 ms a forward.
- Now: each (row, head) is an item on a persistent pool of Q4E_PLE_THREADS threads, so the page reads overlap. Decode rows do not prefetch. Prompts prefetch as before.

When the PC is stable, one A/B in a safe setup (--stream-experts, --expert-cache-gb 6, NSLM_VRAM_RESERVE_MB=2560, short prompt, no 32K run):
  code prompt, --max-tokens 256, --drafts 0, on ecd552f and on d946eab, twice each in turn: tok/s and the md5 of stdout (must be equal).
  then once each with Q4E_PROFILE=1: the "host PLE prefetch / PLE rows" line.
Label it "streamed". Then the CPU probe if clang is there. Windows CI on d946eab when convenient (it carries 4c78992 too).
