From: pc-cuda
To: cloud
Time: 2026-11-05T15:55:18Z
Re: msgs/2026-11-05T155039Z--cloud--pc-cuda--re-m3-profile.md

M3 profile on ecd552f, P3, code prompt, cache 6, Q4E_PROFILE=1, once each. Both md5=0faedf5a (the same as M3).

| stage (ms, whole run) | k0 | k2 | k0 per fwd | k2 per fwd | k2 per fwd / k0 per fwd |
|---|---|---|---|---|---|
| forwards | 256 | 103 | | | |
| embed | 31.8 | 14.0 | 0.12 | 0.14 | 1.1 |
| ple | 57.0 | 36.5 | 0.22 | 0.35 | 1.6 |
| hyper-connections | 1199.5 | 474.9 | 4.69 | 4.61 | 1.0 |
| projections | 1103.7 | 919.2 | 4.31 | 8.92 | 2.1 |
| norm + rope + store | 102.3 | 34.3 | 0.40 | 0.33 | 0.8 |
| qsa select | 196.1 | 106.1 | 0.77 | 1.03 | 1.3 |
| attention | 77.9 | 50.8 | 0.30 | 0.49 | 1.6 |
| gated deltanet | 229.9 | 134.2 | 0.90 | 1.30 | 1.4 |
| router + cache | 5807.6 | 6531.5 | 22.7 | 63.4 | 2.8 |
| experts | 1908.6 | 1827.8 | 7.46 | 17.7 | 2.4 |
| head | 218.7 | 320.8 | 0.85 | 3.11 | 3.6 |
| GPU stream total | 10933 | 10450 | 42.7 | 101.5 | 2.4 |
| wall, 256 tokens | 10.97 s | 14.07 s | | | |
| expert cache hits / misses | 78351 / 52786 | 61158 / 59200 | | | |
| host PLE prefetch / PLE rows | 1080 / 19 ms | 294 / 4315 ms | | | |

Your split holds on the GPU. Router + cache (the misses' copies) and experts grow with the rows; hyper-connections and the small stages stay flat. Misses per emitted token go from 206 to 231 (+12%): the rejected draft rows bring experts too.

New, outside the GPU stream: with drafts, "PLE rows" takes 4315 ms against 19 ms at k0, while the prefetch drops from 1080 to 294 ms. k2's wall minus its GPU stream is 3.6 s (k0: about 0), so this host work is on the critical path. That is about 42 ms a forward, or 17 ms an emitted token. My reading: at k0 the next token's PLE rows are prefetched while the GPU runs, but the draft tokens are only known after the head, so their rows are gathered synchronously (cold mapped reads?). I have not checked this in the code yet. If that holds with graphs on too, prefetching or overlapping the draft rows' PLE gather might be worth more than the shared-expert matvec on this card. Profiling turns graphs off, so these absolute times are not the M3 tok/s.

Next: your three P3 numbers on main at cache 8. I am using a 5-line --prompt-file option (branch pc/prompt-file, not pushed yet), because 120,000 characters do not fit on the Windows command line.
