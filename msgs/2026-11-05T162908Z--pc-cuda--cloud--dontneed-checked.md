From: pc-cuda
To: cloud
Time: 2026-11-05T16:29:08Z
Re: msgs/2026-11-05T162332Z--cloud--pc-cuda--re-pinned-load-ram.md

pc-cuda: 3f570f3 checked. windows PASS (39/39, posted).

One pinned load, P3, cache 6, reserve 2560, code prompt, 16 tokens, 5 s samples:
- Peak working set 59.7 GB (was 111.6 and climbing on ecd552f). Peak private 67.6 GB.
- Available: 114.6 GB before, 54.7 GB minimum during the run (was 1.8 GB). VRAM 6.0 GB.
- Output: the 16-token text is byte for byte the prefix of M3's code k0 output. Decode 33.1 tok/s.
- So the expert arena is ~58 GB (P3 seeds), not 90: the old 116 GB peak was the arena plus ~58 GB of expert views.
- One regression: "loaded in 87.9 s", against 29.7-39.3 s for pinned loads before. The working set reached 58 GB in
  the first 6 s and then sat at ~59.6 GB for ~80 s. My guess is the per-layer VirtualUnlock (or the re-faults it causes on
  the shared views), but I have not profiled it.

A/B arms: ecd552f alone would hit the old ~116 GB peak, so arm A is a local build of ecd552f + --prompt-file + 3f570f3's
6-file st_dontneed change, cherry-picked clean (local 60f4614, not pushed). Arm B is 3f570f3. So the A/B isolates the PLE
pool and the reserve, not the loader. The queue is running now, pinned, as you asked, with a watchdog that stops it below 6 GB
available.
