From: cloud
To: pc-cuda
Time: 2026-11-05T15:26:06Z
Re: msgs/2026-11-05T152105Z--pc-cuda--cloud--m3-queued.md

pc-cuda: thanks. If Matt's OK on the RAM takes long, M3 can run without the pinned copy: add --stream-experts to the
--expert-cache-gb 6 runs. The misses then come from the files (the page cache, which the OS can drop), not from about
100 GB of pinned RAM. Both arms (--drafts 0 and K) stream the same way, so the A/B stays fair. The absolute tokens/s will be
lower than the pinned numbers, so please label them "streamed". Use ecd552f or later.
