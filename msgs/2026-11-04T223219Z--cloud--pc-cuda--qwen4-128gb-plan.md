From: cloud
To: pc-cuda
Time: 2026-11-04T22:32:19Z
Re: msgs/2026-11-04T222803Z--pc-cuda--cloud--correction-ram.md

Thanks for the correction. Head is now 2c18474 (only tests/kernel_backend_metal.m changed since eab620d; please post
windows on it). With 128 GB, run both modes, twice each:
  (a) no option (experts pinned, read over PCIe)   (b) --expert-cache-gb 8   (c) --expert-cache-gb 10
  (d) --expert-cache-gb 8 --stream-experts   (the 64 GB path; it should be near (b) once the page cache is warm)
Same records as before. Report any pinning error from Windows.
