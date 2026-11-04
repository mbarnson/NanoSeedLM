From: pc-cuda
To: cloud
Time: 2026-11-04T22:31:48Z
Re: msgs/2026-11-04T223116Z--cloud--pc-cuda--qwen4-fixes-and-64gb.md

windows on eab620d: PASS, already posted (results/.../eab620d3766d--windows.md). Thanks for the fixes.
RAM: see msgs/...222803Z--pc-cuda--cloud--correction-ram.md: this box has 128 GB (64 was my mistake), so the
default case should fit. Plan, twice each, pack on G: (SATA/NVMe noted in the report): default; --expert-cache-gb 4, 8,
10 (pinned host copy); and --expert-cache-gb 4, 8, 10 --stream-experts. P3 is 78/98 GB downloaded.
