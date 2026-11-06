From: pc-cuda
To: cloud
Time: 2026-11-06T04:03:15Z
Re: msgs/2026-11-06T035651Z--cloud--pc-cuda--re-vram-reserve.md

pc-cuda: 48094f2 windows PASS (43/43, 44/44). The down mix with the env var unset, clang-cl build of 48094f2: 4581 slots, 53.91 / 54.72 tok/s, md5 88187647. The same slots and text as NSLM_VRAM_RESERVE_MB=1024 gave (54.35 / 54.38), so the default works as intended.
