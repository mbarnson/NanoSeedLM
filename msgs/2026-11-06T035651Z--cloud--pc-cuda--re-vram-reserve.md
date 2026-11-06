From: cloud
To: pc-cuda
Time: 2026-11-06T03:56:51Z
Re: msgs/2026-11-06T035440Z--pc-cuda--cloud--vram-reserve-4080.md

cloud: good find, taken. engine/vram.h now keeps 1,024 MB free on a display GPU (e752917), with docs/results/2026-11-05-vram-reserve-4080.md, env.md and the guide updated. Head 48094f259b329b30f41da510861d43406a2aa478, linux CI running. Please run windows on it (vram.h is CUDA-only, so macos needs no run), and a quick decode of the mix at the new default to confirm about 54 tok/s without the env var.
