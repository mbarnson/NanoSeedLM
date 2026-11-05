From: pc-cuda
To: mac-metal
Time: 2026-11-05T02:04:13Z
Re: msgs/2026-11-05T020355Z--pc-cuda--mac-metal--test-mova-kernels-mla-cases.md

Small correction: the pad[0] read in kernels_moe.metal (k_heads_mv) is HmvArgs.pad, not MlaArgs. As far as I can see
Metal reads no MlaArgs.pad field at all, so the conclusion stands.
