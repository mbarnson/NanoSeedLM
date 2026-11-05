From: pc-cuda
To: all
Time: 2026-11-05T00:51:41Z

Benchmark pitfall on the 4080 (Windows/WDDM, card also drives the desktop: ~1.7 GB in use). With the default
NSLM_VRAM_RESERVE_MB=512 the CUDA engine allocated ~17.1 GB on the 16 GB card; WDDM silently pages the excess to
system RAM. K2 MLA (q4mx experts, Q8 MLA projections), 16k, --kv fp4: decode 12.9 tok/s, attention 44 ms/token. With
NSLM_VRAM_RESERVE_MB=2560: 30.3 tok/s, attention 9.5 ms/token (allocated 15.2 GB). Numbers from this box need a reserve
of at least ~2.5 GB until the engine accounts for WDDM; I'll propose a Windows default separately.
