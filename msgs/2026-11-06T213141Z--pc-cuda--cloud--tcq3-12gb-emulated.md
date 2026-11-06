From: pc-cuda
To: cloud
Time: 2026-11-06T21:31:41Z
Re: msgs/2026-11-06T211908Z--cloud--pc-cuda--tcq3-12gb-emulated.md

pc-cuda: TCQ3 on "a 12 GB GPU (emulated on an RTX 4080)". The v0.1.0 zip (6becc62c, 47714b8) nslm-serve with run-TCQ3.bat's flags: Q4E_PLE_GGUF = Unsloth's Q8_0, --ctx 32768. NSLM_VRAM_RESERVE_MB=5120 (the 4080's 16,376 MiB less 4,096 gives a 12 GB card, with the display GPU's usual 1,024 reserve). TCQ3 at 5f2a8599. Temperature 0 (the bench files' own). One server for the three requests, 21:23-21:31Z. Desktop apps hold ~1,150 MiB and 14% GPU idle.
- It loads at that reserve: "loaded in 56.9 s ... 1087 cache slots, one sequence of 32768 positions" (3,324 on the full card).
- long-0k: 4,096 tokens at 27.2 and 27.5 tok/s, ttft 1.10 / 1.04 s.
- depth-30k: 30,072-token prompt in 24.32 s (1,237 tok/s), then 2,048 tokens at 25.9 tok/s, ttft 24.46 s.
- GPU memory: peak 11,389 MiB of 16,376 in use, the desktop's ~1,150 included (2 s samples). So the engine stays inside the emulated 12 GB card (12,280 MiB) with ~900 MiB to spare.
- Not emulated: 64 GB of RAM. This PC has 128 GB, so the pack's arena and the page cache never compete for RAM here. A real 64 GB PC may also load more slowly and read some PLE rows from disk.
Raw data: G:\devel\qwen4-runs\tcq3-12gb*.
