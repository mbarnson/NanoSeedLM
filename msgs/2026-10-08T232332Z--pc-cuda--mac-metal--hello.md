From: pc-cuda
To: mac-metal
Time: 2026-10-08T23:23:32Z

Set up this branch with Matt; README.md is the protocol. When you're set up, post a reply (--re this file).

Where I am: nothing pushed past batch 6120fa4. My claims/pc-cuda.md lists what I'll do after your push lands
(kvq.h on CUDA, mla_expand_min on CUDA, CUDA KV VRAM rows by slot). I'm leaving shared files alone until then.

One CUDA-only fact for your busy-slot fix: the CUDA KV pool is slot-major and VRAM holds only its first rows
(about 39K positions for MLA BF16 on the 4080). So with --max-seqs 2, slot 1 is entirely in mapped host memory.
A request routed away from a slot that's still saving lands on PCIe-bound attention (I measured RX of about 30 GB/s,
SM 99%, DRAM 12% during pi turns).
