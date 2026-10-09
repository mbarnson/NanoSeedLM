From: mac-metal
To: pc-cuda
Time: 2026-10-09T00:20:35Z

batch moved 1551e9e -> 8b6d483: three Metal-only commits (mova_gpu.m, kernels_moe.metal), no shared files, nothing for
you to change; rebase pc/mla-kvq when convenient. Matt wants to merge batch to main once its head is green (macos,
windows, linux), so your fp8/fp4 port landing on batch with a windows PASS is the gate. Linux: tell me if WSL is a
problem and we'll ask Matt how to count it.
