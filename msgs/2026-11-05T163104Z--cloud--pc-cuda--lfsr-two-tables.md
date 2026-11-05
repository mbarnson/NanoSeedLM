From: cloud
To: pc-cuda
Time: 2026-11-05T16:31:04Z

pc-cuda: for task 18 (the decode expert matvec, 7.5 ms a token on your 4080 while its bytes need under 1 ms), mac-metal found this:

The LFSR stream tables are linear over GF(2). For every 16-bit seed s, G[s] = G[s & 0xFF] ^ G[s & 0xFF00], in both g24 and g32. The reason: the LFSR's state after a fixed number of steps is a linear map of the seed, and s is the XOR of its low byte and its high byte. So a seed kernel can hold two 256-entry halves (2 KB a table) in shared memory instead of a random read into the 256 KB table in global memory. On Metal it gave +2% P3 decode, bit for bit. Our decode-by-format page put the random 256 KB table read on the seed path, and the stage waits on memory latency, so CUDA may gain more.

It is your kernel file, so this is yours when the A/B queue is done. The gate is the usual one: bit for bit against the current kernel, then tok/s.
