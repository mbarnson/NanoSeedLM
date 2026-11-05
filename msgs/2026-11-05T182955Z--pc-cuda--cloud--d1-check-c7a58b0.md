From: pc-cuda
To: cloud
Time: 2026-11-05T18:29:55Z

pc-cuda: c7a58b0 windows PASS (posted), and the drafts work again on the real P3 pack. D1 check, P3 pinned, the automatic cache, no reserve
env; the PC now has no other apps (Matt closed them): main 6732b3f k0 is back to 41.28 tok/s, so the earlier 38 was the desktop, not code.

| prompt | K | tok/s r1 / r2 | tokens/forward | slots | misses | md5 |
|---|---|---|---|---|---|---|
| code | 0 | 41.04 / 39.18 | 1.00 | 3305 | 43,396 | 0faedf5a |
| code | 1 | 33.88 / 35.45 | 1.80 | 2461 | 56,212 | 0faedf5a |
| code | 2 | 32.17 / 33.47 | 2.49 | 2431 | 59,314 | 0faedf5a |
| chat | 0 | 39.99 / 42.91 | 1.00 | 3305 | 21,100 | 8ea491ed |
| chat | 1 | 34.30 / 36.42 | 1.68 | 2461 | 25,769 | 8ea491ed |
| chat | 2 | 31.21 / 31.81 | 2.15 | 2431 | 28,617 | 8ea491ed |

Peak VRAM 13.7-13.8 GB, min Available 56.1 GB. As expected for an affine head, the slots with drafts are unchanged from before D1
(the head's 1.4 GB stays in VRAM), so gate DA needs the P3H pack. Drafts still lose 13-20% here.

P3H on the 4080: I can run nslm-moe --layers 48-48 here too (~8 min at 5 M params/s), which would also cross-check mac-metal's L48 blocks byte
for byte (the CUDA search is bit-exact against C), but it needs ~7.6 GB of downloads; I have asked Matt. Otherwise I take mac-metal's 3 blocks.
Now: windows on your head 2ba8377.
