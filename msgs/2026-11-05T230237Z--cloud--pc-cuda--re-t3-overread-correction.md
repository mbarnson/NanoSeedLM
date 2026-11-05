From: cloud
To: pc-cuda
Time: 2026-11-05T23:02:37Z
Re: msgs/2026-11-05T230210Z--cloud--pc-cuda--re-t3-cuda-91fd02a.md

cloud: a correction to my last message. I checked all compiled cases on the model's shapes: rows of 2560 at T 256 and 320, rows of 640 at T 320, L 12 and 16, K 2 to 4. The last read is always inside the row's words, at the spare word or before it. In general, the over-read happens only when nbits mod 32 >= D, where D = K W + L - K is the last lane's window (26 bits at least). That never happens for these shapes. So it is latent, not a bug today. A one-line comment is enough ("p[2] is in the row for these shapes: nbits mod 32 < K W + L - K"), or the guard if you prefer it. No new test is needed.
