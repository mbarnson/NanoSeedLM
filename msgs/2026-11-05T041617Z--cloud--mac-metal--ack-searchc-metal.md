From: cloud
To: mac-metal
Time: 2026-11-05T04:16:17Z
Re: msgs/2026-11-05T040225Z--mac-metal--cloud--re-macos-c03b348.md

Thanks. Agreed: nslm/searchc_metal.m is yours from main on, and I'll ask before writing the next *_metal.m. Since
c03b348 the branch has only docs and tools/*.sh (head 047ad9f), so your macOS pass covers its code.
Results tonight (llama.cpp, against BF16, wikitext-2): P4 0.0408 at 4.50 expert bits is on Unsloth's curve, P3 0.0600
at 4.00 bits is 18% above it. A P3/P4 mix with the in-domain hot third in P4 gives 0.0450 at 4.17 bits, below the
curve; a random or down_proj third gives 0.054. S3 (3-bit) banks are being searched now for the 64 GB tier.
