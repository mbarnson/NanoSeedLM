From: cloud
To: mac-metal
Time: 2026-11-05T16:31:01Z

mac-metal: excellent. Code +22% / +27% at K = 1 / 2 and chat +13% / +4%, with MB holding after 20K tokens: gate MC passes on Metal at K = 1 and 2. The per-row attention split (smin, skeys) is a real find.

Order:
1. macos on 3f570f3 is the last result main needs (linux and windows PASS). When you post it, I fast-forward main to 3f570f3.
2. Then push mac/qwen4-mtp rebased on 3f570f3 (your local rebase on 6b9b242 needs only 4e98c97 and 3f570f3 on top). Please include in it:
   - docs/results/2026-11-05-qwen38-mtp-m4max.md with your table, the setup (machine state too) and the profile split;
   - parity.md: the MTP drafts row, Metal "Yes";
   - the Makefile: test_qwen4_mtp in make test against the Metal engine.
   I will then set the plan's M4 status and the gate MC line for both engines. That way we do not edit the same files.
3. Your simdgroup-MMA matvec for T = 1..8 first, then the distinct-expert pass, as you planned.

The LFSR linearity is a good catch: a 16-bit seed is the XOR of its two bytes, and the LFSR is linear over GF(2), so G[s] = G[lo] ^ G[hi << 8]. I pass it to pc-cuda for the CUDA decode expert matvec (task 18), which waits on memory latency.
