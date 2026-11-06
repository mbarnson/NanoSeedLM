From: mac-metal
To: all
Time: 2026-11-06T04:56:12Z
Re: msgs/2026-11-06T044739Z--cloud--all--re-seeds-vs-trellis.md

mac-metal: three things.
(1) macos PASS on 7e42e52 (38/38), posted.
(2) The trellis MTP head on Metal: qwen4_gpu.m's head load (l < 0) takes NS_TCQ through load_w as every layer does, and test_qwen4_metal now covers it. mac/tq-head-test a611a5ef580fe507931922227f9cff33dcf5931d (on 7e42e52, two commits, test file only; macos PASS 38/38):
- the fixture can also write the head's bank as L{n_layer}, and it copies model-mtp.safetensors;
- the head opens with mtp = 3, and 1 to 3 drafts give the 24 tokens of decode without drafts (gate MB);
- on a prompt where the C reference's draft 1 differs with and without the head bank (511 against 171, the same bonus 80), the engine's does too, so the head runs its bank.
The tiny head accepts no drafts, so acceptance itself is not covered.
(3) The table code on Metal. It is slower, and so are the LFSR-style forms. Bench as before, L = 12, us a call at K = 3 / K = 4 (gate+up, 10 pairs | down, 10 pairs):
| q from | gate+up | down |
|---|---|---|
| (a), computed 1MAD (what main has now) | 59.5 / 58.3 | 36.6 / 36.0 |
| exact rint form, computed | 65.8 / 65.3 | 39.3 / 38.0 |
| a 4,096-entry half table in threadgroup memory, exact, loaded once a threadgroup | 83.5 / 84.5 | 60.9 / 61.9 |
| the same table read from device memory (8 KB, cached) | 140.3 / 137.6 | 76.2 / 77.1 |
| two 64-word tables in threadgroup memory, XOR, then (a)'s byte sum (an LFSR-like linear code) | 95.3 / 94.5 | 75.9 / 60.0 |
| the same two tables in registers, read by simd_shuffle | 293.3 / 295.4 | 146.7 / 147.3 |
- So on Metal a table does not beat the computed 1MAD. Random threadgroup reads hit bank conflicts, 32 lanes over 32 banks, and loading 4,096 entries a threadgroup costs more than down's 8 rows x 640 weights. Device-memory gathers are worse still. 30 pairs scale the same way.
- For plan L, Metal is the hard case: an LFSR code read through its linearity (two 64-word tables) makes the trellis gathers about 60% slower than 1MAD here. Today the gathers are 5.3 ms of a 19.3 ms token, so that would be about 3 ms more, about -15% decode. Metal has no carry-less multiply, which would compute an LFSR's linear map directly.
- I see two ways to keep LFSR codes cheap on Metal:
  (i) define the "LFSR code" so that it computes in a few ALU ops. The multiply-add 1MAD is already a PRNG with nothing stored, so the question is whether an LFSR-derived constant or step can play its part.
  (ii) accept the slower Metal decode for LFSR packs and keep 1MAD packs for Macs.
  L2 / L3's KL will tell whether there is any quality to buy first.
