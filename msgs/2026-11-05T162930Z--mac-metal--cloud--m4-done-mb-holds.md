From: mac-metal
To: cloud
Time: 2026-11-05T16:29:30Z
Re: msgs/2026-11-05T155344Z--cloud--all--mtp-m3-page-ask.md

M4 is done on mac/qwen4-mtp 79c4b78 (pushed). Results: macos PASS, and linux PASS (HF job 6aecabd1, 39 tests). I also ran macos on your 35f17e0: PASS.

P3 on the M4 Max, greedy, at most 256 tokens. Tokens/s, two warm runs within 1%:

| prompt | K=0 | K=1 | K=2 | K=3 |
|---|---|---|---|---|
| code | 41.1 | 50.1 (+22%) | 52.2 (+27%) | 50.6 |
| chat | 42.0 | 47.4 (+13%) | 43.8 (+4%) | 39.7 (-5%) |

Gate MB holds: the md5 is the same at every K, on both prompts and after a 20,131-token prompt (35.9 / 38.4 tok/s at K = 0 / 2). Gate MC as you wrote it for the 4080 would pass on Metal at K = 1 and 2. mlx-vlm's own MTP reached 51.8 / 43.1 at K = 2.

MB holds by construction, which I checked on P3. A 4-row forward equals four 1-row forwards bit for bit after 20 to 2,400 tokens. That needed one fix beyond the safe-math matvecs: the decode attention's split count and kernel were picked from the longest row of the forward. Now each row picks its own (Q4AttArgs smin, skeys). Before the fix, 4 rows straddling 512 keys or a 128-key boundary changed about 240,000 logits a row.

Design:
- As yours: the head one row behind, keys only (no q projection on Metal, whose projections are separate).
- Drafts are one command buffer, with k_argmax feeding the next row.
- Snapshot buffers a layer. q4e_accept swaps the kept row's buffers in, so nothing is copied.

Rebase: I have it rebased locally on your 6b9b242 with your 7cfd557 semantics (ndraft = Q4EOpts.mtp; that many snapshot buffers). test_qwen4_mtp passes on Metal, including the one-draft case. I'll push it when main moves.

Profile (Q4E_PROFILE=2, code, ms per round): K=0 21.4; K=2 47.2 (3 verify rows + 2 drafts).
- Experts: 7.6 -> 19.3
- DeltaNet: 5.5 -> 11.1
- Hyper-connections: 3.6 -> 6.4
- Draft head: about 2 ms a draft

On P3 only the routed experts are seeds. Every dense matrix is affine Q4 g32, and a microbenchmark shows why each verify row costs so much: my dense Q4 matvec scales almost linearly in T (DeltaNet qkv 36 / 63 / 87 / 115 us at T = 1..4). Its multi-row path is ALU-bound.

Next, in my kernel file: a simdgroup-MMA matvec for T = 1..8 (8 weight rows x 8 tokens a simdgroup, the weights dequantized once). Each output's sum is then the same for any T, so MB keeps holding. I'm microbenchmarking it before touching the engine. Your distinct-expert pass comes after that: on Metal it saves about 13% of the expert stage at K = 2, and the dense matvecs are the bigger lever.

Also found: the LFSR stream tables are linear over GF(2), G[s] = G[s & 255] ^ G[s & 0xFF00] for every s (both tables). A seed kernel can keep two 256-entry halves in threadgroup memory instead of a random lookup into 256 KB. On Metal: +2% P3 decode, bit for bit. On CUDA it could go in shared memory too.
