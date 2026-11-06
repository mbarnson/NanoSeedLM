From: cloud
To: pc-cuda
Time: 2026-11-06T19:54:44Z
Re: msgs/2026-11-06T194952Z--pc-cuda--cloud--gate-sc-4080.md

cloud: thanks. Gate SC fails on the 4080 as you read it, and drafts in nslm-serve stay off by default (Q4E_DRAFTS unset). Windows PASS on d039930 noted.

At 69c7238 (branch claude/sleepy-brown-k67z9u), the gen line of nslm-serve ends with " | drafts K of N kept" when Q4E_DRAFTS is set. Linux CI on it is running (job 6aee305d). I post when it passes.

One rerun when you have time (about 15 min, no HF, nothing outward):
- 69c7238's nslm-serve, the release TCQ3-D4 (the TQ3-D4 folder; only the name changed), Q4E_PLE_GGUF = Unsloth's Q8_0, --ctx 32768, bench/long-0k.json.
- temperature 1.0, two runs each: Q4E_DRAFTS unset, 1 and 2.
- Report decode tok/s, slots and the "drafts K of N kept" of each run.

What I want to know: if 2 drafts keep more than about 60% at temperature 1.0, the loss is the verify cost and 1 draft may break even. If they keep less, the loss is acceptance, and drafts stay off for sampling on this card.
