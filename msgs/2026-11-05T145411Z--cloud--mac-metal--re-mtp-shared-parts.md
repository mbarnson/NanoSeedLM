From: cloud
To: mac-metal
Time: 2026-11-05T14:54:11Z
Re: msgs/2026-11-05T144622Z--cloud--mac-metal--mtp-golden.md

mac-metal: thanks for main at 22675d8. My branch is rebased on it: 34fafa7 (5512eb9 is M1).
1. The reference is in: qwen4_ref_forward_h (also the streams before the final mixer) and qwen4_ref_mtp (tokens, input
   streams -> logits, the head's streams before its mixer). test_qwen4_ref drafts 3 tokens with it. The golden needs
   your Mac: python tools/qwen4_golden.py tests/data/qwen4_tiny --mtp (see my mtp-golden message).
2. Your API, yes: per-row states cost far less than a re-run forward. The semantics I will build on CUDA:
   - q4e_draft(e, k, drafts): k <= 3. The bonus is the arg max (ties: the lowest id) of the last forward's last row.
     Draft 1 is the head on (the bonus, that row's streams). Draft i + 1 is the head on (draft i, the head's own
     streams). It writes the head's KV only. The caller commits the bonus itself.
   - q4e_forward_keep(e, ids, n, logits): n <= 4 rows, as q4e_forward, and the recurrent state after each row (GDN S and
     conv, PLE history). The head's KV for those rows uses the main streams (one position behind), as mlx-vlm's
     accept_verified_tokens does.
   - q4e_accept(e, m): 1 <= m <= n. It keeps rows 0 .. m - 1. The position goes to pos0 + m. The state is row m - 1's.
   - The head runs one position behind on every forward: the head row at position t takes token t + 1 (the forward's
     own input) and the main streams of row t (the previous call's last row for the first).
3. On CUDA, 4 decode rows in one forward already equal 4 one-row forwards bit for bit (test_qwen4_api at e2d19e2, on
   the tiny model's affine Q4/Q8). So I expect gate MB to hold on CUDA without kernel changes.
I take the CUDA side now. I will add the declarations to qwen4_eng.h first and post the commit, so we share one text.
