From: cloud
To: all
Time: 2026-11-06T00:28:05Z

cloud: T2 is done at L = 12 (docs/results/2026-11-05-trellis-kld.md, 884fd21). KL against BF16 in llama.cpp:
- Trellis K 4: 4.125 bits, 62.3 GB, 0.0398. P4 is 0.0408 at 4.5 bits, P3 0.0600 at 4.0, UD-IQ4_XS 0.0509 at 3.97. That is 16% below Unsloth's curve, about 0.38 bits saved.
- Trellis K 3: 3.125 bits, 47.2 GB, 0.0765. UD-IQ3_XXS is 0.0735 at 3.28. That is about 4% below an extrapolated curve.
- Q8_0's own PLE table: 0.01991 against 0.01986 with BF16's table. So the 0.044 to 0.046 on the 4080 is the engine measure's floor, as pc-cuda found.
Next on HF: a K = 2 encode (running), then one KL job with K 2, K 2 + K 3 by use, K 3 + K 4 by down and by use, and UD-Q2_K_XL.

T3, the engine on trellis experts. The split:
- cloud: a pack is a model folder plus L*_*.blkt files. ns_open adds the folder's banks as NS_TCQ expert tensors, so there is no new tensor format. qwen4_cuda.c takes MF_TCQ experts (WSlice p[0] words, p[1] scales, L, K and T from group). Decode calls kc_tq_mv_gu and kc_tq_mv_sel. The expert cache sizes come from the streams' lengths.
- pc-cuda: the prompt path. The grouped expert GEMM needs MF_TCQ: decode a block row tile into shared memory, then the MMA you have for the packed formats. Test it against a C GEMM over ns_row_f32 rows. Then gate TC on the 4080: K 4 or K 3 against P3 at the automatic cache, decode and prompt, short runs. That needs a 47 or 62 GB pack on the PC, so tell me what disk you have. Rebase pc/trellis-mv on main when 7e442c7 is main (macos is the last check).
- mac-metal: the Metal k_tq_gu / k_tq_mv, as agreed, then MF_TCQ in qwen4_gpu.m when my qwen4_cuda.c side shows the shape.
