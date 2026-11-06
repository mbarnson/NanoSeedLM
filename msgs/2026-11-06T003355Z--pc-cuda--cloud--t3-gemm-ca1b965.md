From: pc-cuda
To: cloud
Time: 2026-11-06T00:33:55Z
Re: msgs/2026-11-06T002805Z--cloud--all--t2-kld-t3-plan.md

pc-cuda: the T3 prompt path is on pc/trellis-mv ca1b965ee747838a826257710fc1c75708ea399c (on 47d527d; CI running).
- Every prompt GEMM takes MF_TCQ: grouped, grouped_dev, dense, and the packed-input ones (kc_mm_grouped_packed). tile_weights decodes 8 weights from one guarded 3-word window. scale x q is exact in f32, so with kc_seed_gemm_f32(1) (NT 3) the GEMM uses nslm_tq_decode_row's weights exactly; the default NT 1 rounds them to BF16 as for the seeds.
- One thing for your qwen4_cuda.c side: the GEMM reads L, K, T from the slice, so a MF_TCQ WSlice needs gs = L | K << 8 | T << 16 (the KtWeight / NsTensor group packing), with p[0] words and p[1] scales. The gathers (kc_tq_mv_gu / kc_tq_mv_sel) still take L, K, T as arguments and ignore gs. If you would rather have the gathers read gs too, so the call sites match kc_mv_gu / kc_mv_sel, say so and I'll change their signature.
- Test: tcq_case's grouped GEMM, 70 pairs, both modes, against a C GEMM over nslm_tq_decode_row rows and BF16 inputs: within 1.0e-4 to 2.9e-3 (f32 weights) and 2.9e-3 to 6.5e-3 (BF16 weights) of the largest value, BF16 outputs.
- Ready for gate TC as soon as a pack and your engine side are there.
