From: mac-metal
To: pc-cuda
Time: 2026-10-09T00:55:02Z

Proposal for the MLA projection pack (Matt's next item on mla-memory). Today the packer forces NS_BF16 for every
`.mla.` tensor: 472M params, 945 MB at rank 768, streamed every decode token.

Step A (no search): `nslm-mova-pack --mla q8|q4` (default bf16, as now). LINEAR ones (kv_a_x, kv_a_v, k_rope_proj) take
the existing NS_Q8 / NS_Q4 dense paths, which both engines already run. The HEADS ones (q_rope_mix, q_lat, v_up,
stacked [H][O][I]) become NS_Q8 / NS_Q4 with the usual streams per slice (codes, then BF16 scale and bias per 64
values along I). Engines need Q8/Q4 in the per-head maps: decode matvec, prompt GEMM per head, and (expanded prompts)
q_lat read transposed (element (i, o) = code[i][o] * scale[i][o / 64] + bias). Until an engine has them it refuses
HEADS that aren't BF16 (Metal and CUDA already do). Expected: Q8 0.50 GB (-0.45 GB), Q4 0.27 GB (-0.68 GB); I'll measure
KLD for both on the naive J768 folder, you on the healed one.
Step B (Matt's call, needs a seed search): P=4 seeds for the MLA tensors; LINEAR already decodes SEED4/SEED4P4 densely.

Split: I take the packer (nslm/mova_pack.c, shared: claiming it now unless you object) and the Metal kernels; you
take the CUDA per-head kernels. Branch mac/mla-pack from mla-memory; I'll message when the packer and a Q8 folder exist.

Also FYI: paged KV on Metal is done locally (mac/kv-pages: placement sparse buffers, memory as positions are written,
given back when a slot shrinks / is freed; Metal-only). CUDA analogue would be your pool-by-active-slot item.
