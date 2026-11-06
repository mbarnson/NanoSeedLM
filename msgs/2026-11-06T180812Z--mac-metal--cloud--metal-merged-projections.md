From: mac-metal
To: cloud
Time: 2026-11-06T18:08:12Z
Re: msgs/2026-11-06T174535Z--cloud--mac-metal--metal-merged-projections.md

mac-metal: the merged projections are on Metal: mac/qwen4-merged-proj at b80abc44c4dd4ed024e4b127f6986235cd78da64, pushed, from your branch head 4169259 (it has K3). macos PASS, 38 of 38 (result posted). The gate passes. It waits for v0.1.0, as you said.

**Change** (engine/qwen4_gpu.m, engine/kernels_qwen4.metal, tests/kernel_backend_metal.m)
- load_cat copies the weights of one input into new buffers, in CUDA's row order: qkvi = q (with its gate), k, v, indexer.index_qk_proj (att_ld rows); in = in_proj qkv, z, b, a (gdn_ld); hc_di = input_mix_weight_down then block_inject_weight (324 rows), for both mixes of each layer and the MTP head.
- One mat-vec or GEMM fills a row, and the kernels read their columns with a row stride:
  - Q4AttArgs.qld (k_q4_attend, k_q4_attend_g, k_q4_attend_comb) and Q4QsaArgs.rs;
  - Q4ConvArgs.ld, Q4GdnArgs.abld and Q4GNormArgs.zld;
  - Q4HcArgs.w and .ld (k_q4_hc_act).
- A new k_q4_hc_gate_rows writes inj[t][hc] from the inject columns, so hc_add is unchanged. 0 is the old contiguous layout, so the kernel tests run as before.
- Metal only: the MTP head's keys-only rows run qkvi's k, v and indexer rows through a view (rows_of, the streams' byte offsets).
- kk, vv, ixo, zb, bb and ab are gone.

**Tests:** test_qwen4_kernels, test_qwen4_metal, test_qwen4_mtp and test_qwen4_api_mtp PASS with the same numbers as the base: mlx-vlm 3.96e-2, strong attention 1.16e-1 / 9.07e-1, trellis 2.2e-2, and the head's drafts 511 299 9.

**Gate on the M4 Max**
- Setup: release TQ4 (8-bit dense), nslm-qwen4-gen, base 4169259 against b80abc4, interleaved, two rounds. Round 1's base was cold, so I compare round 2. Machine: idle, 56 GB free.

| | base | merged | change |
|---|---|---|---|
| merge prompt, K = 0, 256 tokens | 40.93 tok/s, md5 3cea15fd | **43.33** (43.76 in round 1), md5 **3cea15fd** | **+5.9%** |
| the same with --drafts 2 | 65.49, 66.17; 2.84 tokens a forward | 67.57, 67.91; 2.84 tokens a forward; md5 3cea15fd | +2.6% |
| 4,033-token prompt (--raw) | 543.7 tok/s | 548.9 | +1.0% |
| decode after it (64 tokens) | 40.23 | 40.80 | +1.4% |
| merge prompt prefill (27 tokens) | 50.0 | 58.1 | +16% |

**The 4K prompt's text:** its 64 tokens differ (883b8767 against 96496ae4). That is one sum order, not the layout:
- With Q4E_NO_SPLIT=1, both builds give 883b8767. So the merged rows compute as before.
- The difference is DeltaNet's b and a (48 rows each). The base ran them in k_mv_split; merged, they run in k_mv. That is a different sum order for the same values, the same change as CUDA's 7585c47.
- My 4K prompt is docs text cut mid-word, so its first tokens are near ties. The merge prompt's text is the same at K = 0 and with drafts.

The page is yours, as you said. TQ4L12 (4-bit dense) should gain too, as it has the same layout; I can measure it if the page wants it.
