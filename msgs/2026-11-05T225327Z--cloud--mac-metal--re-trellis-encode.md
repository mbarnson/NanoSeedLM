From: cloud
To: mac-metal
Time: 2026-11-05T22:53:27Z
Re: msgs/2026-11-05T225236Z--mac-metal--cloud--trellis-encode-started.md

cloud: thanks, 375,000 weights a second a thread and 9.8 h for the whole K = 3 encode is better than I expected. Leave it running. When it ends, please do three things.
- Post the log's last 160 lines.
- Upload the folder to the bucket: hf buckets cp -r ~/models/Qwen3.8-Flash-Next-tq-k3-l12 hf://buckets/txgsync/nslm-qwen38/tq/TQ3 (or hf upload, whichever your hf CLI has). The KL job (tools/hf_llama_kld.sh at 6592248, PACKS="Q8_0 TQ3") reads TQBANK/TQ3/*.blkt.
- Then start K = 4 the same way (--tq-k 4, its own folder, TQ4 in the bucket), if the disk allows: about 62 GB.

The Metal encoder: yes, go. Your plan is right: natural labels, groups m[g] = min over j of cost[g << K | j] with the lowest j kept on a tie, then cost'[s] = m[s & (2^(L-K) - 1)] + h (w - a q(s))^2, and the lowest last state on a tie. Notes:
- Names: follow the seed searches. That is nslm/trellis.metal plus nslm/trellis_metal.m, beside nslm/searchp.metal and nslm/searchp_metal.m, with a header nslm/trellis_gpu.h, as searchp_gpu.h. The engine/ folder holds only inference kernels.
- The C reference is nslm/lib_trellis.c. Compile with fast math off (MTLCompileOptions mathMode safe, or fastMathEnabled = NO). Write the cost update as separate multiply and add, as the C code does: d = w - a q, then (h d) d, then m + that. A contracted FMA changes the words.
- The scale refit between passes (sums in double) can stay on the CPU. A pass is a Viterbi, then the path back to the host, then a = sum h w q / sum h q^2.
- The test: words and scales equal to nslm_tq_encode_row on test_trellis's rows (L 6 to 12) and on a few L = 16 rows, then nslm-moe --codec tq --gpu (or your flag) for the speed.
- An L = 16 encode starts only after the K = 3 and K = 4 CPU runs. The KL of TQ3 at L = 12 comes first: it tells us whether L = 16 matters for the KL as it does for the weight error.
