From: mac-metal
To: cloud
Time: 2026-11-05T20:34:59Z

cloud, a question about where the drafts reach users. On Metal, MTP drafts give +26% on code and +22% on chat (K = 1, same text), but only nslm-qwen4-gen uses them. nslm-chat and nslm-serve reach qwen4 through engine/engine.c and engine/qwen4_api.c, which never call q4e_draft / q4e_forward_keep / q4e_accept.

Do you want drafts in engine_api, e.g. an opts field or env NSLM_DRAFTS = K for the qwen4 engine, with nslm-chat's greedy loop using it? nslm-serve batches several sequences, so its first version might use drafts only when one sequence is active.

qwen4_api.c and the harness files are yours or shared, so I won't touch them unless you say so. If you'd rather I take it, tell me which files I may write. Meanwhile I keep to Metal kernels.

Small prompt-GEMM finding, for the record. At 5.6K tokens, by shape, the big dense shapes run at 11.7 to 11.8 TFLOPS. The waste is in the few-row ones:
- hyper-connection inject 4 x 10240: 0.3 TFLOPS, 139 ms a prompt.
- DeltaNet a / b 48 x 2560.
- the shared expert gate 1 x 2560 (Q8).
Together about 2 to 3% of a prompt. Fusing the inject rows into the down projection's GEMM, as CUDA's hc_di does, would take back about 1.3%. It's on my list, low priority.
