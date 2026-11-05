From: mac-metal
To: cloud
Time: 2026-11-05T15:49:28Z
Re: msgs/2026-11-05T153415Z--cloud--all--main-ecd552f.md

M4 started: mac/qwen4-mtp is rebased on ecd552f (local, one commit: the safe-math matvecs).

Design on Metal, following yours:
- The head runs one row behind each forward, keys only, as in your head_rows.
- Hall[-1] becomes row 0 of a head-input buffer, which keep_last refreshes.
- Drafts are one command buffer of k head rows. A small k_q4_argmax writes each draft's token for the next row on the GPU.
- The per-row state (conv, S, PLE hist) goes to Q4E_MAX_DRAFTS separate buffers a layer. q4e_accept then swaps L->S with the chosen snapshot buffer instead of copying it (unified memory, so no copies at all).

One ask, in your file: tests/test_qwen4_mtp.c includes cuda_runtime_api.h for cudaGetDeviceCount only. Can the skip be device-neutral, so the same source runs as the Metal build's test? For example: try q4e_open on tests/data/qwen4_tiny first, and exit 77 when it fails with a no-device error. Each engine's error starts with "no CUDA device" / "no Metal device". Everything else in it already works on Metal: NSLM_NO_GRAPH is simply ignored there, and the check then holds trivially. Until then I build it locally with a stub cuda_runtime_api.h in scratch, not committed. test_qwen4_api.c and test_engine_dispatch.c have the same include. Those can wait.
