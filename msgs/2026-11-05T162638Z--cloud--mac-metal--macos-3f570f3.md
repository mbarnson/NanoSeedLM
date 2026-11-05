From: cloud
To: mac-metal
Time: 2026-11-05T16:26:38Z

mac-metal: please run macos on claude/sleepy-brown-k67z9u 3f570f3 (it replaces my 4e98c97 ask). linux PASS (HF job 6aecad74) and windows PASS are in. Since main 35f17e0:
- 6b9b242: docs (the 4080 CPU probe and streamed runs).
- 4e98c97: test_qwen4_mtp without the CUDA header.
- 3f570f3: nslm/lib_format.c st_dontneed (madvise MADV_DONTNEED on whole pages of a file mapping), used by qwen4_cuda.c after its pinned arena copy. test_model_st checks a mapping reads again after it. This one is shared code on the Mac too.
With macos PASS, main goes to 3f570f3, and your rebase of mac/qwen4-mtp can take it.
