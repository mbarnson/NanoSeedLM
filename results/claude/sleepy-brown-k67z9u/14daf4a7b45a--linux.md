Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 14daf4a7b45af54c1bc4b4550a6b806701dbef68
Platform: linux
Agent: cloud
Time: 2026-11-06T00:38:44Z

HF Jobs rtx-pro-6000 (job txgsync/6aed2102c78863f9c633bd76), CUDA 12.8, Clang, Ninja Release: ctest 44/44 PASS. test_qwen4_engine: trellis experts (L 12, K 3) give logits 8.1e-03 of max |logit| from the C reference. The argmax differs in 0 of 28 rows (a 20-row prompt through the GEMM, 8 decode rows through the gathers), and the cache changes nothing.
