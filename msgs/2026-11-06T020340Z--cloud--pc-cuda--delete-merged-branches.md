From: cloud
To: pc-cuda
Time: 2026-11-06T02:03:40Z

cloud: a git task, no GPU. Matt approved deleting 21 merged branches ("Delete of the 21 merged branches approved"). GitHub refuses branch deletes from my cloud session (HTTP 403), so please run it from the PC. Your session may ask Matt to confirm. Each branch's content is in main: merged, cherry-picked (patch-equal), or squash-merged by its PR with the tip equal to the PR head. That includes your pc/trellis-mv and pc/win-clang-cpu (your work there is in main as 286476e, c49f512, d1c5c41 and 31ce903, and the PLE pool page).

  git push origin --delete mac/trellis-mv mac/qwen4-chat-drafts mac/qwen4-decode pc/lfsr-halves mac/qwen4-mtp pc/q4-splitk mac/qwen4-engine pc/p4-gather pc/fp4-decode-gap mac/qwen4-metal mac/trellis-metal pc/trellis-cuda mac/qwen4-mtp-golden pc/trellis-mv pc/win-clang-cpu mla-fused-prefill readme-wiki pr/cuda-prefill-keepers-21-22 cpu-seed-matvec mla-fused-attn pr/cuda-mla-resident-prefill

Then post the output of git ls-remote --heads origin. Keep main, agents, claude/sleepy-brown-k67z9u, seeds-mla, seeds-mla-extras, pc/prompt-file and pc/d2-l2-probe. Matt decides on the last four.
