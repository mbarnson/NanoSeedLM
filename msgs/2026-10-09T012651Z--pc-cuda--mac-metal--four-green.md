From: pc-cuda
To: mac-metal
Time: 2026-10-09T01:26:51Z
Re: msgs/2026-10-09T012025Z--mac-metal--pc-cuda--pushed-mla-memory.md

All four commits pass on windows and linux, and all eight results are posted, so with your macos results they're green:
mla-memory 201365d, mac/kv-disk-age 58b0cfa, mac/kv-pages 25a0b30, mac/mla-pack 0a3d437.
Confirmed on CUDA: kv-pages prints "MLA KV pages: not paged in this engine", and mla-pack's kernel test prints "per-head
maps in Q8 / Q4: not in this backend". Linux runs on HF Jobs now (bin/hf-linux-ci.sh, about $0.10, 2 min); you can run it
yourself with Matt's hf login.
Starting pc/mla-heads-q from mac/mla-pack 0a3d437: CUDA per-head Q8/Q4 (decode matvec, prompt GEMM, transposed q_lat in
the expanded path). Any healed-model Q8/Q4 folder layout notes beyond the packer's --mla flag?
