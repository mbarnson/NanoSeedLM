# pc-cuda claims (updated 2026-11-06T01:00Z)

- **Online** (Claude Code CLI session). Matt gave the PC; no run over about an hour. HF: nothing without a message to cloud (the cap); I no longer start Linux CI.
- Now: gate TC on the 4080 (main 54a7130 binaries in G:\devel\bin-54a7130). TQ4L12 is on the PC, TQ3L12 downloading (bucket read only). Packs: P3 hard-linked plus the .blkt files (G:\devel\P3-TQ4L12, P3-TQ3L12). Decode at the automatic cache against P3, a prompt run each, a KL sanity run. Script: G:\devel\qwen4-runs\tc.sh.
- pc/trellis-mv: all of it is on main (cherry-picked). The branch stays only as history.
- Done this session: windows PASS on cb3d6b6, e816fe8, bc50f41, 7dd456a, 09adc5a, 41a8fe6, cc566be, 7e442c7, 54a7130; D4 on the 4080; T5 and its floor; T3 CUDA gathers and GEMM.
