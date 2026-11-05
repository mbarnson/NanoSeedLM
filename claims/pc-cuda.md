# pc-cuda claims (updated 2026-11-05T20:57Z)

- **Back** (Matt gave the PC back; this is the Claude Code CLI session). Working cloud's 20:55Z queue in order:
  1. windows on claude/sleepy-brown-k67z9u cb3d6b6 (cl and clang-cl), in G:\devel\NanoSeedLM-lfsr.
  2. D4 decode on the 4080 (clang-cl, P3 pinned, auto cache, code prompt, 256 tokens): --cpu-tail 0 1 2 0 1 2, Q4E_PROFILE=1 at 1 and 2, and at 1 with Q4E_PLE_THREADS=9.
- No code edits planned.
