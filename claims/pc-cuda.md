# pc-cuda claims (updated 2026-11-06T22:15Z)

- **Working on (2026-11-06T22:23Z):** cloud's v011-correctness: windows on b911cb5, then the new start-here.ps1 of bd6cbc7 in a copy of v010-test4 (correctness only, no speeds).
- **Queued for the next pc-cuda session** (cloud, msgs/2026-11-06T220625Z--cloud--pc-cuda--v011-windows-check.md), after cloud's "v0.1.1 draft" message:
  - The Windows release check of the v0.1.1 zip, as for v0.1.0 (msgs/2026-11-06T203030Z--pc-cuda--cloud--release-check-6becc62c.md): SHA-256; START-HERE.bat TCQ3-D4 <models> with the new pinned revisions (small files download, banks checked); chat; run-TCQ3-D4.bat; one nslm-serve request.
  - windows PASS/FAIL on the final main SHA that cloud names.
- **For that session:**
  - CI: G:/devel/ci-one.sh REV (reconfigures, then cl and clang-cl ctest) in G:/devel/NanoSeedLM-lfsr.
  - Packs: TCQ3-D4 in G:/devel/v010-test/nslm-v0.1.0-windows-x64-cuda12.8/models (with the Q8_0 table hard-linked in Qwen3.8-Flash-Next-GGUF/engram); TCQ3 in G:/devel/rel/TQ3; TCQ4 in G:/devel/rel/Qwen3.8-Flash-Next-NanoSeedLM-TQ4.
  - Pass start-script answers as arguments (TCQ3-D4 <folder>), never through stdin.
  - Stop processes by PID.
  - See msgs/...--pc-cuda--all--handoff-and-feedback.md.
- **Rules:** no run over about an hour while Matt wants the PC; nothing on HF without a message to cloud; outward actions as ask-* messages to matt.
