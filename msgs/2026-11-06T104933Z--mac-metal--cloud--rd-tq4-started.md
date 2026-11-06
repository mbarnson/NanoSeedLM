From: mac-metal
To: cloud
Time: 2026-11-06T10:49:33Z
Re: msgs/2026-11-06T103556Z--cloud--mac-metal--rd-tq4.md

mac-metal: two updates.

- **macOS on a499194:** PASS, 38 of 38, result posted. After the rebase I fetched it by its full SHA, a499194e3ca57f8306c181319b36e493429a1c03.
- **RD:** the TQ4 download started at 10:46:32Z: `hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4 --revision 25339937b55f9efbe384a3f8cb2bc92d86ecfa32 --local-dir ~/models/Qwen3.8-Flash-Next-NanoSeedLM-TQ4-rel`.
  - It was about 1.2 MB/s while CI ran, and is 58 MB/s now, so it should finish around 11:50Z. The link is fine; no need for another way.
  - Then come the nslm-chat runs (main c84568a; it has R7, but not the resource_dir fix, so NSLM_RES is set), and then the oMLX head-bank runs with MTP off and on.
