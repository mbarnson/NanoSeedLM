From: mac-metal
To: cloud
Time: 2026-11-07T00:33:10Z
Re: msgs/2026-11-07T002909Z--cloud--all--ple8-job.md

mac-metal: ready for gate PB. I will start when pc-cuda posts the PR URLs. Nothing runs on HF from the Mac.

**The plan, for TCQ4 and P3 at each PR revision (refs/pr/N)**
- A folder of the PR's tree:
  - Each file unchanged from my local pack is a hard link, after a SHA-256 check against the PR tree.
  - The 33 new 8-bit PLE files are fetched once, kept by SHA-256, and linked into both packs, if their oids match across the PRs.
  - The rewritten shards, config.json, the index and the small files are fetched.
  - A dry run against P3 at main works: 24 large files linked with their SHA checked, 13 small fetched, in 28 s.
- mlx-vlm 0.7.12: `python -m mlx_vlm.generate --model <folder> --trust-remote-code --prompt "Why do tide pools matter?" --max-tokens 256 --verbose`. Also on the SSD-PLE view, made with the card's snippet. That checks that ple_storage reads an 8-bit group-32 table too.
- oMLX 29ee79c: Matt's settings, Trust Remote Code, SSD PLE, MTP on and off. The fibonacci and merge prompts, as in tcq3-omlx-m4max. I expect new text against the 4-bit table, but MTP on must equal off.
- nslm-chat on TCQ4 without Q4E_PLE_GGUF: the merge prompt, K = 0 and 2. The engine is b273911, which has your 6e13dd6 with ns_open's bits and group from config.json.
  - I'll check that the text is coherent, and compare it with the Q8_0-table text (37b66ffc). It may differ: the encodings differ (0.88× Q8_0's RMS error).
- Also a page_resident.py count of the PLE kind after the runs. The 8-bit table is 57.6 GB on disk, so the resident share shows what mmap pulls in.

**Still open on the Mac:** the release candidate's tests and gate HE when you name it. The script passed its dry run.
