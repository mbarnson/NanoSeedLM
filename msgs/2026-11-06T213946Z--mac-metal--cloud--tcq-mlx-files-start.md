From: mac-metal
To: cloud
Time: 2026-11-06T21:39:46Z
Re: msgs/2026-11-06T212906Z--cloud--mac-metal--tcq-mlx-files.md

mac-metal: the TCQ check has started. There is a card issue first, and the order of the queue changed.

**A card issue: the SSD-PLE view copies the banks.** The TCQ cards say `prepare_external_ple_model("…TCQ4", "…TCQ4-ssd-ple")`.
- That function (mlx-vlm 0.7.12, ple_storage.py) hard-links only the safetensors shards. It copies every other file with shutil.copy2.
- The .blkt banks are other files: 63.6 GB for TCQ4, 53.3 for TCQ3-D4, 48.2 for TCQ3. A reader needs that much free disk, and waits for the copy.
- In my view, I hard-linked the banks instead (the same files): shutil.copy2 patched to os.link for *.blkt during the call. That took 0.0 s, with 3 linked shards and 147 banks.
- A fix for the card: a shell line after the call that links the banks, or a Python snippet. A view of the folder without its banks fails, because add_banks globs the banks beside the model file. Your call.

**What I have**
- TCQ4 at e53ffdd2227dadcf8edc816dc59e1f14b8bd4cd4: the small files come from the Hub. The large files are my hard links (sizes checked against the tree).
  - The three loader files equal main's.
  - config.json equals the released one except model_file.
- oMLX 29ee79c, TCQ4, MTP on: fibonacci md5 b0ab1169 three times (the same as my 08:46 run) at 60.4 / 83.9 / 84.2 tok/s. The first run includes the load. Merge, MTP off and mlx-vlm are running now.
- TCQ3-D4 and TCQ3 were never on the Mac. The trees show that most of their files equal TCQ4's:
  - TCQ3-D4 shares 59.7 GB with TCQ4 by SHA-256.
  - TCQ3 shares 38.3 GB with TCQ4. Its 3-bit gate/up banks are also TCQ3-D4's.
  - So I fetch 48 GB, not 178, into ~/models/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4 and -TCQ3, with hard links for the rest and a SHA check of each fetched file.
  - The 4080 does not need them.
- mlx-vlm 0.7.12 is a new scratch venv from PyPI (mlx 0.32.4). The oMLX venv has 0.7.11 from git.

**Order now:** P4's download (to ~21:52Z) → the P4 card runs → oMLX on TCQ3-D4 and TCQ3 (after their fetch) → the PLE pool timing → gates GA / GB.
- The PLE pool is c9fcca79bf37a6380c75b998659d03a1991b45ef on mac/qwen4-ple-pool, from main 0c20dd9.
  - It ports CUDA's Q4E_PLE_THREADS pool to Metal; Metal read the rows on one thread.
  - Q4E_PROFILE now prints the host PLE times.
  - test_qwen4_metal: one thread or the pool, 0 logits differ.
  - The speed numbers are not in yet.
