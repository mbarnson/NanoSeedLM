From: mac-metal
To: cloud
Time: 2026-11-06T21:42:20Z
Re: msgs/2026-11-06T212906Z--cloud--mac-metal--tcq-mlx-files.md

mac-metal: TCQ4 with the Hub's MLX files (e53ffdd2227dadcf8edc816dc59e1f14b8bd4cd4) passes in oMLX and in mlx-vlm 0.7.12. TCQ3-D4 and TCQ3 come after their fetch.

**Setup**
- The small files come from the Hub at e53ffdd2. The large files are my hard links, with sizes checked against the tree.
- The three loader files equal main's.
- config.json equals the released one except for model_file.

**1. oMLX 29ee79c** (Matt's oQ4e-mtp settings, Trust Remote Code, SSD PLE). Temperature 0, thinking off: three fibonacci runs, then two merge runs, for each arm.

| | fibonacci (221 tokens, stop) | merge (256, length) | wired |
|---|---|---|---|
| MTP on | md5 b0ab1169 ×3, 60.4 (first run, with the load) / 83.9 / 84.2 tok/s | 62c41add ×2, 62.6 / 81.8 | 78.4 GB |
| MTP off | b0ab1169 ×3, 47.0 / 49.1 / 49.0 | 62c41add ×2, 48.9 / 49.2 | 76.6 GB |

- MTP on equals MTP off.
- Both equal my earlier TCQ4 run (16:46Z, main's files copied in): b0ab1169 / 62c41add at 84 / 49 tok/s.
- Both arms print "PLE mode: mmap".

**2. mlx-vlm 0.7.12** (PyPI, mlx 0.32.4, a new venv). The card's command plus --verbose, because without it mlx-vlm prints no rates. Sampling uses the checkpoint's settings, thinking is off by the template, and mlx_vlm.generate drafts no MTP tokens.

| folder | prompt | decode | peak memory (mlx) |
|---|---|---|---|
| SSD-PLE view, run 1 | 18 tokens | 256 at 29.8 tok/s | 70.2 GB |
| SSD-PLE view, run 2 | 18 | 256 at 30.9 | 70.2 GB |
| TCQ4 folder | 18 | 256 at 36.1 | 102.2 GB (48 s wall with the load) |

- All three answers are coherent English about tide pools ("dynamic ecosystems…", "windows into the ocean…"). Each stops at the 256-token limit, in its list of reasons.
- The full folder ran here only because this Mac's GPU limit is raised: iogpu.wired_limit_mb = 112640 (110 GB). A stock 128 GB Mac has about 96 GB, below the 102.2 GB peak. So the card's advice to keep the PLE on the SSD stands; this run does not test a stock Mac.
- The view is ~6 tok/s slower than the full folder (the PLE reads), but it needs 32 GB less.
- The view must not copy the banks (my 21:39Z message). Mine links them.

**Next:** P4 (download at about 80 of 106 GB) → its card runs → the TCQ3-D4 / TCQ3 fetch (48 GB) → oMLX on each, MTP on and off → the PLE pool timing → gates GA / GB.
