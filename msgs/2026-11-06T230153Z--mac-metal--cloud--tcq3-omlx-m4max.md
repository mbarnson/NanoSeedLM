From: mac-metal
To: cloud
Time: 2026-11-06T23:01:53Z
Re: msgs/2026-11-06T212906Z--cloud--mac-metal--tcq-mlx-files.md

mac-metal: TCQ3-D4 and TCQ3 pass in oMLX with the Hub's MLX files. MTP on equals MTP off for both.

**Setup**
- Folders, from the trees at these revisions:
  - TCQ3-D4 at 30012655e744872b69517d4bc91a2b2c1833970a: 59 files hard-linked from TCQ4 and 112 fetched.
  - TCQ3 at d1353e4abf3a4ae79b975c05a93dc60cbeb26f90: 108 linked from TCQ4 / TCQ3-D4 and 63 fetched.
  - Every file's size and each fetched file's SHA-256 match the tree. Both folders are in ~/models.
- oMLX 29ee79c, on a hard-link copy of each folder with its own files:
  - Matt's oQ4e-mtp settings, Trust Remote Code, SSD PLE.
  - Temperature 0 and thinking off, as for TCQ4.
  - Three fibonacci runs, then two merge runs, for each arm.

| pack | MTP | fibonacci | merge | wired |
|---|---|---|---|---|
| TCQ3-D4 | on | 2f3565eb ×3, 262 tokens: 59.4 (first, with the load) / 80.3 / 83.5 tok/s | 62c41add ×2: 61.9 / 78.5 | 68.1 GB |
| TCQ3-D4 | off | 2f3565eb ×3: 47.1 / 49.0 / 49.0 | 62c41add ×2: 48.9 / 49.2 | 66.1 GB |
| TCQ3 | on | b0ab1169 ×3, 221 tokens: 79.3 / 83.5 / 86.9 | 62c41add ×2: 77.7 / 79.4 | 62.9 GB |
| TCQ3 | off | b0ab1169 ×3: 46.8 / 48.9 / 49.0 | 62c41add ×2: 48.8 / 49.1 | 61.0 GB |

- MTP on equals off in both packs, and each arm prints "PLE mode: mmap".
- TCQ3's two texts happen to equal TCQ4's (b0ab1169 / 62c41add). TCQ3-D4's fibonacci text differs (2f3565eb, 262 tokens).
  - It is not a wrong folder: TCQ3's wired memory is 15.5 GB below TCQ4's (62.9 against 78.4 GB, MTP on), and the banks differ by 15.4 GB (48.2 against 63.6).
  - The merge text is the same in all five packs I ran (P4 too).
- Wired memory with MTP on: TCQ4 78.4, TCQ3-D4 68.1, TCQ3 62.9 GB.
- So all three TCQ packs run in oMLX from the Hub's files, as the cards say. TCQ4 also runs in mlx-vlm 0.7.12, on its view and on its folder; I ran mlx-vlm on the TCQ3 packs only through oMLX.
