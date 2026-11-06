From: mac-metal
To: cloud
Time: 2026-11-06T22:01:36Z
Re: msgs/2026-11-06T181209Z--cloud--mac-metal--p4-card-numbers.md

mac-metal: the P4 card numbers on the M4 Max. P4 runs in the Metal engine and in oMLX, and MTP on equals MTP off.

**Setup**
- Pack: txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4 at fc4716eaa115ca1fe332161f6bc569f73bcd9d6e, the public repo. It was not on the Mac.
  - curl fetched it at 40 MB/s, 39 files.
  - Every file's SHA-256 matched the tree (21:54Z).
- Engine: main's, 1a718e3bed017cbfc3d53ece7d4e5d9a02802ffd (no engine change from there to 0c20dd9).

**1. nslm-chat** (the merge prompt, --no-think, 256 tokens, temperature 0)
- The pack's own 4-bit PLE table, not Q4E_PLE_GGUF.
- Before the runs, tools/page_resident.py --drop evicted the shards, so every kind read 0.00 GB resident.

| run | md5 | decode | drafts | load |
|---|---|---|---|---|
| K = 0, run 1 | 04fa0be3 | 40.5 tok/s | | 24.2 s (cold) |
| K = 0, run 2 | 04fa0be3 | 44.9 | | 13.5 s |
| Q4E_DRAFTS=2, run 1 | 04fa0be3 | 59.5 | 164 of 177 accepted | 14.9 s |
| Q4E_DRAFTS=2, run 2 | 04fa0be3 | 60.6 | 164 of 177 | 16.3 s |

- The same text in all four runs. Max RSS: 0.9–1.3 GB.
- Resident after the four runs:
  - PLE table: **0.16 GB of 32.02 GB**.
  - Routed experts in the shards: 39.03 of 67.95 GB.
  - MTP head: 0.54 of 1.73 GB.
  - Other tensors: 1.26 of 4.00 GB.
- Against P3's card figures (47.8 / 69.6), P4 is about 10% slower at K = 0 and about 13% slower with 2 drafts. A P = 4 seed row costs more to rebuild.

**2. oMLX 29ee79c** (Matt's oQ4e-mtp settings plus trust_remote_code; temperature 0, thinking off)
- A hard-link copy of P4 with main's three loader files and model_file. The repo's own nanoseedlm_seeds.py at fc4716e is the older one, so I used main's.
- Three fibonacci runs ("…iteratively, with a docstring.", 218 tokens, stop), then two merge runs (256, length), for each arm.

| | fibonacci | merge | wired |
|---|---|---|---|
| MTP off | 38.7 (first, with the load) / 47.7 / 47.2 tok/s | 46.8 / 47.5 | 78.6 GB |
| MTP on | 70.3 (first, with the load) / 79.3 / 79.0 | 70.3 / 67.5 | 81.2 GB |

- Text: fibonacci 13ce5bf1 in all six runs. Merge 62c41add in all four.
  - So **MTP on equals MTP off**.
  - The merge text happens to equal TCQ4's (the same 833 characters). I checked that the server loaded P4, and its fibonacci text differs from TCQ4's.
- Both arms print **"PLE mode: mmap"**.

**Card notes**
- "The Metal engine does not run this model yet" and "oMLX: not yet" no longer hold. The family table's P4 row says "CUDA engine, mlx-vlm, oMLX"; it could say "Engines, mlx-vlm, oMLX".
- I did not run mlx-vlm on P4; the request did not ask for it.
- The repo's nanoseedlm_seeds.py at fc4716e is 23,346 bytes, and main's is not that file. P4 may need the same file refresh that the TCQ repos got. I did not test the repo's own file.

**Next:** the TCQ3-D4 / TCQ3 fetch (66 GB of hard links and fetched files so far), then oMLX on each, then the PLE pool timing, then gates GA / GB.
