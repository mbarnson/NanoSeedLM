# The release packs from the Hub on the M4 Max (2026-11-06)

**Result.** The download of TQ4 from the Hub gives a folder that the Metal engine and oMLX run with the same text in
each run. `hf download` stalled on read timeouts on the Mac, and curl for each file finished the download at 119 MB/s.
TQ4 decodes at 38 to 42 tokens per second in `nslm-chat`, and at 67 to 69 with 2 drafts. In oMLX, it decodes at 49, and
at 81 to 84 with MTP from the trellis banks of the head. The v0.1.0 draft's macOS archive reaches a chat and a server in
one run of `start-here.sh` (gate RE).

**Status.** Current. Step R5 of the [release plan](../plans/2026-11-05-release-family.md), gates RD and RE on the Mac.
The PC's test is [the release packs from the Hub on the RTX 4080](2026-11-06-release-downloads-4080.md).

## Setup

- Repository (private at the test): `txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4` at `25339937`. 169 files, 101.9 GB.
- Mac: M4 Max, 128 GB. No other work ran. The page cache held the folder after the hash check.
- Engine: `main` at `17eeb5d` (the engine of `1424085`), with `NSLM_RES` set.
- Runs: `nslm-chat --model DIR --no-think --max-tokens N "PROMPT"`, two runs at K = 0 and two with `Q4E_DRAFTS=2`.
- oMLX: Matt's fork at `29ee79c` (the fix for remote code), `--headless`, one server for each setting. The settings
  of oQ4e-mtp, with `trust_remote_code`, the PLE table on the disk and MTP of adaptive depth 5. Thinking off,
  temperature 0.
- The oMLX folder is a copy of hard links. Its `config.json` has `"model_file": "nanoseedlm_qwen4.py"`, and the three
  loaders of `17eeb5d` are beside it. The release folder does not change.
- Archive: `nslm-v0.1.0-macos-arm64-metal.tar.gz` of the draft (SHA-256 `16a6574a`, built from `1424085`), in a new
  folder. `./start-here.sh TQ4 <folder>`, with a folder of hard links to the checked download.

## Results

Download:

| Client | Files | Data | Time | Rate |
|---|---|---|---|---|
| `hf download`, with and without Xet | 124 | 53.2 GB | about 5.7 hours, with restarts | stalls, then about 1 GB in 15 to 40 minutes |
| curl, one for each missing file, 4 at a time, as `start-here.sh` | 45 | 48.7 GB | less than 8 minutes | 119 MB/s |

Check: 157 files match their SHA-256 on the Hub, and the 12 small files match by size. No file is bad.

`nslm-chat`:

| Prompt | K = 0 (run 1, run 2) | 2 drafts (run 1, run 2) | Text md5 (4 runs) |
|---|---|---|---|
| The code prompt of `hf_release_pack.sh` (a Fibonacci function), 128 tokens | 39.2, 41.5 | 67.1, 67.9 (2.82 tokens a forward, 83 of 87 drafts accepted) | `b12f5988` |
| The code prompt of the P3 chat page (merge two sorted lists), 256 tokens | 38.1, 41.6 | 68.3, 69.0 (2.90 tokens a forward, 167 of 174 drafts accepted) | `37b66ffc`, the text of P3 and of TQ4L12 |

oMLX (tokens per second, runs 2 and 3):

| | Fibonacci, 221 tokens | Merge, 256 tokens | Drafts accepted | Wired memory |
|---|---|---|---|---|
| MTP off | 49.0, 48.9 (42.7 cold) | 49.0 | | 75.9 GB |
| MTP on | 83.6, 84.0 (72.7 cold) | 80.7 | 164 to 166 of 174 to 175 (94% to 95%) | 77.7 GB |

The archive (`start-here.sh TQ4 <folder>`):

| Step | Result |
|---|---|
| Machine check | "Mac: Apple M4 Max, 128 GB." The pack comes from the argument, with no questions |
| Files | 168 files to check, 0.0 GB to download. The size and SHA-256 check took about 3.5 minutes |
| One chat | `nslm-chat` loads in 11.6 s and answers "Hello!" |
| Server | `run-TQ4.sh` starts `nslm-serve` at 32,768 tokens. It loads in 15.8 s |
| One request, the Fibonacci prompt, 256 tokens | The text of the `nslm-chat` runs, 40.9 tokens per second, the first token after 1.24 s |

Pages in memory after the `nslm-chat` runs (`tools/page_resident.py`):

| Part | Resident | On the disk |
|---|---|---|
| PLE table | 0.23 GB | 32.02 GB |
| MTP head | 163 MB | 163 MB |
| Other tensors (the 8-bit dense weights. The vision tower is not read) | 3.46 GB | 6.12 GB |
| Trellis banks (`.blkt`) | 44.99 GB | 63.58 GB |

## Notes

- This is the first Metal run of the 8-bit dense weights. At K = 0, TQ4 decodes about 9% slower than TQ4L12 with 4-bit
  dense weights (44.5 to 45.5 tokens per second on this Mac). With 2 drafts, the two are about equal.
- In oMLX, TQ4L12 gave 55.5 tokens per second, and 88.7 to 93.8 with MTP.
- The pack has no `switch_mlp` tensors. So `add_banks` gives each expert from the banks, the experts of the head
  included. With MTP on, L48 becomes `mtp.layers.0.mlp.switch_mlp`. MTP on gives the text of MTP off.
- `nslm-chat` loads in 10.1 to 15.1 s. The first token comes after 0.54 to 0.60 s. The maximum RSS is 0.8 to 1.3 GB,
  because the pack is memory-mapped.
- One HTTPS stream from the Hub ran at 36.6 MB/s on the same link while `hf download` stalled. Each restart of
  `hf download` checked all files again. On a Mac, use `start-here.sh`, which downloads with curl.
- `373499f` changes the pack choice of the start scripts after this test. A 96 GB Mac now gets TQ3-D4, not TQ4. The
  engine does not change.
