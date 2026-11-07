# Trellis experts in safetensors shards (2026-11-07)

**Result.** The trellis experts of a pack are now safetensors entries in `model-tcq-*` shards, with the same bytes as
the `.blkt` banks. On the M4 Max, a converted TCQ3 gives the same text as the `.blkt` TCQ3 in the Metal engine and in
mlx-vlm. Warm decode is within 0.7%, and the load time and the peak memory do not change. On CUDA, a tiny model gives
the same logits from shards as from banks, bit for bit.

**Status.** Current. Plan tcq-safetensors, steps TS1 to TS7 ([the plan](../plans/2026-11-07-tcq-safetensors.md)). The
CUDA check used the tiny model only.

## Setup

- Code: `cdabec6` (branch `mac/tcq-safetensors`). `nslm-qwen4-pack --tcq-st DIR` converts a folder in place.
- Mac: M4 Max, 128 GB, macOS 27.2, on battery (61% to 56%), load 1.9 to 3.5. A browser and the terminal ran.
- Model: TCQ3 at `a850ead` (its 147 banks have the SHA-256 of the published ones), as two APFS clones. Both clones
  had the MLX loader files of `cdabec6`. One clone was converted.
- Engine: `nslm-chat --temp 0 --no-think --max-tokens 128`, two prompts ("Why do tide pools matter?", and a C prime
  function with an explanation), in 4 rounds. Rounds 1 and 2 started with the `.blkt` folder, rounds 3 and 4 with the
  converted folder.
- MLX: mlx-vlm 0.7.11, MLX 0.32.4, `python -m mlx_vlm.generate --trust-remote-code --temperature 0 --max-tokens 64`.
- CUDA: `test_qwen4_engine` in the Linux CI job `6aef74990e0338d1bb83ab5f` (rtx-pro-6000).
- Hugging Face: the job `6aef74b50e0338d1bb83ab7d` (cpu-xl) ran `tools/hf_tcq_st.py` on each pack, one pull request
  each. Then a card note, and the job `6aef7b6ce6320c66fad634a8` (cpu-xl) ran `start-here.sh` at the old pin and at
  the new pin, as a v0.1.1 release folder would (`632e383`).

## Results

The conversion of TCQ3 on the M4 Max:

| | Value |
|---|---|
| Banks in, shards out | 147, 49 (one for each layer and one for the MTP head) |
| Time, with the check of each stream against its bank | 68 s |
| Size of a layer's shard | 938 MB |

The two folders in the Metal engine (prompt 2, the run after each switch of folder and the run after it):

| Folder | Cold run: load, decode | Warm run: load, decode |
|---|---|---|
| `.blkt` (round 3) | 19.3 s, 39.4 tok/s | 4.1 s, 44.2 tok/s |
| `model-tcq-*` (round 3) | 4.0 s, 44.4 tok/s | 4.1 s, 44.2 tok/s |
| `.blkt` (round 4) | 19.7 s, 39.1 tok/s | 4.2 s, 44.2 tok/s |
| `model-tcq-*` (round 4) | 19.4 s, 38.8 tok/s | 4.3 s, 43.9 tok/s |

| Check | `.blkt` | `model-tcq-*` |
|---|---|---|
| Text of `nslm-chat`, 2 prompts, all rounds | the same | the same |
| Peak memory footprint (`time -l`) | 3.25 GB | 3.25 GB |
| Text of mlx-vlm, 64 tokens | the same | the same |
| CUDA, tiny model: logits of 20 prompt rows and 8 decode rows | | bit for bit the same |

The three packs on Hugging Face (`cpu-xl`). The download, the conversion with its check, and the upload of the pull
request with the 49 new shards:

| Pack | Old revision | New revision | Download | Conversion | Upload |
|---|---|---|---|---|---|
| TCQ3 | `a850ead` | `603685c` | 99 s | 321 s | 18 s |
| TCQ3-D4 | `e29730a` | `b3bc4f8` | 103 s | 241 s | 18 s |
| TCQ4 | `4e0d254` | `37f9319` | 112 s | 443 s | 21 s |

- Each new revision has 101 files: no `.blkt`, 49 `model-tcq-*` shards. The old revisions stay in the history.
- TCQ3's 49 shards have the SHA-256 of the M4 Max conversion. Its index adds the same 588 entries (147 banks times
  4) to the old index, and its other 3,380 entries do not change.

`start-here.sh` on the HF job, with the release engine of v0.1.1 (the chat fails: the job has no GPU):

| Run | Files | GB | Time | Result |
|---|---|---|---|---|
| The v0.1.1 pin of TCQ3, curl, 8 files at once | 198 to download | 112.1 | 1,271 s | 198 checked, 147 banks |
| Then the new pin, in the same folder | 53 to download, 47 kept | 48.2 | 254 s | 48.2 GB of banks removed, 100 checked |

## Notes

- The upload of each pull request took about 20 s for 48 GB. Xet stored the bytes of the banks before, so the shards
  added little (hypothesis H3).
- The job of the start script found a bug: `hf version` of huggingface_hub 2.x does not print "huggingface_hub", so
  `start-here.sh` did not find `hf` and used curl. The script now asks `hf download --help` for `--local-dir`.

- The first run after a switch of folder is cold for both formats. APFS clones are separate files, so each folder has
  its own page cache, and the other folder's pages go. Only the warm runs compare the formats.
- Hypothesis H2 of the plan did not hold: the load time and the peak memory did not fall. With the `.blkt` banks the
  peak memory footprint is also 3.25 GB, so that path did not copy the 47 GB of banks into memory either.
- The engine test `test_qwen4_metal` and the CUDA test `test_qwen4_engine` now open a pack with `model-tcq-*` shards
  and compare it with the `.blkt` folder, bit for bit.
