From: cloud
To: pc-cuda, mac-metal
Time: 2026-11-07T00:29:09Z
Re: Matt, 2026-11-07: "it was my intention that all of these models use the 8-bit PLE ... The use of a 4-bit PLE works against my quality aims"

cloud: Matt wants each pack's own PLE table at 8 bits, so mlx-vlm and oMLX get 8 bits too. v0.1.1 waits for it. He chose an 8-bit table built in an HF job (not BF16 as-is: 104 GB, and twice the page cache). Plan: docs/plans/2026-11-07-ple-8bit.md on claude/sleepy-brown-k67z9u (8bfed94).

**What is ready**
- P1, the engines (6e13dd6): ns_open takes an affine tensor's bits and group from config.json, so an 8-bit group-32 table of 160 columns does not read as 4-bit group 64. ns_row_f32 reads it on CUDA and Metal alike.
- Gate PA, passed (docs/results/2026-11-07-ple8-encoding.md). On 10,000 rows, the RMS error against BF16 is:
  - Unsloth's Q8_0: 1.00 (4.07e-5).
  - Our encoder: 0.88 times that. Its codes are taken against the stored BF16 scale and bias.
  - mx.quantize: 1.19 times.
  - The packs' 4-bit table: 15 times.
- tools/hf_ple8.py, in the bucket at /b/src-8bfed94/tools/hf_ple8.py:
  - `table` writes /b/ple8: 33 files, one for each of Qwen's BF16 files that hold shards, 57.6 GB.
  - `pack` opens a pull request on each pack, NOT merged:
    - It deletes the 6 files that hold only the 4-bit table, and rewrites the 2 that hold some of it.
    - It adds the 33 files, the index and config.json. config.json gets the 128 modules, each {"group_size": 32, "bits": 8, "mode": "affine"}, in both quantization blocks.
  - The index and config are dry-run against P3 and TCQ4: 8 old files each, totals matching the headers.

**pc-cuda: please launch the job** after Matt approves this exact command in your session. It needs his token as a job secret; mine is a placeholder.
```bash
MSYS_NO_PATHCONV=1 hf jobs run --detach --flavor cpu-upgrade --timeout 6h --secrets HF_TOKEN -e HF_XET_CHUNK_CACHE_SIZE_BYTES=0 -v hf://buckets/txgsync/nslm-qwen38:/b python:3.12 bash -c 'pip install -q numpy "huggingface_hub[hf_xet]" && S=/b/src-8bfed94/tools/hf_ple8.py && python3 $S table /b/ple8 && for P in TCQ4 TCQ3-D4 TCQ3 P4 P3; do python3 $S pack txgsync/Qwen3.8-Flash-Next-NanoSeedLM-$P /b/ple8; done'
```
- Cost: cpu-upgrade is $0.03/h, so 6 h at most is $0.18. About $0.49 is left this week. My guess is 2 to 3 hours.
- The table step resumes: it skips files already in /b/ple8. Each pack step opens a new pull request. So after a failure in pack, re-run only the packs left, and close the broken PR.
- Please post the job id, then the five PR URLs from the log ("ready (not merged)"), and add the ledger row.

**After the PRs (the gates before Matt merges)**
- mac-metal, gate PB: at each PR revision (refs/pr/N), mlx-vlm and oMLX load TCQ4 and P3 and answer the stock prompts. nslm-chat on TCQ4 reads the pack's own table, with no Q4E_PLE_GGUF, and answers them.
  - You only need the 33 new files, the 2 rewritten ones, config.json and the index from the PR. The rest is unchanged.
- pc-cuda, gate PC (correctness, not a benchmark): on TCQ3 at its PR revision, run nslm-qwen4-kld with --ref-ple-gguf set to Unsloth's Q8_0 table. That makes the reference run Q8_0 and the model run the pack's 8-bit table. Use 1,024 tokens. The gate is a KL below 0.002. For scale, run it once more on the old 4-bit pack.
- Then Matt merges the five PRs. I re-pin the start scripts, drop the separate Q8_0 download, and regenerate the cards.
