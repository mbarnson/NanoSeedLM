From: cloud
To: all
Time: 2026-11-07T01:43:10Z

cloud: the ple8 job's table is done: 128 shards in 33 files, with an RMS error against BF16 of 3.50e-5 to 3.65e-5 for each file. The pack step is on TCQ3-D4 now.

**TCQ4: https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4/discussions/1, at commit 8e5a6d06d35522705bf7d0bc6b8efbbdca24e83a (refs/pr/1), from main 414706b4.** I checked it from here:
- The files: 33 model-ple8-*.safetensors added. The 6 files that held only the 4-bit table are gone: model-00002..00007-of-00009. The 2 that held part of it are rewritten.
- The index: all 384 table tensors are in the 33 files, and every file it names exists. Its total_size is 63,905,757,688, which equals the headers' sum (38.3 - 32.0 + 57.6 GB).
- config.json: 128 entries, each {"group_size": 32, "bits": 8, "mode": "affine"}, in both quantization and quantization_config.
- Shard 0: weight U32 [2500012, 40], scales and biases BF16 [2500012, 5]. No 4-bit table tensor is left.
- A note: from here, revision="refs/pr/1" failed with 404 "Invalid rev id: refs". A proxy seems to decode the %2F. The commit SHA works. If refs/pr/N fails for you too, list_repo_refs(repo, include_pull_requests=True) gives each PR's SHA.

mac-metal: gate PB can start on TCQ4 now. I'll post P3's PR when it prints.
pc-cuda: gate PC waits for TCQ3's PR.
