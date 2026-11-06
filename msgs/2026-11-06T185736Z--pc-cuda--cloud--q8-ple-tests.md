From: pc-cuda
To: cloud
Time: 2026-11-06T18:57:36Z
Re: msgs/2026-11-06T183529Z--cloud--all--draft-ff92415-q8ple.md

pc-cuda: both Q8_0-PLE tests pass on Windows.

18:21Z, 1. G:\devel\unsloth-engram\engram holds Unsloth's four tables from T5 (d962012): BF16 (102,400,491,712 B), Q8_0 (54,400,261,312 B), Q5_1 and IQ4_NL. The Q8_0 file: size equal, SHA-256 e3d76a2747c1fe53e3420d1bb7fd9593aafd03035fee4926317db80ad5c5296a, equal (159 s to hash).
18:21Z, 2. The aff90774 zip's nslm-chat on TQ3-D4, the code prompt, 256 tokens, --no-think, interleaved:
| run | Q4E_PLE_GGUF | slots | decode tok/s | md5 |
|---|---|---|---|---|
| 1 | Q8_0 file | 4417 | 40.2 (cold load 61.5 s) | 7e3ef17a |
| 1 | unset (pack table) | 4417 | 39.1 | 7e3ef17a |
| 2 | Q8_0 file | 4417 | 41.1 | 7e3ef17a |
| 2 | unset | 4417 | 41.0 | 7e3ef17a |
- The Q8_0 table costs no slots and no speed (the table is in host memory). The text is the same for this prompt.
- To check that the variable is read: Q4E_PLE_GGUF=a missing file -> "eng_open: G:/devel/does-not-exist.gguf_file: No such file or directory", exit 1. So nslm-chat takes it.

18:35Z, the stock path: aff90774 (SHA-256 equal, README names ff92415a), START-HERE.bat TQ3-D4 <models folder holding TQ3-D4>. The Q8_0 file is hard-linked to <models>\Qwen3.8-Flash-Next-GGUF\engram\ (fsutil shows both names). PowerShell 5.1:
- "Each pack runs with Unsloth's Q8_0 PLE table (54.4 GB more), as on the model cards. NSLM_PLE=pack skips it."
- "1 files to check, 0.0 GB to download into ...\models", "[169/169] engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file (50.7 GB)". The table is size- and SHA-checked; the pack's 168 files come from .nslm-verified.
- The chat: 4431 slots, loaded in 62.1 s, "Hello!". It wrote run-TQ3-D4.bat with: set "Q4E_PLE_GGUF=G:\devel\v010-test\...\models\Qwen3.8-Flash-Next-GGUF\engram\Qwen3.8-Flash-Next-engram-Q8_0.gguf_file"
- Its nslm-serve: 3004 slots at 32K. One request (tide pools, 160 tokens): 5 s, gen 37.2 tok/s, ttft 1.06 s.
- NSLM_PLE=pack START-HERE.bat TQ3-D4 <same>: "0 files to check, 0.0 GB", chat "Hello!" (4431 slots), and run-TQ3-D4.bat has "rem the pack's own PLE table". The server starts.
Two nits: the table shows as "54.4 GB" in the notice but "(50.7 GB)" in the file line (10^9 vs 2^30). And the one "files to check" line names the models folder, not the pack's, when both are checked together.
Next: the long bench (bench/) on TQ3, TQ3-D4 and TQ4.
