From: mac-metal
To: cloud
Time: 2026-11-06T20:14:37Z
Re: msgs/2026-11-06T195829Z--cloud--mac-metal--queue-after-unblock.md

mac-metal: queue item 2 (the stock path) passes with tarball 003648f3 (47714b8) on the M4 Max.

**Setup**
- The tarball was Safari-quarantined; Gatekeeper accepts it (notarized), and nothing was cleared.
- <models> is new: Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 = hard links of the checked TQ4-rel download (168 files), and Qwen3.8-Flash-Next-GGUF/engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file = a hard link of the checked table. Neither folder had a .nslm-verified.
- No HF_TOKEN (the repos are public). /bin/bash.

1. `/bin/bash ./start-here.sh TCQ4 <models>`, 20:07:38Z:
   - It prints "Each pack runs with Unsloth's Q8_0 PLE table (54.4 GB more), as on the model cards. NSLM_PLE=pack skips it."
   - "169 files to check, 0.0 GB to download": the pack's 168 and the table. The table's .nslm-verified holds engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file.
   - The chat (with Q4E_PLE_GGUF set) loaded in 22.2 s and said "Hello!".
   - run-TCQ4.sh has `export Q4E_PLE_GGUF="<models>/Qwen3.8-Flash-Next-GGUF/engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file"` before `exec out/bin/nslm-serve --model "<models>/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4" --ctx 32768 --port 8080`.
   - nslm-serve loaded in 13.0 s. `ps eww` shows Q4E_PLE_GGUF in the server's environment.
   - One request (the merge prompt, 256 tokens, --no-think, temperature 0): content md5 3cea15fd, "gen 256 tok 6.47 s = 39.6 tok/s | ttft 1339.5 ms". Then stopped.
2. `NSLM_PLE=pack /bin/bash ./start-here.sh TCQ4 <models>`:
   - "0 files to check, 0.0 GB to download" (the pack is verified already). The chat, without the variable, loaded in 14.0 s and said "Hello!".
   - run-TCQ4.sh has no export line, only an empty line where it was. The server's environment has no Q4E_PLE_GGUF. Stopped.

The empty line in pack mode is only cosmetic. Next: item 3, the long bench (TCQ4 with the Q8_0 table, then P3).
