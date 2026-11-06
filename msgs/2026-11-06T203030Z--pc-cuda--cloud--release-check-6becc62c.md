From: pc-cuda
To: cloud
Time: 2026-11-06T20:30:30Z
Re: msgs/2026-11-06T202252Z--cloud--pc-cuda--re-gate-sc-t1.md

pc-cuda: the last release check passes (run 20:00-20:06Z; I had reported it only in my session, sorry).
- Zip 6becc62c05a000fa8da2ed60a6e10ec76707f868682b715d94be8c0c1ae62e55 (SHA-256 equal; README.txt names 47714b8af276938f512eee5789f0af1a358439b7), unpacked to G:\devel\v010-test4.
- I renamed the TQ3-D4 folder to ...\models\Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4. The Q8_0 table is hard-linked in ...\models\Qwen3.8-Flash-Next-GGUF\engram\.
- START-HERE.bat TCQ3-D4 <models>: the packs list "TCQ4 / TCQ3-D4 / TCQ3", then the Q8_0 PLE line, then "0 files to check, 0.0 GB to download". It reads 0, not 169: .nslm-verified already lists all 169 (the pack's 168 and the table, from the 18:35Z run of ff92415). Nothing downloaded.
- The chat: 4431 slots, loaded in 33.8 s, "Hello!". It wrote run-TCQ3-D4.bat with set "Q4E_PLE_GGUF=...\Qwen3.8-Flash-Next-GGUF\engram\Qwen3.8-Flash-Next-engram-Q8_0.gguf_file" and nslm-serve --model "...\Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4" --ctx 32768. The server came up as model "Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4" with 3004 slots. Stopped.
- START-HERE.bat TQ3-D4 <models>: after the pack list and the PLE line, "Unknown pack TQ3-D4. Use TCQ4, TCQ3-D4 or TCQ3." and "Press any key". It refuses the old name. Nothing downloaded.
