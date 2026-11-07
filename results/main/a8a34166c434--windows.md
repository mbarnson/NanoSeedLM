Status: PASS
Branch: main
Commit: a8a34166c434620d13595037bffe09bbdf9e2416
Platform: windows
Agent: pc-cuda
Time: 2026-11-07T03:45:17Z

Windows 11, RTX 4080 16 GB, driver 610.88, CUDA 13.1, 128 GB RAM.

**Tests at a8a34166c434620d13595037bffe09bbdf9e2416** (G:/devel/NanoSeedLM-rel; cmake reconfigured, Release, Ninja):
- cl: 100% tests passed out of 46.
- clang-cl: 100% tests passed out of 47.

**The v0.1.1 draft's Windows release check (the zip built from a8a3416):**
1. **The zip:** nslm-v0.1.1-windows-x64-cuda12.8.zip, 17,604,329 bytes.
   - SHA-256 ca6b1c268ecfb15eef089287b2cb853a9799f279f49ca77e0a7581d515939210, the same as the draft's SHA256SUMS.
   - Its start-here.ps1 equals git a8a3416's start/start-here.ps1, apart from CRLF line endings.
2. **The upgrade of a v0.1.0 folder:** `START-HERE.bat TCQ3-D4 G:\devel\v010-test\nslm-v0.1.0-windows-x64-cuda12.8\models`, the first run of the new .ps1. The folder is v0.1.0's TCQ3-D4 at 7e5edb2b, with its path-only .nslm-verified and no hard links.
   - The menu shows "Each pack carries its PLE table at 8 bits".
   - It checked the earlier files by SHA-256, then printed "Removed 26.5 GB of earlier files that this revision of the pack does not have" (the six 4-bit shards).
   - It then printed "49 files to download, 60.9 GB". These were model-00001 and -00008, the 33 table files, and the small files that changed.
3. **The stop and resume:** the download fell to about 0.8 to 1.2 MB/s around table file 15, then recovered. At file 17 I stopped it by PID, with 696,872,960 bytes in its .part file.
   - The rerun said "26 files to download, 29.0 GB".
   - It went on from the .part: the file reached 1.14 GB about 20 s after the restart.
   - Then it finished, the chat answered "Hello!", and it wrote run-TCQ3-D4.bat and started nslm-serve.
4. **The folder afterwards matches the Hub tree** of txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4 at the pin e29730a3fea357b710f0b5bd57158c74a82cd7f1:
   - 198 files, each with the Hub's size. `.gitattributes` is the only Hub file it doesn't have.
   - .nslm-verified has 198 "path size id" lines, equal to the Hub's sizes and SHA-256 or git blob ids.
   - No .part left; 110 GB on disk.
5. **run-TCQ3-D4.bat:**
   - Its PLE line is a rem ("the pack's own 8-bit PLE table (set Q4E_PLE_GGUF=FILE reads another, as a GGUF)"). No Q4E_PLE_GGUF is set.
   - The server was up in 35 s (loaded in 33.0 s).
   - /v1/models lists Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4.
   - One /v1/chat/completions request ("What is the capital of France? Answer in one word.", temperature 0) answered "Paris", finish_reason stop.
   - Server log: prompt 64 tokens in 851 ms, 36 tokens at 36.7 tok/s, TTFT 965 ms.
6. **The rerun of START-HERE.bat TCQ3-D4:** "0 files to download, 0.0 GB". Then the chat ("Hello!", decode 24.4 tok/s) and the server.
- The servers were stopped by PID.
- Speeds are shown only as the tools print them. This was not a benchmark.
