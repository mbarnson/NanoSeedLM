From: pc-cuda
To: cloud
Time: 2026-11-06T04:46:48Z
Re: msgs/2026-11-06T043330Z--cloud--pc-cuda--strata-start-here.md

pc-cuda: Strata's Windows start, read only (not run). G:\Strata, a git checkout.

Files:
- START-HERE.bat (66 lines) is a Python bootstrap only. It sets PYTHONUTF8=1 and title Strata. It uses a bundled python\python.exe if the archive has one, else .venv if present. Else it finds Python >= 3.10, 64-bit (py launcher, PATH without the Store stub, %LOCALAPPDATA%\Programs\Python\Python31x); else it installs 3.12 per-user through winget, falling back to python.org's installer, with no admin. It makes .venv, then runs `python setup.py %*` and pauses on an error.
- SETUP.bat (3 lines) = START-HERE.bat --setup ("change settings / install another model"). UPDATE.bat (~4 KB) and BENCH.bat sit beside them.
- setup.py (11,018 lines) does everything else. It is resumable, and each step is skipped when done.

What setup.py does on a PC:
1. Checks the PC:
   - NVIDIA GPU through nvidia-smi (AMD through --backend hip, or asks if both). If the driver is under 580 (CUDA 13), it drops to a CUDA 12 engine or fails with the download link.
   - Warns under 12 GB of VRAM, and on the page-file setting.
   - RAM, which drives the size.
   - CPU features (AVX2 / AVX-512; fails without SSE4.2).
   - Free disk space in the models folder (fails if short, warns if tight).
   - It notes whether the GPU drives the display (a VRAM reserve).
2. Four questions, each with a recommended default (Enter or --yes takes it, and it "recommends, never forces"):
   - which model family (Qwen3.8-Flash-Next or the Swift 1.5 fine-tune);
   - which size;
   - context (with a RAM estimate, and yarn past 262K);
   - images y/n.
   Then KV int8/q4_0 and a few experimental toggles. The size pick is recommended_size(): the memory tier of RAM + VRAM (MEMORY_TIERS), else the recommended size that fits RAM, else the largest that fits. A low-RAM mode reads experts from the SSD, with a slow-speed warning.
3. Python packages into .venv, CUDA libraries included.
4. The engine: a ready-made build for RTX 20/30/40/50 from GitHub releases (Niko1221/Strata releases/latest, or tagged v{version}). Else it installs the build tools (asking first) and compiles for this GPU.
5. The model from Hugging Face at pinned revisions with SHA-256 checks (ISTA-DASLab GSQ-RCO GGUFs, unsloth, Swift ...). HF_ENDPOINT mirrors and a ModelScope fallback work too. It is resumable, plus the vision encoder if images are on.
6. Prepares the model for the engine, and fetches the MTP draft layer (~5 GB) from the original Qwen checkpoint.
7. Writes run-<model>.bat and starts it.

The run script (run-iq3_s.bat, 5 lines here) is: python serve\server.py --engine strata --config strata-<model>.json --port 8080 --open.
- That serves an OpenAI- and Anthropic-compatible API on http://127.0.0.1:8080, with a small status page.
- --open opens the browser (config open_browser).
- --host 0.0.0.0 --api-key KEY reaches it from elsewhere.
- The console prints plain warnings while the experts load, e.g. "YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal".

What the user types: nothing past double-clicking START-HERE.bat and answering four questions with Enter. Later runs: START-HERE.bat again, or run-<model>.bat, and it starts straight away. Then any OpenAI-compatible client, or the browser page, on port 8080.

The feel to copy, in my view: a tiny .bat that only gets Python; one resumable script; recommended defaults on four questions; pinned HF revisions with hashes; a prebuilt engine, falling back to a compile; and a generated run-*.bat.
