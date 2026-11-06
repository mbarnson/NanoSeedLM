NanoSeedLM engine @TAG@: nslm-chat and nslm-serve, built from https://github.com/mbarnson/NanoSeedLM at @SHA@.

Requirements:
- Windows x64 and Linux x64: an NVIDIA GPU of sm_80 or newer (RTX 30, 40 or 50 series) and driver 570 or newer.
  Windows 10 1903 or newer. Linux: glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or newer).
- macOS: Apple Silicon and macOS 15 or newer.

To start: double-click START-HERE.bat (Windows), or run ./start-here.sh (Linux, macOS). It checks this machine,
recommends a pack, downloads and checks it, chats once, writes run-PACK.bat (or run-PACK.sh) and starts the server on
http://127.0.0.1:8080/v1. Run it again to resume a download. Arguments skip the questions, for example
START-HERE.bat TQ3-D4 D:\models or ./start-here.sh TQ4 ~/models. It also downloads Unsloth's Q8_0 PLE table
(54.4 GB, read from the disk), because the KL on the model cards is with that table. NSLM_PLE=pack skips it.

To run by hand, run the commands from this folder. The binaries are in build/bin (Windows, Linux) or out/bin
(macOS, with the Metal kernels in out/res). The macOS binaries are signed with a Developer ID and notarized by Apple.
If macOS does not accept them, start-here.sh clears their download quarantine (xattr -dr com.apple.quarantine out).
Download a model first, for example:

  hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 --local-dir Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4

Then chat, or serve an OpenAI-compatible API (macOS: out/bin in place of build/bin):

  build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 "Why do tide pools matter?"
  build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 --ctx 32768 --port 8080

Models: https://huggingface.co/txgsync (Qwen3.8-Flash-Next-NanoSeedLM-TQ4, -TQ3-D4 and -TQ3).
Guide: https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/run-and-serve.md
This is a research preview. License: LICENSE (MIT). Each model has its own license.
