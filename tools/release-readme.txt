NanoSeedLM engine @TAG@: nslm-chat and nslm-serve, built from https://github.com/mbarnson/NanoSeedLM at @SHA@.

Requirements:
- Windows x64 and Linux x64: an NVIDIA GPU of sm_80 or newer (RTX 30, 40 or 50 series) and driver 570 or newer.
  Windows 10 1903 or newer. Linux: glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or newer).
- macOS: Apple Silicon and macOS 15 or newer.

Run the commands from this folder. The binaries are in build/bin (Windows, Linux) or out/bin (macOS, with the Metal
kernels in out/res). Download a model first, for example:

  hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 --local-dir Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4

Then chat, or serve an OpenAI-compatible API (macOS: out/bin in place of build/bin):

  build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 "Why do tide pools matter?"
  build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 --ctx 32768 --port 8080

Models: https://huggingface.co/txgsync (Qwen3.8-Flash-Next-NanoSeedLM-TQ4, -TQ3-D4 and -TQ3).
Guide: https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/run-and-serve.md
This is a research preview. License: LICENSE (MIT). Each model has its own license.
