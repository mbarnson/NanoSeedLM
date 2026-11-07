NanoSeedLM engine @TAG@: nslm-chat and nslm-serve, built from https://github.com/mbarnson/NanoSeedLM at @SHA@.

Requirements:
- Windows x64 and Linux x64: an NVIDIA GPU of sm_80 or newer (RTX 30, 40 or 50 series) and driver 570 or newer.
  Windows 10 1903 or newer. Linux: glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or newer).
- macOS: Apple Silicon and macOS 15 or newer.

To start: double-click START-HERE.bat (Windows), or run ./start-here.sh (Linux, macOS). It checks this machine,
recommends a pack, downloads and checks it, chats once, writes run-PACK.bat (or run-PACK.sh) and starts the server on
http://127.0.0.1:8080/v1. Run it again to resume a download. Arguments skip the questions, for example
START-HERE.bat TCQ3-D4 D:\models or ./start-here.sh TCQ4 ~/models. Each pack carries its PLE table at 8 bits. The
engines read it from the disk, and the page cache holds the rows in use.

To run by hand, run the commands from this folder. The binaries are in build/bin (Windows, Linux) or out/bin
(macOS, with the Metal kernels in out/res). The macOS binaries are signed with a Developer ID and notarized by Apple.
If macOS does not accept them, start-here.sh clears their download quarantine (xattr -dr com.apple.quarantine out).
Download a model first, for example:

  hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4 --local-dir Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4

Then chat, or serve an OpenAI-compatible API (macOS: out/bin in place of build/bin):

  build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4 "Why do tide pools matter?"
  build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4 --ctx 32768 --port 8080

Models: https://huggingface.co/txgsync (Qwen3.8-Flash-Next-NanoSeedLM-TCQ4, -TCQ3-D4 and -TCQ3).
All packs: https://huggingface.co/collections/txgsync/qwen38-flash-next-trellis-seedlm-oqe-mxfp4-6aebb5915b4644081194047a
Guide: https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/run-and-serve.md
Serving to other machines: nslm-serve listens on 127.0.0.1 only. --host 0.0.0.0 needs an API key (NSLM_API_KEY=KEY,
or --api-key KEY), and clients then send "Authorization: Bearer KEY". A web page can use the server only with
--allow-origin ORIGIN.

Changes since v0.1.0:
- The packs carry an 8-bit PLE table, made from Qwen's BF16 table: every runtime reads it, the engines, mlx-vlm and
  oMLX. Its row error against BF16 is 0.88 times that of Unsloth's Q8_0 table. The start scripts no longer download
  Unsloth's table. The packs are now 112 to 131 GB. v0.1.0's start scripts read the packs' newest files, which its
  engine cannot read: use this release.
- Upgrading a v0.1.0 folder: run this release's start script on it. It checks each file by SHA-256, keeps the ones
  that did not change, downloads the changed ones (about 61 GB for a TCQ pack), and removes the old PLE shards. The
  folder Qwen3.8-Flash-Next-GGUF, Unsloth's table from v0.1.0, is no longer used: delete it to free 54.4 GB.
- nslm-serve: a stranger cannot use it.
  - Each reply used to allow every web origin. Now a browser origin needs --allow-origin.
  - Without a key, the server takes only a loopback Host, which stops DNS rebinding.
  - An API key is needed off loopback.
  - Sanitizers (ASan, UBSan and LeakSanitizer, compute-sanitizer, MSVC ASan) found and fixed reads past the end of a
    crafted model file's header and JSON. The builds have stack protection, _FORTIFY_SOURCE, full RELRO and Control
    Flow Guard.
- MTP drafts in nslm-serve and nslm-chat at any temperature. A draft comes from the MTP head's distribution, and the
  sampler keeps it with min(1, p/q), so each token keeps the sampler's distribution. Q4E_DRAFTS=2 turns them on, and
  Q4E_DRAFT_MIN_P=0.3 verifies only the drafts that the head is sure of. On a Mac, run-PACK.sh turns them on. Then the
  M4 Max decodes TCQ4 32% faster at temperature 0, with the same text, and 16% faster at 1.0. On the RTX 4080 they break
  even, so they stay off there.
- The KV cache of Qwen3.8-Flash-Next is BF16. At a 32K context the RTX 4080 keeps 379 more experts in VRAM, and
  decode is 5.7% faster.
- Metal: merged projections, TCQ4 decode 5.9% faster with the same text.
- start-here: TCQ3 for a 12 or 16 GB GPU. Sizes in decimal GB. A stalled download resumes on a new connection.

This is a research preview. License: LICENSE (MIT). Each model has its own license.
