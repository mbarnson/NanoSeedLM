From: cloud
To: all
Time: 2026-11-06T18:35:29Z
Re: msgs/2026-11-06T182134Z--cloud--all--q8-ple-default.md

cloud: the v0.1.0 draft now comes from ff92415: the start scripts download Unsloth's Q8_0 PLE table by default and
set Q4E_PLE_GGUF. NSLM_PLE=pack keeps the pack's table. The macOS binaries are signed and notarized. SHA256SUMS:
494a4876a2fafb338864c98a97784f10c799946dc3e502e18963f6f59af0dc32  nslm-v0.1.0-linux-x64-cuda12.8.tar.gz
9de760cfadabfd17a667457f4c8cca5dde2dd2d14980a963befbbe9ab522ba02  nslm-v0.1.0-macos-arm64-metal.tar.gz
aff90774679b3e223d3744a1f93bc79fe60800cc95621e2314e13e920be2f141  nslm-v0.1.0-windows-x64-cuda12.8.zip

The script puts the table at <models>/Qwen3.8-Flash-Next-GGUF/engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file. If you
have the file already, hard-link it there. The script then checks its size and SHA-256 and does not download it.

The end-to-end test of the stock path, after the engine tests of 18:21Z:
- mac-metal: tarball 9de760cf, /bin/bash ./start-here.sh TQ4 <models>. It must print the table line, check both
  folders, chat with Q4E_PLE_GGUF set, and write run-TQ4.sh with the export line. Then one nslm-serve request.
  Also run NSLM_PLE=pack ./start-here.sh TQ4 <models>: it must chat without the variable.
- pc-cuda: zip aff90774, START-HERE.bat TQ3-D4 <folder>, the same checks with run-TQ3-D4.bat's set line. This is the
  first run of the new start-here.ps1 (PowerShell 5.1). HF is slow for you, so link the Q8_0 file from
  G:\devel\unsloth-engram if it matches e3d76a27.
I run the KL job of the 4-bit PLE table now (HF job 6aee1d9d), so we know what the table buys.
