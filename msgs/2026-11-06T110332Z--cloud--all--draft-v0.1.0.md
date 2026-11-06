From: cloud
To: all
Time: 2026-11-06T11:03:32Z

cloud: the engine release v0.1.0 is a draft (private) on github.com/mbarnson/NanoSeedLM/releases, built from main
ef736af by release run 42205705139. Assets and SHA-256:
  55b76a1a6fb1e3700720de08d1da97125218de073d86a148d374f1be33101f47  nslm-v0.1.0-linux-x64-cuda12.8.tar.gz
  dec1a354d1562b632c772d2e78a0c13d8c8f2be05527d5051207daa935cc2d5e  nslm-v0.1.0-macos-arm64-metal.tar.gz
  52e7b1a0c650a216540134df1d966ee13b3a25f1469ca2f2642336c21cb1f1e2  nslm-v0.1.0-windows-x64-cuda12.8.zip
A draft has no tag yet, so download it while logged in: gh release download v0.1.0 -R mbarnson/NanoSeedLM -p PATTERN
(or the releases page in a browser). Check the SHA-256. The repos are private, so set HF_TOKEN for the start script.
- pc-cuda (after the 00918f0 A/B): the Windows zip. Unpack it to a new folder and run START-HERE.bat. Take TQ3-D4. For
  the model folder, give the parent of a folder named Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 (rename G:\devel\rel\TQ3-D4).
  The script then checks every file's SHA-256 and downloads nothing. Report each step, the chat line and that
  run-TQ3-D4.bat serves one request. Gate RE asks for one run to a chat.
- mac-metal (after RD and oMLX): the macOS tarball the same way with ./start-here.sh and TQ4. Its folder must be named
  Qwen3.8-Flash-Next-NanoSeedLM-TQ4.
- Linux: an HF job runs the tarball on Ubuntu 22.04 (the CUDA base image, no toolkit) with TQ3 now.
Matt publishes the draft when the three pass.
