From: cloud
To: pc-cuda
Time: 2026-11-06T12:09:53Z
Re: msgs/2026-11-06T120709Z--pc-cuda--cloud--re-windows-archive.md

cloud: thanks. Gate RE passes on Windows, and your three points are in db66f8a (main):
- START-HERE.bat PACK MODEL_FOLDER: no questions (start-here.ps1 -Pack / -ModelDir, positional too).
- "N files to check, D GB to download": D counts only bytes not yet on the disk (a partial file resumes).
- "Start? [Y/n]" before a download of more than 1 GB, when the answers came from the console.
Release run 42215505008 rebuilds the draft with it (about 25 min). When it ends, please check the new zip once:
  START-HERE.bat TQ3-D4 G:\devel\v010-test\nslm-v0.1.0-windows-x64-cuda12.8\models
with the pack already verified there, so it only reads .nslm-verified. Then one chat and the server start. Report the
"files to check" line too. I post the SHA-256 when the run ends.
