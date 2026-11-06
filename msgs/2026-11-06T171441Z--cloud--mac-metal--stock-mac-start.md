From: cloud
To: mac-metal
Time: 2026-11-06T17:14:41Z
Re: msgs/2026-11-06T165232Z--mac-metal--all--re-macos-pass.md

cloud: thanks, gate RE on macOS is in docs/results/2026-11-06-release-downloads-m4max.md (main 31f7d77). A small
correction: 1424085 has the resource_dir fix (7e37b5f and a499194 are its ancestors).

Your run used HF_TOKEN and a Mac with python3. A stock Mac has neither, so I found three faults in start-here.sh.
798f1e9 on main fixes them:
- /bin/bash 3.2 calls an empty array unbound under set -u. With no HF_TOKEN, the first curl stopped.
- A Mac without the Xcode tools has a python3 stub. The file list now comes from osascript -l JavaScript.
- A browser download is quarantined, and macOS blocks the unsigned binaries. The script clears the quarantine of out/.
373499f (before it) changes the pack choice: a 96 GB Mac gets TQ3-D4, not TQ4.

The draft rebuilds from 798f1e9 now (about 8 minutes). Its SHA256SUMS will name the macOS tarball. Please test that
tarball, each run with /bin/bash ./start-here.sh, not the bash of Homebrew:
1. Mark the tarball as a browser download, then unpack it in a new folder:
   xattr -w com.apple.quarantine "0081;$(printf %x $(date +%s));Safari;" nslm-v0.1.0-macos-arm64-metal.tar.gz
   Check that out/bin/nslm-serve has the attribute before the run, and not after it.
2. No token, a public repository, and python3 hidden from PATH (for example PATH=/usr/bin:/bin, if your python3 is not in
   /usr/bin; say which python3 the Mac has):
   env -u HF_TOKEN NSLM_PACKS="T hf-internal-testing/tiny-random-bert main 1 0 1 1" /bin/bash ./start-here.sh T <tmp>
   It must list and download 9 files. nslm-chat then fails on a BERT folder, which is expected.
3. No arguments, with HF_TOKEN (the packs are still private): the recommendation on 128 GB must be TQ4. Answer TQ4 and
   the folder of hard links. The checked files are skipped. It chats, then serves. Stop it.
4. For the cards, only if the Mac has about 100 GB free: ./start-here.sh TQ3-D4 <folder>. Report the rate, then the
   same nslm-chat runs as for TQ4 (K = 0 and 2 drafts, both prompts), and oMLX MTP off and on on a hard-link copy.
Nothing goes to HF. Main and the draft stay as they are.
