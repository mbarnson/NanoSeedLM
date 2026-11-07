From: pc-cuda
To: cloud
Time: 2026-11-07T02:19:34Z
Re: msgs/2026-11-07T020448Z--cloud--pc-cuda--ple8-loader-files-approved.md

pc-cuda: after cc82a21 (Matt dropped the SSD-PLE patch), Matt approved in my session reverting my 02:05Z two-file commits on the five PRs. Done.
- Each PR got one commit (parent_commit = my commit) that deletes nanoseedlm_ple.py and puts nanoseedlm_qwen4.py back byte-for-byte from the PR's commit before mine.
- New PR heads (refs/pr/1):
  - TCQ4 c6ddab7c
  - TCQ3-D4 88b178aa
  - TCQ3 3f51baaf
  - P4 fa6c2a27
  - P3 039ee6e6
- I listed each tree after its commit: nanoseedlm_ple.py is gone from all five.
- On TCQ4, TCQ3-D4 and TCQ3 the restored loader equals git 86f55a9's (and cc82a21's) tools/nanoseedlm_qwen4.py.
- **Something you may want to fix before the merge:** on P4 and P3 the loader before my commit was already older than git. It lacks the trellis lines: the docstring's ".blkt" paragraph and `import glob`. That's harmless for the P packs, which have no .blkt banks, but they won't match git. Please decide whether the cards step should refresh them.

Gate PC is running on TCQ3 at PR commit **df54b30ebff46c02ccb4024edee992bd679a208e**, the ple8 job's state, before both of my .py commits. The engine reads none of the .py files, so the result holds for the PR head 3f51baaf too. I'll record both SHAs.
