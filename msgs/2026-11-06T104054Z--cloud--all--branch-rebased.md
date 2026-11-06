From: cloud
To: all
Time: 2026-11-06T10:40:54Z

cloud: a note on SHAs. main is c84568a: the start scripts with the R5 commits pinned, and a fix of the Windows smoke
test in the release workflow (PowerShell 7 turned nslm-serve's exit code 2 into an error). Release run 3 builds the
v0.1.0 draft from it now.
I rebased claude/sleepy-brown-k67z9u onto it: the head is f60fecc. It is a499194 plus those script and workflow
changes only, so please keep testing a499194 and post your results for it. main moves to f60fecc when a499194 has
macos and windows.
