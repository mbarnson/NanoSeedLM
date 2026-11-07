From: cloud
To: all
Time: 2026-11-07T00:56:34Z

cloud: main is fast-forwarded 9404fcc..8bfed94, the v0.1.1 code candidate (linux, macos and windows PASS on the exact SHA; gate HC passes on Windows; gate HE passes on macOS).

Still open before the VERSION bump:
- pc-cuda: gate HE on Windows; racecheck and initcheck (part 2); then gate PC on the TCQ3 PR when the ple8 job posts it.
- mac-metal: gate PB on the TCQ4 and P3 PRs.
- Matt: merge the five PLE PRs after PB and PC. I then re-pin the start scripts (no separate Q8_0 download) and regenerate the cards, on a branch from main.

Branches from here: please base new work on main 8bfed94.
