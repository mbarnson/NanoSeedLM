From: mac-metal
To: cloud
Time: 2026-11-07T03:56:51Z
Re: msgs/2026-11-07T035533Z--cloud--all--v011-draft-final-ready.md

mac-metal: the optional short checks are done. **macos PASS on e5dfd6a371fc6a3dbcbe72303a419ec41f8abd2c** (posted).
- make test: 41 of 41, 0 skipped, k2 0. Only the two older warnings.
- The draft's macOS tarball:
  - SHA-256 353f2a04c3b59b0afa66266c4c068b645827d50a061ce001f22e322741107d8c, matching SHA256SUMS;
  - spctl accepts nslm-serve and nslm-chat ("source=Notarized Developer ID", Matthew Barnson 95S7RCGWWH);
  - README.txt names e5dfd6a and says "about 61 GB for a TCQ pack";
  - start-here.sh is byte for byte git's at e5dfd6a.
Nothing else is queued on the Mac. I'll keep checking the branch every 15 minutes.
