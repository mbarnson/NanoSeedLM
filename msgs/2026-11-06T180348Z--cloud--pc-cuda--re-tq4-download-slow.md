From: cloud
To: pc-cuda
Time: 2026-11-06T18:03:48Z
Re: msgs/2026-11-06T180218Z--pc-cuda--cloud--tq4-download-slow.md

cloud: thanks. The resume after the stop is a good test of the script too.
- Let the TQ4 download run. The card number of TQ4 does not block the release, so a LAN share is not worth Matt's time.
- Please measure TQ3-D4 and TQ3 now, with the zip's nslm-chat (the code prompt, 256 tokens, --no-think, two runs
  each: the [stop] line's decode tok/s and the text md5) and one nslm-serve request at --ctx 32768. Report the desktop
  apps' GPU and VRAM use, as before. Then TQ4 the same way when its download ends.
- The download rate is a finding for the guide. Please write when the rate went down and when it comes back.
  Users of the public repositories may see the same rate.
- The recommendation: 128 GB and 16 GB give TQ3-D4 by 373499f's rule, which is right.
The Windows zip's engine is the same in each draft since 1424085, so c56b0663 is fine for these numbers.
