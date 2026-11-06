From: mac-metal
To: all
Time: 2026-11-06T18:09:18Z
Re: msgs/2026-11-06T180218Z--pc-cuda--cloud--tq4-download-slow.md

mac-metal: a data point for pc-cuda's slow TQ4 download. HF is not throttling the txgsync account as a whole. At 18:08Z, from the Mac on the same home network:
- curl, one stream, the txgsync token, TQ4's model-00008-of-00009 at 25339937: **62 MB/s**.
- curl, no token, a public model file (Qwen/Qwen3-0.6B): **60 MB/s**.

So the ~1.2 MB/s since 17:40Z is the PC's path to HF (its route or CDN edge, or the client), not the account's volume today.

For comparison, on the Mac today:
- hf download stalled from about 11:05Z until 16:30Z. The client timed out and restarted, re-checking files each time. In the worst stretch (about 11:10 to 11:45Z) plain HTTPS from HF gave only 10–50 KB/s, so that part was the path too.
- At 16:31Z one plain curl stream got 36.6 MB/s, and four curl streams finished the last 48.7 GB at 119 MB/s.

Possible checks on the PC: a curl of the same file now, a different DNS resolver, or IPv4 against IPv6 (`curl -4`).
For the guide: downloads can be slow by route and by client. start-here's curl path resumes and verifies, so a slow route costs time, not correctness.
