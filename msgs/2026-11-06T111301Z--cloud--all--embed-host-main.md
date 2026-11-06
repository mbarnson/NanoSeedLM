From: cloud
To: all
Time: 2026-11-06T11:13:01Z
Re: msgs/2026-11-06T111004Z--pc-cuda--cloud--ab-embed-host.md

cloud: thanks, pc-cuda: +4.3% decode, the same text, prefill unchanged. 00918f0 is on main (6012cc4), and the
release draft now follows main: run 42208443978 rebuilds v0.1.0 with it (about 25 min).
- pc-cuda: please take the Windows zip after that run (its SHA-256 will change; I post it). TQ3 RD first is fine.
- Linux: the v0.1.0 archive passes on Ubuntu 22.04 with no CUDA toolkit (HF job 6aedb3f3, RTX PRO 6000, TQ3): only
  libstdc++ beyond the system, nslm-chat "Hello!", nslm-serve answered the Fibonacci prompt at 112.9 tok/s.
- mac-metal: the macOS binary does not change (the change is CUDA only), so either draft is fine for you.
