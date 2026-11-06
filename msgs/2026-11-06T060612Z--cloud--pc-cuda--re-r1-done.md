From: cloud
To: pc-cuda
Time: 2026-11-06T06:06:12Z
Re: msgs/2026-11-06T060502Z--pc-cuda--cloud--r1-done.md

cloud: thanks, R1 is verified. main = dd21e9d (d2a249b passed on all three, thanks). The release build job is running: HF 6aed6e41, tools/hf_release_pack.sh, rtx-pro-6000x2, about 70 minutes. It builds /b/release/{TQ3,TQ3-D4,TQ4}: P3 with oQ8e's 8-bit dense modules and the banks. Each folder gets two nslm-qwen4-gen runs, then a KL as shipped. R5 comes after it: Matt's upload job to three private repos, then your download-and-run test. I will post the commands. The PC is free until then.
