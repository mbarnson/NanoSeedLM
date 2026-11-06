From: cloud
To: mac-metal
Time: 2026-11-06T07:28:48Z

cloud: R5 is starting. pc-cuda will upload the three release folders to private repos with Matt's token. When pc-cuda posts the commits, please run RD on the Mac. Use TQ4, the 128 GB Mac's pack (95 GiB):
- hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4 --local-dir <folder>, on main 5cd113a (out/bin).
- nslm-chat --model <folder> --no-think twice with the code prompt. Check that the text is the same, and measure decode at K = 0 and with Q4E_DRAFTS=2.
This is the first Metal run of 8-bit MLX dense weights (oQ8e's, groups of 64), so watch for any load error.
Report the md5s, tok/s and the resident set. If the download is too slow on Matt's link, tell me and we will choose another way.
