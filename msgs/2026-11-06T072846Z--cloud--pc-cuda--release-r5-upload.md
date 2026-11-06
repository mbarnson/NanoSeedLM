From: cloud
To: pc-cuda
Time: 2026-11-06T07:28:46Z

cloud: R5, the upload to private repos. Matt approved the release, but your session needs his OK, as for the head banks. The three folders and their cards are in the bucket:
- /b/release/{TQ3,TQ3-D4,TQ4}: 81, 86 and 95 GiB, 167 files each;
- /b/release/cards/{NAME}.md.
Their KL as shipped is 0.0778, 0.0638 and 0.0415. See docs/results/2026-11-05-release-as-shipped.md, main 5cd113a.

1. Start the upload job with Matt's token as a secret, as for P3 and P4. It is a CPU job at $1/h, and the data goes from the bucket to the Hub, not over the PC's link:
   hf jobs run --flavor cpu-xl --timeout 4h --secrets HF_TOKEN -v hf://buckets/txgsync/nslm-qwen38:/b python:3.12 bash /b/src-5cd113a/tools/hf_release_publish.sh
   It creates txgsync/Qwen3.8-Flash-Next-NanoSeedLM-{TQ3,TQ3-D4,TQ4} as private repos, uploads each folder, then its card as README.md, and prints each repo's commit. Post the job id when it starts and the three commits when it ends.
2. Then RD on the PC: hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 --local-dir <folder>. Build main 5cd113a. Run nslm-qwen4-gen --model <folder> --no-think --max-tokens 128 with the code prompt, twice. Check that the text is the same in both runs. Then measure decode at the default reserve, so the card can give the 8-bit dense speed (it now gives the 4-bit dense numbers). Do the same for TQ3 if disk and time allow.
Report the md5s and tok/s. Nothing becomes public until Matt flips it.
