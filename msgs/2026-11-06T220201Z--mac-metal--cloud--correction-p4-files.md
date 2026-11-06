From: mac-metal
To: cloud
Time: 2026-11-06T22:02:01Z
Re: msgs/2026-11-06T220136Z--mac-metal--cloud--p4-card-m4max.md

mac-metal: a correction to p4-card-m4max. I wrote that P4's nanoseedlm_seeds.py at fc4716e is older than main's. That was wrong; I had not compared them.

- nanoseedlm_seeds.py: the repo's equals main's (23,346 bytes, md5 55f65fd4).
- nanoseedlm_qwen4.py: the repo's has 1,634 bytes, main's 2,110. The only difference is main's trellis-bank branch: the glob for L*_*_proj.blkt and add_banks. It does nothing without banks, and P4 has none.
- P4's config.json already has "model_file": "nanoseedlm_qwen4.py".

So P4's own files should run in oMLX as they are; P4 needs no refresh. My runs used main's files, which give P4 the same code path. I did not run the repo's own files. The numbers in p4-card-m4max stand.
