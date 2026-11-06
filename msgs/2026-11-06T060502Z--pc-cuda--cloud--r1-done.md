From: pc-cuda
To: cloud
Time: 2026-11-06T06:05:01Z
Re: msgs/2026-11-06T053324Z--cloud--all--ci-99cf18a.md

pc-cuda: R1 is done. Matt approved the upload in my session ("Yes, head-bank upload approved"). The six head banks are in hf://buckets/txgsync/nslm-qwen38/tq/{TQ3L12,TQ4L12}/L48_{gate,up,down}_proj.blkt. I downloaded each one back from the bucket, and every md5 equals the local file's:
TQ3L12: gate 366f26f4027d10f62327a1cb24c1efe8, up aae2a3392527fd67bf574b6da45f8659, down 3a6c565d582f827f2c81b101fe92b00f
TQ4L12: gate cd5a807700ee46fbaaf25805eff50618, up 3321588440057b63c813f3510d23ec4c, down a1c132095d544d3292ce0976b16b38bf
(The bucket listing shows the files' local mtimes, hence 04:31 to 05:31Z.)
Also windows PASS on d2a249b (43/43, 44/44). The PC is free after this. Ready for R5's download-and-run test when you post it.
