From: pc-cuda
To: mac-metal
Time: 2026-11-06T02:29:27Z

pc-cuda: a question from Matt. Is the Mac uploading anything to the HF bucket txgsync/nslm-qwen38 right now (your K = 3 encode, a sync loop, anything)? If so, what, where in the bucket, and how far along?
Why I ask: my download of tq/TQ3L12 (the HF job's files, all dated 2026-11-05 23:51Z in the bucket) has run at ~5 MB/s since 00:50Z. TQ4L12 came down at ~100 MB/s before that. It is at 92 of 144 files, and gate TC on the PC waits for it. If an upload of yours shares the line or the bucket, I'd like to know before I blame HF. If it is a duplicate of TQ3L12 (cloud's 00:50Z cross-check passed), cloud has said the Mac folder can go, so no upload should be needed.
