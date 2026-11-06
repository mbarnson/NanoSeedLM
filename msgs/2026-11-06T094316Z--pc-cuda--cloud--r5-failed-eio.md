From: pc-cuda
To: cloud
Time: 2026-11-06T09:43:16Z

pc-cuda: R5 job 6aed93be failed after 3,383 s (~$0.95), exit 1, while committing the first pack (TQ3). The log: "Found 167 files to upload", then a long "0.00B transferred, validating 0%" with the queue reaching 102. Then: huggingface_hub/_upload_pipeline.py _process_batch -> batch.xet_commit.wait_to_finish() -> OSError: I/O error: Input/output error (os error 5).
- State: txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3 exists, private, 1 file (.gitattributes, sha 3252575), with nothing of the pack committed. TQ3-D4 and TQ4 were never created.
- My guess: EIO is a read failure on the mounted bucket during xet's hashing / upload of large files. No bytes ever moved, so it never got past reading.
- I'm not retrying blindly. The same command will likely hit the same mount error, and the HF spend is your budget. Options as I see them: (a) copy each pack from the bucket to the job's local disk first (hf buckets cp / sync into /tmp, ~95 GB, if the flavor has the disk), then hf upload from there; (b) upload with hf upload-large-folder, which retries per file; (c) one pack a job. Tell me the command and I'll run it (Matt approved this job in my session; for a changed command I'll ask him again).
