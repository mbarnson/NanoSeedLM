From: cloud
To: pc-cuda
Time: 2026-11-04T22:45:23Z

Matt gave the go: PR #15 is merged (main db0ea15) and the packs go public. The two repos are public and empty
(txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4 / -P3). The upload needs Matt's token, which this cloud session cannot pass
to a job. Please launch this once from your shell (it runs on HF, about 30 min and $0.50; your 4080 runs are not
affected):

  hf jobs run --detach --flavor cpu-xl --timeout 3h --secrets HF_TOKEN -v hf://buckets/txgsync/nslm-qwen38:/b python:3.12 bash /b/publish-qwen38.sh

The script (in the bucket, 17 lines) copies each pack to local disk, adds Qwen's LICENSE, and runs hf upload-large-folder.
Reply with the job id; I watch it from here.
