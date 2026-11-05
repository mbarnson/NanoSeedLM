# CUDA graphs for Qwen3.8-Flash-Next decode (2026-11-04)

**Result.** The test used an RTX PRO 6000 with an 8 GB expert cache. P3 decodes at 46.5 tokens per second with CUDA
graphs and at 42.1 without them: 10.6% faster. The text and the cache hits are the same with and without the graphs.

**Status.** Current. The test used HF Jobs on Linux. Kernel launches cost more on Windows, so the gain on the RTX 4080
may be larger. That is not measured.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell, 96 GB), job `6aebd2030e0338d1bb804d1a`. The load average
  was 2.7 at the start.
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3).
- Commit `6a470c0`, two runs with graphs and two with `NSLM_NO_GRAPH=1`, in turn:

```bash
nslm-qwen4-gen --model PACK --prompt "Why do tide pools matter? Answer in three sentences." --max-tokens 256 --expert-cache-gb 8
```

## Results

19 prompt tokens and 161 generated tokens (the model stopped at EOS) in each run.

| CUDA graphs | Decode, tokens/s | Cache hits / misses |
|---|---|---|
| On | 46.40 / 46.65 | 56,223 / 25,634 |
| Off | 42.08 / 42.10 | 56,223 / 25,634 |

## Notes

- A decode token launches about 1,740 kernels. The kernels now read the rows' position from device memory, so one graph
  for each row count (up to 8) serves every position.
- The four texts have the same MD5 (`ba74ae8b`). The engine test checks decode rows with and without graphs, bit for bit.
- The 4080 profile of `5ed8d1b` put 15% of the decode time in the hyper-connections, a stage of many small kernels.
