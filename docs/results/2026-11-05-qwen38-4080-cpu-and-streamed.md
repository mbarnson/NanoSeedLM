# The RTX 4080 PC: CPU seed rate and streamed experts for Qwen3.8-Flash-Next (2026-11-05)

**Result.** The PC's Ryzen 7 5800X3D computes P3 seed experts for one token at 20.7 G weights per second on 16
threads. With a CPU share of the missed experts, decode may gain at most about 6 ms of 34 ms a token, before RAM
bandwidth contention. That is too small for the code it needs now, so the CPU expert path stays on hold on this PC.
Streamed experts (`--stream-experts`) decode P3 at 14.7 to 19.8 tokens per second with a warm page cache.

**Status.** Current. The CPU numbers come from one matrix that fits in the 96 MB L3 cache. Misses from DRAM can only
be slower.

## Setup

- Machine: pc-cuda's PC with Windows and an RTX 4080 (16 GB, PCIe 4). CPU: Ryzen 7 5800X3D (8 cores, 16 threads,
  AVX2). RAM: 4 x 32 GB DDR4-3200 in two channels (51.2 GB/s peak). During the CPU runs, the GPU was idle, and only the
  desktop ran.
- CPU: `harness/nslm-cpu-seed-bench.c` and `nslm/lib_cpu_seed4.c` at `35f17e0`, built with clang 23.1.5 (`-O3
  -march=native`). The bench times one Qwen3.8-Flash-Next decode layer: 10 experts of 640 x 2560, 49.2 M weights.
- Streamed: `ecd552f` with `--prompt-file` (`c9512f9`), P3, `--expert-cache-gb 6`, `NSLM_VRAM_RESERVE_MB=2560`,
  `--stream-experts`. VRAM, RAM and power were sampled every 5 s.

```bash
nslm-cpu-seed-bench 1,4,8,16 20 qwen4
nslm-qwen4-gen --model P3 --prompt-file PROMPT --max-tokens 256 --expert-cache-gb 6 --stream-experts
```

## Results

CPU, G weights per second for each token (best of 20):

| Threads | P4, 1 token | P4, 2 tokens | P4, 4 tokens | P3, 1 token | P3, 2 tokens | P3, 4 tokens |
|---|---|---|---|---|---|---|
| 1 | 4.2 | 4.8 | 6.8 | 3.9 | 6.4 | 8.9 |
| 4 | 15.5 | 18.3 | 25.2 | 13.9 | 23.1 | 32.6 |
| 8 | 21.0 | 24.8 | 36.3 | 18.2 | 30.5 | 47.4 |
| 16 | 21.2 | 28.0 | 42.9 | 20.7 | 35.4 | 51.8 |

Streamed P3 decode. The md5 of stdout is the same as with pinned experts on each run:

| Run | Prompt | Prompt time | Decode, tokens/s |
|---|---|---|---|
| Short, cold page cache | 63 tokens | 48.2 s | 3.80 |
| 15,934 tokens, first run | 15,934 tokens | 188.5 tokens/s | 17.94 |
| Short, warm | 63 tokens | 7.2 s | 14.68 |
| 15,934 tokens, second run | 15,934 tokens | 571.9 tokens/s | 19.77 |

## Notes

- At 4 threads the PC matches a cloud Xeon to the digit. Past 4 threads it scales poorly: 8 cores give 1.3 times the
  rate of 4, and SMT adds about nothing.
- The estimate: a token misses about 580 M weights. The CPU needs about 28 ms for all of them, and PCIe about 16 ms.
  The best split gives the CPU about 36%, for about 10 ms a token instead of 16. The CPU reads and the PCIe copies use
  the same DRAM, and the PLE reads also need the cores. So the real gain is less than 6 ms.
- Several tokens at once raise the CPU rate (51.8 G weights per second for each token at 4 tokens). Only experts
  that more than one token of an MTP verify window uses get this rate.
- Streamed decode is about half of pinned decode (29.7 tokens/s at cache 6). Each layer copies its missed experts from
  the page cache into a pinned buffer on one host thread before PCIe. RAM classes that fit the pack pin it instead.
- During the streamed runs, VRAM was 11.7 to 13.2 GB of 16.4 GB, and the display stayed on. Free RAM fell from 105 GB
  to about 54 GB as the page cache filled.
