# Qwen3.8-Flash-Next on CUDA: long-context decode 34% faster, the shared expert beside the copies (2026-11-05)

**Result.** After a 15,975-token prompt, P3 decode goes from 43.8 to 58.8 tokens per second on an RTX PRO 6000. The
expert cache is 8 GB. That is the speed of a short prompt. Decode attention now splits a row's keys over 32 blocks of a KV
group, and the QSA top-k selects in 8-bit passes. At short prompts, the shared expert on a second stream adds 2.5%.
The decode rows' KL against the prompt rows is the same: 0.0272.

**Status.** Current.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB, PCIe 5). Load average 1.8 to 10.4 in
  the A/B runs, 13 to 16 in the KL run.
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`.
- Long prompt: "Summarize this text in detail, section by section:" and the first 64,000 characters of the wikitext-2
  test set (15,975 tokens). It takes the chat template and effort low. 128 greedy tokens (`scratchpad/rowsjob/ablong.sh`).
- Short prompt: "Why do tide pools matter? Answer in three sentences.", 92 or 111 greedy tokens (`ab.sh`).
- Each A/B builds both commits on one node and runs them in turn, three times each.
- Kernel times: nsys with CUDA graph nodes (`nsyslong.sh`, job `6aec74bb`, at `fe32045`).
- KL: `nslm-qwen4-kld --ref P3 --model P3 --tokens 4096 --decode 256 --expert-cache-gb 48` (`kldec.sh`, job
  `6aec783c`). The model's last 256 rows go one a forward, through the decode kernels. The reference is P3's prompt rows.

## Results

Decode after the 15,975-token prompt, tokens per second:

| Commit | Change | Run 1 | Run 2 | Run 3 |
|---|---|---|---|---|
| `fe32045` | | 43.77 | 43.77 | 43.73 |
| `5ce032e` | split-key decode attention | 54.84 | 54.84 | 54.86 |
| `5ce032e` (second job) | | 55.14 | 55.17 | 55.17 |
| `d6c0606` | 8-bit QSA top-k | 58.83 | 58.77 | 58.82 |

Decode after the short prompt, tokens per second:

| Commit | Change | Run 1 | Run 2 | Run 3 |
|---|---|---|---|---|
| `0233258` | | 56.59 | 56.57 | 56.58 |
| `fe32045` | the shared expert on a second stream | 58.23 | 58.22 | 57.56 |
| `fe32045` (second job) | | 58.44 | 58.40 | 58.34 |
| `5ce032e` | split-key decode attention | 58.33 | 58.30 | 58.32 |

The two kernels at `fe32045` after the long prompt (nsys, 128 decode steps):

| Kernel | Share of the decode GPU time | µs a call |
|---|---|---|
| `k_q4_attend` (decode attention) | 23.0% | 432.7 |
| `k_q4_qsa_topk` | 6.7% | 126.7 |

KL of P3's decode rows against its prompt rows (the last 256 of 4,096 wikitext tokens):

| Commit | Decode rows, mean KL | All rows, mean KL | Top-1 agreement |
|---|---|---|---|
| `fe32045` | 0.02712 | 0.00170 | 99.73% |
| `601333c` | 0.02722 | 0.00170 | 99.68% |

## Notes

- **Attention.** The old kernel took a block a head, and each warp took one key at a time. Each key was a chain of a
  global load, a dot product and a warp sum. Now a block takes one KV group and one of 32 splits of the keys. Each warp
  takes one head. A tile of 16 keys comes through shared memory once for the 12 heads, and its 16 dot products reduce
  together. A second kernel combines the splits.
- **Top-k.** The old kernel found the k-th largest score in 32 one-bit passes, with a block sum each. Then one thread
  looked for the ties. Now 4 passes count the 256 values of each byte in shared memory. A block-wide prefix takes the
  ties from the highest index. The selection and its order are the same, so the text is the same (md5 `9877f0eaad88`).
- **Shared expert.** In decode it reads only its input, so it starts before the router, on a second stream. It runs
  beside the router, the admit and the PCIe copies. The combine waits for it. The text is the same (md5 `5d12b308dd51`).
- `5ce032e` changes the greedy text: the sum order of the keys changes, and near ties go the other way. The kernel
  test passes at 1e-5 against the C reference. The KL of the decode rows is the same within 0.4%.
- The decode rows differ from the prompt rows by a KL of 0.027. The prompt rows use TF32 attention and BF16 GEMM inputs.
  This difference was there before these commits.
