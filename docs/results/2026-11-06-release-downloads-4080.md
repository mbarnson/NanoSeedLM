# The release packs from the Hub on the RTX 4080 (2026-11-06)

**Result.** `hf download` of TQ3-D4 and TQ3 gives folders that the CUDA engine runs with the same text in two runs.
On the RTX 4080, the 8-bit dense weights cost speed, not quality. TQ3-D4 decodes at 38.0 tokens per second, against
54.4 with 4-bit dense weights. Two thirds of that loss is VRAM: the expert cache has 23% fewer slots. The token
embedding in host memory gives back 320 slots and 4.3% (39.6 tokens per second), with the same text.

**Status.** Current. Step R5 of the [release plan](../plans/2026-11-05-release-family.md), gate RD on the PC. The Mac's
test of TQ4 follows on its own page.

## Setup

- Repositories (private at the test): `txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3` at `5f2a8599`, `-TQ3-D4` at
  `7e5edb2b`, `-TQ4` at `25339937`. HF job `6aeda68a` (`cpu-xl`, 29 minutes, about $0.49) uploaded them from the
  bucket. Each repository has the file names and sizes of its bucket folder.
- PC: RTX 4080 (16 GB), Windows, the clang-cl builds of `a499194` and `00918f0`. The automatic expert cache and the
  default VRAM reserve. Desktop apps used 33% to 36% of the GPU and 1.4 GB of VRAM, so the absolute speeds carry that.
- Runs: `nslm-qwen4-gen --no-think --max-tokens 128` and 256-token decodes of the code prompt. The two builds ran
  interleaved. The first run of a session is a warm-up and is not counted.

## Results

| Pack | Two runs | Text |
|---|---|---|
| TQ3-D4 | the same (md5 `f8149f3b` at 128 tokens, `16994974` at 256 tokens) | `merge_sorted_lists`, with a docstring, type hints and doctests |
| TQ3 | the same (md5 `f8149f3b` at 128 tokens, `52768f8c` at 256 tokens) | the same 128 tokens as TQ3-D4 |

Decode, the code prompt, 256 tokens:

| Pack, dense weights, engine | Slots | Misses / hits | Decode (tokens per second) |
|---|---|---|---|
| TQ3-D4, 4-bit MLX ([the trellis speed page](2026-11-05-vram-reserve-4080.md)) | 4,581 | | 54.4 |
| TQ3-D4, 8-bit MLX, `a499194` | 3,545 | 42,680 / 88,462 | 37.99, 37.99 |
| TQ3-D4, 8-bit MLX, `00918f0` (the embedding in host memory) | 3,865 | 40,095 / 91,047 | 39.43, 39.74, 39.63 |
| TQ3, 8-bit MLX, `00918f0` | 4,277 | 36,623 / 94,036 | 42.46, 43.33 |

A prompt of 3,916 tokens runs at 1,465 tokens per second with `a499194` and 1,468 with `00918f0`.

Where the 8-bit dense weights of TQ3-D4 cost time (`Q4E_PROFILE=1`, one run each, `a499194`):

| Stage | 8-bit dense | 4-bit dense | Change |
|---|---|---|---|
| Router and cache (the misses over PCIe) | 4,370 ms | 3,452 ms | +918 ms |
| Dense projections | 1,649 ms | 1,134 ms | +515 ms |
| Head | 336 ms | 216 ms | +120 ms |
| Experts | 1,201 ms | 1,283 ms | -82 ms |
| Hyper-connections | 1,463 ms | 1,487 ms | -24 ms |

## Notes

- The 8-bit dense weights take about 1,036 slots. The misses rise 21%.
- `00918f0` keeps the token embedding (0.68 GB at 8 bits) in mapped host memory. A token reads one row of 2.6 KB over
  PCIe. The engine test matches the C reference on Linux and Windows.
- The KL of the packs does not change. Step D2a of the [dense plan](../plans/2026-11-05-trellis-dense.md) measures
  which dense kinds can go to 4 bits.
- Each upload sent less than 1 GB: the Hub's storage found the bucket's chunks. `hf upload` of the bucket mount fails
  with EIO, so the upload script copies each folder to the job's disk first.
- The v0.1.0 engine archive for Linux runs TQ3 on Ubuntu 22.04 with no CUDA toolkit (HF job `6aedb3f3`, RTX PRO
  6000): `nslm-chat` answers, and `nslm-serve` answers a code prompt at 112.9 tokens per second.
