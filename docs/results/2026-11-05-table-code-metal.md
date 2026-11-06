# Trellis codes from tables on the M4 Max: slower than the computed 1MAD (2026-11-05)

**Result.** On the M4 Max, the trellis gathers are slower with a table code than with the computed 1MAD code. A
4,096-value table in threadgroup memory takes 84 us a call against 59 us. An LFSR code read through its linearity, two
64-value tables and an XOR, takes 95 us, about 60% more. In decode that is about 3 ms more of a 19.3 ms token, about
-15%. So gate LA of the [LFSR plan](../plans/2026-11-05-lfsr-trellis.md) fails on Metal.

**Status.** Current.

## Setup

- Machine: mac-metal's M4 Max, the Mac quiet. Build of `main` with `7e42e52`'s computed gathers.
- Bench: mac-metal's kernel bench of the trellis gathers, 100 calls back to back, 512 random experts, L = 12. The
  table holds 1MAD's values, so every form gives the same q.
- Calls: 10 pairs of experts (one decode token), gate and up together, and down.

## Results

Microseconds a call, K = 3 / K = 4:

| q from | Gate and up | Down |
|---|---|---|
| The computed 1MAD, with the 255 and the 510 once a block (`main`) | 59.5 / 58.3 | 36.6 / 36.0 |
| The computed 1MAD, exact (`rint`) | 65.8 / 65.3 | 39.3 / 38.0 |
| A 4,096-value half table in threadgroup memory, exact | 83.5 / 84.5 | 60.9 / 61.9 |
| The same table in device memory (8 KB, cached) | 140.3 / 137.6 | 76.2 / 77.1 |
| Two 64-value tables in threadgroup memory, XOR, then the byte sum (an LFSR-like linear code) | 95.3 / 94.5 | 75.9 / 60.0 |
| The two tables in registers, read with `simd_shuffle` | 293.3 / 295.4 | 146.7 / 147.3 |

Calls of 30 pairs scale in the same way.

## Notes

- 32 lanes read the table at random places, so the reads collide in the 32 banks of threadgroup memory.
- A threadgroup loads all 4,096 values. That costs more than the 8 rows x 640 weights of `down_proj` that it decodes.
- Metal has no carry-less multiply, so the linear map of an LFSR needs tables or a loop of XORs.
- The CUDA kernels are not measured with tables.
