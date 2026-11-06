# The LFSR code against 1MAD at L = 12: weight error (2026-11-05)

**Result.** At L = 12, a trellis code from NanoSeedLM's LFSR has the weighted error of QTIP's 1MAD code within 0.6%
at K = 3. At K = 4 it is within 0.2% for `gate_proj` and `up_proj`, and 1.5% higher for `down_proj`. So gate LB of
the [LFSR plan](../plans/2026-11-05-lfsr-trellis.md) passes at K = 3 and misses by 0.5 points at K = 4 for
`down_proj`, on a small sample.

**Status.** Current. Step L2 of the LFSR plan. Weight error only.

## Setup

- Rows: the rows of the [trellis probe](2026-11-05-trellis-probe.md): layer 20, experts 0 to 3, with the imatrix
  weights of that page.
- Tool: the probe's encoder `tq` (a scratch tool of that page), L = 12, T = 256 for `gate_proj` and `up_proj`, T = 320
  for `down_proj`.
- Codes: 1MAD, x = 34038481 s + 76625530, or the LFSR, x = `lfsr_stream32(s)` (`nslm/lfsr.h`). In both, c(s) is the
  byte sum of x, scaled to zero mean and unit variance over the 2^L states.
- Error: the imatrix-weighted relative error (AW) and the plain relative error, as on the probe page.
- Machine: the cloud container's CPU, 8 threads.

## Results

| Projection | K | 1MAD, AW | LFSR, AW | LFSR against 1MAD | 1MAD, plain | LFSR, plain |
|---|---|---|---|---|---|---|
| `gate_proj` and `up_proj` | 3 | 0.01792 | 0.01786 | -0.3% | 0.01993 | 0.01989 |
| `gate_proj` and `up_proj` | 4 | 0.00493 | 0.00492 | -0.2% | 0.00545 | 0.00546 |
| `down_proj` | 3 | 0.01733 | 0.01743 | +0.6% | 0.02026 | 0.02042 |
| `down_proj` | 4 | 0.00470 | 0.00477 | +1.5% | 0.00548 | 0.00556 |

## Notes

- The sample is 4 experts of one layer. A difference of 1% may be inside its noise. The KL of a whole model (step L3)
  decides.
- The probe scales c(s) in floating point. The format uses the integer byte sum less 510 with a BF16 scale for each
  block, so its errors differ a little from these.
- At L = 16 the LFSR code had 0.6% to 1.7% more error ([the trellis probe](2026-11-05-trellis-probe.md)). At L = 12
  the gap is smaller or negative.
