Status: FAIL
Branch: pc/mla-seeds-cuda
Commit: 5bda3cb8835066c1959610ce958c91f235c23e08
Platform: macos
Agent: mac-metal
Time: 2026-10-09T03:19:14Z

macOS 27.2, M4 Max, AC power. make test on 5bda3cb: test_mova_kernels FAIL (only in the Metal backend):

```
FAIL tests/test_mova_kernels.c:859: per-head maps SEED4P4 (O 128, I 192, T 37, gate): 46 mismatches
FAIL tests/test_mova_kernels.c:859: per-head maps SEED4P4 (O 128, I 192, T 37, transposed): 7 mismatches
```

Cause: tests/kernel_backend_metal.m ran prompt-row seed maps through k_mm on SEED4P4 (exact weights); the new reference (BF16-decoded weights) is the engine's behaviour. Fixed on the Metal side in 59a0b6c (local, on top of 5bda3cb); all macOS tests pass there.
