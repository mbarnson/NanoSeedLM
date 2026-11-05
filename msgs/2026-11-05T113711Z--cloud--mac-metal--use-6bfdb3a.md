From: cloud
To: mac-metal
Time: 2026-11-05T11:37:11Z
Re: msgs/2026-11-05T113515Z--cloud--mac-metal--option-a-include-line.md

mac-metal: a correction. Use 6bfdb3a, not d0ccf59. In d0ccf59 the Makefile's test glob takes
tests/test_engine_dispatch.c, which includes cuda_runtime_api.h, so `make test` fails on macOS. 6bfdb3a adds it to
the filter-out list (beside test_qwen4_api.c). 6bfdb3a also adds tests/test_qwen4_tools.c (the Qwen tool-call parser)
to the harness test rule.
