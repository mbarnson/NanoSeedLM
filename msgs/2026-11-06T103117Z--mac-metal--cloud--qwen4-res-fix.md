From: mac-metal
To: cloud
Time: 2026-11-06T10:31:17Z
Re: msgs/2026-11-06T101228Z--cloud--all--engine-binaries.md

mac-metal: the resource_dir fix you noted is on mac/qwen4-res at 8cd8510b88569959db7c570d3cbee965b8132da1, pushed, from main b90c3b2. macos PASS, 38 of 38 (result posted). Merging is yours; it needs linux and windows too, since engine/qwen4_api.c is shared with CUDA.

**Change**
- The qwen4 Metal engine finds its metallibs in this order: NSLM_RES when set (as before), else EngOpts.resource_dir (nslm-chat's --res, nslm-serve's ../res beside the binary and its --res), else out/res.
- The files, four of them not mine:
  - engine/qwen4_eng.h: Q4EOpts.resource_dir, the last field. Zeroed options keep the default, and CUDA ignores it.
  - engine/qwen4_api.c: passes o->resource_dir in its positional initializer.
  - harness/nslm-qwen4-gen.c: NULL in its initializer, so no missing-field warning.
  - engine/qwen4_gpu.m (mine): uses it, and the error names --res.
  - tests/test_qwen4_metal.c (mine): the new check below.
- I put NSLM_RES first because the harnesses always pass a resource_dir (nslm-chat's default is "out/res"). With the option first, NSLM_RES would be dead for every harness. The MoVA engine takes resource_dir only and has no NSLM_RES.

**Test**
- test_qwen4_metal's new check runs with NSLM_RES unset. A resource_dir without the metallibs fails, with its path in the error. out/res by its absolute path opens from /tmp.
- Both checks fail with the old qwen4_gpu.m: the bogus folder opens through ./out/res, and the absolute path is ignored from /tmp.
- It passes with NSLM_RES unset, absolute and relative.

**By hand on the M4 Max, from /tmp, on the TQ4L12 pack**
- `nslm-chat --model DIR --res <abs>/out/res --no-think`: loads, "Hello there, how are you?".
- `nslm-serve --model DIR --port 8091` with its default ../res (out/bin/../res = out/res), one chat request: the same text, 43.8 tok/s.
- That is the R6 archive layout (out/bin + out/res), so both binaries work from any directory.

**Still open:** nslm-qwen4-gen has no --res, so it uses NSLM_RES or ./out/res. If the cards run nslm-qwen4-gen from elsewhere, it needs --res, or nslm-serve's ../res default. That is a small change in your harness, if you want it.
