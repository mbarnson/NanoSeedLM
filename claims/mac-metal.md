# mac-metal claims (updated 2026-10-08T23:28:31Z)

- **Local on `batch`, not pushed yet (waiting for Matt's OK to push):** on top of 6120fa4,
  1. Metal MLA: prompts of 256+ new tokens expanded per head (`k_mla_prefill`), shorter prefills absorbed
     (`EngOpts.mla_expand_min`, expansion only up to the last multiple of it); `q_lat` read transposed in place.
  2. nslm-serve: wait for a slot that is being saved to the cold cache when it shares a block more of the prompt.
  3. MLA KV caches `--kv fp8` / `--kv fp4` on Metal: `nslm/kvq.h` (C reference codec), `tests/test_kvq.c`,
     `engine_api.h` (`ENG_KV_FP8`, `ENG_KV_FP4`, `eng_kv_parse`), harness `--kv` parsing, server slot sizing,
     kernel tests, test_engine. Shared files touched: `engine/engine_api.h`, `nslm/kvq.h`, `harness/*.c` (--kv),
     `tests/test_engine.c`, `tests/test_mova_kernels.c`, `tests/kernel_backend.h`, `README.md`. Also two small edits in
     your files (yours to keep or redo): `engine/mova_cuda.c` refuses fp8/fp4 by name, `tests/kernel_backend_cuda.c`
     has return-1 stubs for the new kt_* entry points.
- **Next (Metal only):** MLA decode attention efficiency at long context (`k_mla_attn`: about 15% of the M4 Max's
  arithmetic at 8k; your tensor-core kernel shows how much is there).
