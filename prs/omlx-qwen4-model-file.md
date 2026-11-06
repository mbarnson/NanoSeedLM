## Summary

`_patch_model_file_bypass` (omlx/patches/mlx_vlm_qwen4_exp_compat) drops `config["model_file"]` for every
`qwen4_exp` checkpoint. That is right for the stock `qwen4_exp.py` that published checkpoints ship (it has no
`ModelConfig`, so mlx-vlm 0.7's `custom_model` route fails). But it also drops a checkpoint's *own* architecture
file, so a `qwen4_exp` checkpoint with custom layers can never load, even with Trust Remote Code on. Example:
[txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3), whose
routed experts are SeedLM seed tensors built by its `nanoseedlm_qwen4.py`. Today it fails at load with
`Received 432 parameters not in model: ...switch_mlp.down_proj.exp_bias, ...nibbles, ...seeds`.

The change: drop `model_file` only when it is the stock `qwen4_exp.py`, or absent (the current route for those, so
stock checkpoints such as `Qwen3.8-Flash-Next-oQ4e-mtp` are unchanged). Any other file keeps mlx-vlm's `model_file`
route, which `ensure_model_code_trusted` already gates behind Trust Remote Code.

```diff
-        if str(config.get("model_type", "")).lower() == "qwen4_exp":
+        stock = config.get("model_file", "qwen4_exp.py") == "qwen4_exp.py"
+        if str(config.get("model_type", "")).lower() == "qwen4_exp" and stock:
```

Regression test: `test_qwen4_exp_custom_model_file_keeps_its_module` (fails on main, passes with the fix). The
existing `test_qwen4_exp_model_file_checkpoints_resolve_to_vendored_module` still passes.

## Validation

Apple M4 Max, 128 GB, macOS 27.2. Python 3.13.5, pytest 9.1.1; mlx 0.32.4, mlx-lm 0.32.3.dev8+g0d4d3d04f, mlx-vlm
0.7.11 (138d7617), the pins of 2165e75. Worktrees of this branch and of 2165e75, each importing its own `omlx`.

- **Regression test** (`pytest tests/test_mlx_vlm_qwen4_exp_compat.py::test_qwen4_exp_custom_model_file_keeps_its_module`):
  fails on 2165e75 with the test file copied in (`module is not qwen4_exp`: the vendored module comes back), passes on
  this branch.
- **Affected files** (`pytest tests/test_mlx_vlm_qwen4_exp_compat.py tests/test_model_loading.py`): 212 passed (211
  on 2165e75, without the new test).
- **Default suite** (`python -m pytest`, `-m "not slow and not integration"`): 19,296 passed, 938 skipped, 98
  deselected, 0 failed, 7 min 52 s.
- **End to end** (`omlx serve`, temperature 0, the settings of a `Qwen3.8-Flash-Next-oQ4e-mtp` entry: SSD PLE, MTP
  with adaptive depth 5, thinking):

| Model | Run | Result | Decode, warm |
|---|---|---|---|
| NanoSeedLM-P3 (custom `model_file`), Trust Remote Code | MTP off, a code prompt | correct code, 222 tokens | 41 to 46 tok/s |
| NanoSeedLM-P3 | MTP on, the same prompt, 3 runs | byte-identical to MTP off; drafts accepted 131/138, 159/169, 158/168 | 85.5, 85.0 tok/s |
| NanoSeedLM-P3 | MTP on, thinking on | `reasoning_content` and `content` split correctly | 42 to 48 tok/s |
| NanoSeedLM-TQ4 (custom `model_file`, trellis experts; a local copy of [txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4) with its `model_file` and loader files) | MTP off, then on, the code prompt | MTP on byte-identical to MTP off; drafts accepted 94% to 95% | 49.0 off, 83.6 to 84.0 on |
| Qwen3.8-Flash-Next-oQ4e-mtp (no `model_file`) | MTP on, the code prompt, 2 runs | byte-identical to its output on 2165e75 | 101.6 tok/s |

Both models log "Qwen4-Exp PLE mode: mmap" and "Lightning MTP enabled". Not tested: vision input, streaming,
prefix-cache reuse, concurrent requests, long contexts, the macOS app.

## Checklist

- [ ] I read [CONTRIBUTING.md](https://github.com/jundot/omlx/blob/main/docs/CONTRIBUTING.md).
- [ ] I understand every change in this PR and can explain it in review.
- [ ] For UI or UX changes, I attached before and after screenshots. (No UI change.)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01XpZd8fMJG2A39rzxjsrHHG
