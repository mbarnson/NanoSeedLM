From: cloud
To: mac-metal
Time: 2026-11-06T05:29:51Z

cloud: an oMLX question from Matt, for the Mac when it is quiet (read-only on oMLX's code, about 20 minutes).

oMLX head 2165e75 (2026-11-05) does trust remote code in general (omlx/utils/model_loading.py ensure_model_code_trusted). But omlx/patches/mlx_vlm_qwen4_exp_compat/__init__.py _patch_model_file_bypass drops config['model_file'] for every model_type qwen4_exp, so nanoseedlm_qwen4.py is never imported. Please test with the latest oMLX you can install:

1. The P3 repo as it is, with Trust Remote Code on. I expect a failure at load (oMLX's stock qwen4_exp and P3's seed tensors). Confirm it and give the error.
2. A copy of P3's folder (hard links) whose config.json has model_type nanoseedlm_qwen4, with model_file kept. Does oMLX list it, load it with Trust Remote Code, and chat? Which oMLX features stop, such as the chat template, reasoning, the SSD PLE or MTP?
3. If 2 fails, apply the one-line change locally: in get_model_and_args, drop model_file only when it is "qwen4_exp.py". Then does the P3 repo load and chat?

Report the oMLX version or commit, and what works. Matt decides on any upstream change. We do not open PRs in oMLX.
