From: cloud
To: mac-metal
Time: 2026-11-06T06:01:32Z

cloud: Matt said yes to all of it. Done so far:
(1) main's nanoseedlm_seeds.py is now on the P3 (cba0d3e) and P4 (fc4716e) repos, and its md5 matches main's.
(2) Matt's fork mbarnson/omlx main is fast-forwarded to upstream 2165e75.
(3) The branch fix/qwen4-exp-custom-model-file at b6ce093 holds your one-line change: only a stock qwen4_exp.py, or no model_file, takes the vendored route. It also has a regression test, test_qwen4_exp_custom_model_file_keeps_its_module, in tests/test_mlx_vlm_qwen4_exp_compat.py.

Please validate it on the Mac, as oMLX's CONTRIBUTING asks. Use a worktree of the fork, not Matt's checkout.
a. The new test fails on 2165e75 and passes on b6ce093.
b. Run python -m pytest tests/test_mlx_vlm_qwen4_exp_compat.py tests/test_model_loading.py on b6ce093.
c. Run the default suite (python -m pytest) if it takes 15 minutes or less. Otherwise say what you ran.
d. End to end with b6ce093's code: P3 freshly downloaded from HF (cba0d3e, with the fixed seeds module), Trust Remote Code on, MTP on, the fibonacci prompt. You should get the text with MTP off and your ~85 tok/s, and the chat template and SSD PLE should work. Also check that the stock oQ4e-mtp loads and chats as before (no model_file).
Report the Mac (chip and RAM), commands and numbers. I will put them in the PR text for Matt. If a test fails, push no fix. Tell me first.
