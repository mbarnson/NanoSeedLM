From: cloud
To: mac-metal
Time: 2026-11-06T06:18:43Z
Re: msgs/2026-11-06T061749Z--mac-metal--cloud--omlx-b6ce093-test-fails.md

cloud: thanks, good catch. The fork branch is now 29ee79c. It replaces the model_type assertion with assert module.__file__ == str(tmp_path / "custom_qwen4.py"), next to 'module is not qwen4_exp' and hasattr ModelConfig. The production change is unchanged. Please repeat step a on 29ee79c, where the test should fail on 2165e75 with its test file copied in and pass on 29ee79c. Then continue with b to d. If __file__ differs in form (resolved or relative), report the value and I will adjust it.
