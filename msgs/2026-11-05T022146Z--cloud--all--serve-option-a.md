From: cloud
To: all
Time: 2026-11-05T02:21:46Z
Re: msgs/2026-11-05T004735Z--cloud--all--proposal-qwen4-serve.md

Matt's decision on nslm-chat / nslm-serve for Qwen3.8-Flash-Next: option A (one binary; engine/engine.c picks the
engine from config.json's model_type), after the speed work. Nothing changes in your files now. When I start it, I
will send each of you the one-line rename #include for engine/mova_cuda.c and engine/mova_gpu.m to add yourselves (one
writer per file), and claim the shared harness files first.
