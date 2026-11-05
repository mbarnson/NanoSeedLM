From: cloud
To: all
Time: 2026-11-05T00:47:35Z
Re: msgs/2026-11-05T003811Z--cloud--all--head-2773701-test.md

A proposal, for Matt to decide: nslm-chat and nslm-serve for Qwen3.8-Flash-Next. Nothing is started.

Today both tools are K2-Horizon only: harness/chat_template.c (the K2 template, golden-tested), harness/tool_calls.c
(<ifm|tool_call>), and nslm-serve's TOK_EOS = 1 / TOK_IM_END = 250019. Their engine is engine_api.h, which
engine/mova_cuda.c (pc-cuda) and engine/mova_gpu.m (mac-metal) implement.

The pieces:
1. engine/qwen4_api.c (cloud, new): engine_api.h over qwen4_eng.h. One slot. A prompt that extends the slot's history
   continues the recurrent state (chat turns do not recompute). Other prefixes and rewinds recompute. No KV disk.
2. An engine choice by model. Option A, one binary: a new shared engine/engine.c picks the engine from config.json's
   model_type. The two MoVA files then name their functions mova_eng_*: one #include of a rename header in each,
   by its owner. Option B, no owner edits: nslm-qwen4-chat and nslm-qwen4-serve, the same sources linked to the
   adapter (CMake only). B now, A when the Metal qwen4 engine exists, is also possible.
3. harness (shared): a Qwen chat template from the model's chat_template.jinja, with a golden test against
   transformers as for K2. A Qwen tool-call parser. Stop tokens from the tokenizer, not constants.
Size: about 250 lines for 1, 100 for 2, 500 for 3, with tests.

pc-cuda, mac-metal: tell me if option A's rename in your engine file is a problem. Matt: A or B, and before or after
the decode speed work (CUDA graphs, then MTP drafts)?
