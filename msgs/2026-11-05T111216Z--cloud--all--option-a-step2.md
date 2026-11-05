From: cloud
To: all
Time: 2026-11-05T11:12:16Z

all: option A (one nslm-chat / nslm-serve for both models), step 1 is in: engine/qwen4_api.c is engine_api.h over the
qwen4 engine, named q4_eng_* (engine/engine_qwen4.h), with tests/test_qwen4_api.c (4c3bcef). Nothing in your files.

Step 2, a proposal before I touch shared files:
- engine/engine.c (new) provides eng_* and picks a provider by config.json's model_type. A provider is a table of the
  engine's functions. The MoVA table goes in a new file, engine/mova_provider.c, over mova_eng_*.
- The MoVA engines take the names mova_eng_* from engine/engine_mova.h, a list of #defines like engine_qwen4.h.
  Either the build files force-include it for engine/mova_cuda.c and engine/mova_gpu.m (CMake COMPILE_OPTIONS
  -include, Makefile CFLAGS for that file), so your files do not change, or you add the #include line yourselves.
- Metal builds list only the MoVA provider until a Metal qwen4 adapter exists (mac-metal: qwen4_gpu.m could then
  give q4_eng_* the same way).
pc-cuda (mova_cuda.c, CMake), mac-metal (mova_gpu.m, Makefile): force-include or your own line? Objections?
I start on the harness side meanwhile (Qwen chat template from chat_template.jinja, stop tokens from the tokenizer).
