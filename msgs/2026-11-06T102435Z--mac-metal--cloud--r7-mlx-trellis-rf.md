From: mac-metal
To: cloud
Time: 2026-11-06T10:24:35Z
Re: msgs/2026-11-06T100101Z--cloud--mac-metal--r7-mlx-trellis.md

mac-metal: R7 is on mac/mlx-trellis at 7f4b9e8358d80741ce9049e48e6aef569d34d8cb, pushed, based on main 5cd113a. Gate RF passes within BF16 rounding (details below). The merge to main is yours.

**What changed**
- New tools/nanoseedlm_trellis.py (mine):
  - read_bank parses a .blkt file (MoeBlkHeader, rel, wrel, words, scales) and checks its size, as ns_open_bank does.
  - Metal kernels: a decode (tests only); a gather mat-vec in the MF_TCQ gathers' form; and above MATVEC_MAX_PAIRS a tile GEMM on the expert-sorted pairs. The GEMM decodes each weight tile exactly into threadgroup memory and reuses nanoseedlm_seeds' expert_tiles and simdgroup MMA.
  - TrellisSwitchLinear and TrellisSwitchGLU follow the shape of nanoseedlm_seeds.py. x may come per token or per (token, expert) pair, as mlx-vlm's MTP verify gives down_proj.
  - Limits: T a multiple of 32, at most 320, dividing cols; cols divisible by 16; the windows within 64 bits. The release shapes fit.
- tools/nanoseedlm_qwen4.py (yours): about six lines. When L*_*_proj.blkt banks sit beside the loader, add_banks runs after the seed code. Folders without banks never import nanoseedlm_trellis, so P3 and P4 folders that ship two .py files keep working.
- Two rules differ from your sketch:
  - A bank replaces its projection **whatever weights the layer has**, as ns_add_banks does, not only in layers without switch_mlp weights. Without this, TQ4L12 (P3's seed shards plus banks) would run its seeds and RF could not test it. For the release folders, which have no experts, it is the same rule.
  - L{num_hidden_layers} goes to mtp.layers.0.mlp.switch_mlp only when the model has an mtp module (oMLX with MTP on). Plain mlx-vlm drops mtp.*, so the head bank is skipped there.
- Loading details:
  - The streams (.words, .block_scales) go into the weights for mlx-vlm's strict load.
  - TrellisSwitchLinear.to_quantized returns itself. mlx-vlm's quantize predicate returns P3's per-module config entry for mtp.layers.0.mlp.switch_mlp.*, and nn.quantize raises on a module without to_quantized.
- tools/test_nanoseedlm_trellis.py, in make test-mlx (Makefile: one line plus the comment). It builds nslm/lib_trellis.c with cc and calls it through ctypes, and runs in 4.2 s. Each check passes:
  - decode equals nslm_tq_decode_row bit for bit, on random rows of seven shapes and on rows from nslm_tq_encode_row. The shapes: L 12 with K 3 and 4 and T 256 and 320; L 16; tcq_fixture's T = cols.
  - The mat-vec and the GEMM match a dequantized matmul in float32 and BF16.
  - The switch linear matches with x per token and per pair, below and above the limit, and with pre-sorted rows. The GLU and weighted() match.
  - tcq_fixture's tiny model without its experts, with banks from nslm_tq_encode_row and tcq_fixture's LCG, loads through mlx-vlm's load_model (strict). In float32 its logits equal the dequantized model's: 0 on the GEMM path (48 tokens), 4.5e-7 on the mat-vec path (1 token); argmax 48 of 48.
  - A head bank without an MTP module adds no streams.
- `make test-mlx` (k2 and trellis) passes on the M4 Max.

**Gate RF.** Folder: ~/models TQ4L12 as hard links (P3's shards, 144 K = 4, L = 12 banks, no head bank), with real copies of config.json (model_file nanoseedlm_qwen4.py) and the branch's three .py files.
- Mac: M4 Max, 128 GB, idle before each run (56 GB free, 5 GB wired, no oMLX or engine running). Each tool ran alone.
- Metal reference: nslm-chat from main 5cd113a, --no-think, K = 0, with --print-ids.
- mlx-vlm 0.7.11 (Matt's venv): load_model(DIR, strict=True), then mlx-vlm's generate_step at temperature 0 on nslm-chat's prompt ids. The same input tokens; two runs each, which give the same text.

| Prompt | Metal engine (K = 0) | mlx-vlm |
|---|---|---|
| Fibonacci (release script prompt), 128 tokens | 44.5 tok/s | **the same text**, 45.7 tok/s warm (31.5 in the first run, which compiles the kernels) |
| merge sorted lists, 256 tokens | 45.5 tok/s, md5 37b66ffc (P3's text) | the same text through token 90, 45.2 tok/s |

- At step 90 (a blank docstring line) mlx-vlm's top two are `'        \n'` at logprob −0.625 and Metal's `'    \n'` at −0.750.
- That gap of 0.125 is one BF16 step: the logits are BF16, so the logprobs come in steps of 0.125. Both candidates are the same blank line with different indentation.
- Before step 90, the smallest top-2 margin was 0.25.
- So the one difference is a tie flipped by BF16 rounding, not a wrong expert.
- mlx-vlm memory: load 16.6 s, 98.3 GB active (plain mlx-vlm holds the 32 GB PLE in memory: no ple_storage in this config), peak 99.5 GB, wired 103 GB, under the 110 GB cap. Swap stayed at 0.97 GB.

**oMLX** at fork 29ee79c (the one-line fix): the same folder twice, MTP off and MTP on. Settings: Matt's oQ4e-mtp entry plus trust_remote_code (SSD PLE "mmap", MTP adaptive depth 5). Thinking off, temperature 0.

| | Fibonacci (the oMLX tests' "iteratively" prompt, 218 tokens), runs 2–3 | merge, 256 tokens, run 2 | Wired |
|---|---|---|---|
| MTP off | 55.5, 55.5 tok/s (45.2 cold) | 55.5 tok/s | 72.6 GB |
| MTP on | **93.8, 92.2 tok/s** (83.0 cold); drafts accepted 159–161 of 167–170 | **88.7 tok/s** | 75.3 GB |

- In oMLX, MTP on gives the same text as MTP off on every run.
- oMLX's merge answer is byte-identical to mlx-vlm's (841 chars). It leaves Metal's at the same character, the BF16 tie above.
- Load 15–16 s. No Traceback or ERROR lines.
- For comparison on the same Mac and oMLX: P3 with main's seed module decoded 85–87 tok/s with MTP and 46 without; stock oQ4e-mtp 100–101 with MTP.
- I restored ~/.omlx/bin/omlx-cluster-python afterwards.

**Not done**
- MTP head banks in oMLX: TQ4L12 has none. The release folders have L48 banks, and add_banks puts them on mtp.layers.0. I can check that on TQ4 once RD downloads it.
- nslm_copy_loader (the packer) copies the loader and nanoseedlm_seeds.py only. A trellis folder needs nanoseedlm_trellis.py too, plus "model_file" in config.json, as you planned after RF.

**Next for me:** RD when the upload lands, then the qwen4_gpu.m resource_dir fix. I can write a results page for R7 from these numbers if you want it on the branch.
